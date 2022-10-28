from typing import Iterable, Union

import torch
import torch.nn.functional as F

from trlx.model import register_model
from trlx.model.nn.ilql_models import CausalLMWithValueHeads

from .accelerate_base_model import AccelerateRLModel


@register_model
class AccelerateILQLModel(AccelerateRLModel):
    def __init__(
        self,
        config,
        accelerator,
        logit_mask=None,
        metric_fn=None,
        train_mode=True,
    ):
        super().__init__(config, train_mode, accelerator)
        self.logit_mask = logit_mask
        self.metric_fn = metric_fn
        self.reward_fn = None
        self.params = config.method

    def get_arch(self, config):
        return CausalLMWithValueHeads(
            config.model.model_path,
            params=config.method,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
        )

    def tokenize(self, texts: Union[Iterable[str], Iterable[torch.LongTensor]]):
        if isinstance(texts[0], torch.LongTensor):
            return texts

        tokenized = self.tokenizer(
            [self.tokenizer.bos_token + x + self.tokenizer.eos_token for x in texts],
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = list(map(torch.as_tensor, tokenized.input_ids))
        return input_ids

    def post_backward_callback(self):
        if self.iter_count % self.config.method.steps_for_target_q_sync == 0:
            self.accelerator.unwrap_model(self.model).sync_target_q_heads()

    def loss(self, batch):
        input_ids = batch.input_ids.to(self.accelerator.device)
        attn = batch.attention_mask.to(self.accelerator.device)
        rewards = batch.rewards.to(self.accelerator.device)
        states_ixs = batch.states_ixs.to(self.accelerator.device)
        actions_ixs = batch.actions_ixs.to(self.accelerator.device)
        dones = batch.dones.to(self.accelerator.device)

        logits, qs, target_qs, vs, _ = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            actions_ixs=actions_ixs,
            states_ixs=states_ixs,
        )

        actions = input_ids[:, 1:].gather(dim=1, index=actions_ixs).unsqueeze(-1)
        bsize, ntokens, dsize = logits.shape

        # compute two separate q-value estimates, to then select minimum values from both
        if self.params.two_qs:
            Q1 = qs[0].gather(-1, actions).squeeze(-1)
            Q2 = qs[1].gather(-1, actions).squeeze(-1)

            targetQ1 = target_qs[0].gather(-1, actions).squeeze(-1).detach()
            targetQ2 = target_qs[1].gather(-1, actions).squeeze(-1).detach()
            targetQ = torch.minimum(targetQ1, targetQ2)
        else:
            Q = qs.gather(-1, actions).squeeze(-1)
            targetQ = target_qs.gather(-1, actions).squeeze(-1).detach()

        terminal_mask = dones[:, :-1]
        n_nonterminal = max(1, terminal_mask.sum())

        # values of current states
        V = vs[:, :-1].squeeze()
        # values of next states
        Vnext = vs[:, 1:].squeeze() * dones[:, 1:]
        # target to fit Q
        Q_ = rewards + self.params.gamma * Vnext.detach()

        if self.params.two_qs:
            loss_q1 = ((Q1 - Q_) * terminal_mask).pow(2).sum() / n_nonterminal
            loss_q2 = ((Q2 - Q_) * terminal_mask).pow(2).sum() / n_nonterminal
            loss_q = loss_q1 + loss_q2
        else:
            loss_q = ((Q - Q_) * terminal_mask).pow(2).sum() / n_nonterminal

        targetQ = targetQ.detach()

        loss_v = (
            (
                (targetQ >= V).int() * self.params.tau * (targetQ - V).pow(2)
                + (targetQ < V).int() * (1 - self.params.tau) * (targetQ - V).pow(2)
            )
            * terminal_mask
        ).sum() / n_nonterminal

        if self.params.two_qs:
            nactions = qs[0].shape[1]
            loss_cql_q1 = (
                F.cross_entropy(
                    qs[0].reshape(-1, dsize),
                    actions.reshape(-1),
                    reduction="none",
                ).reshape(bsize, nactions)
                * terminal_mask
            ).sum() / n_nonterminal
            loss_cql_q2 = (
                F.cross_entropy(
                    qs[1].reshape(-1, dsize),
                    actions.reshape(-1),
                    reduction="none",
                ).reshape(bsize, nactions)
                * terminal_mask
            ).sum() / n_nonterminal
            loss_cql = loss_cql_q1 + loss_cql_q2
        else:
            nactions = qs.shape[1]
            loss_cql = (
                F.cross_entropy(
                    qs.reshape(-1, dsize), actions.reshape(-1), reduction="none"
                ).reshape(bsize, nactions)
                * terminal_mask
            ).sum() / n_nonterminal

        loss_awac = (
            F.cross_entropy(
                logits[:, :-1, :].reshape(-1, dsize),
                input_ids[:, 1:].reshape(-1),
                reduction="none",
            ).reshape(bsize, ntokens - 1)
            * attn[:, 1:]
        ).sum() / attn[:, 1:].sum()

        loss = (
            loss_q
            + loss_v
            + self.params.cql_scale * loss_cql
            + self.params.awac_scale * loss_awac
        )
        stats = {
            f"losses/{k}": v
            for k, v in locals().items()
            if k in ["loss", "loss_v", "loss_q", "loss_cql", "loss_awac"]
        }

        return loss, stats

    def prepare_learning(self):
        train_dataloader = self.store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.opt, train_dataloader, eval_dataloader
        )

        self.n_updates_per_batch = 1
        self.total_steps = self.config.train.epochs * len(train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

        self.generate_kwargs = {
            "beta": self.config.method.betas[0],
            "max_length": self.max_length,
            "logit_mask": self.logit_mask,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else 0,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else 0,
        }
