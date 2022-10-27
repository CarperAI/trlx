from typing import Iterable, Union

import torch
import torch.nn.functional as F
import megatron

class ILQL:

    def loss(batch):
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