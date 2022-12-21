import torch

from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline.offline_pipeline import ILQLRolloutStorage


@register_orchestrator
class OfflineOrchestrator(Orchestrator):
    """
    Orchestrator that creates a static dataset for offline training
    """

    def __init__(self, trainer, split_token=None):
        self.trainer = trainer
        self.split_token = split_token

    def make_experience(self, samples, rewards):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the model
        """
        if self.trainer.tokenizer:
            input_ids = self.trainer.tokenize(samples)
        else:
            input_ids = samples

        input_ids = list(map(torch.as_tensor, input_ids))

        states_ixs, actions_ixs = [], []
        dones = []
        for s, s_tok in zip(samples, input_ids):
            # split samples on (prompts, continuations) on a given substring `split_token`
            if self.split_token:
                prompt_str_len = s.index(self.split_token) + len(self.split_token)
                prompt_tok_len = len(
                    self.trainer.tokenizer(s[:prompt_str_len]).input_ids
                )
            # else assume that the prompt is a bos token
            else:
                prompt_tok_len = 1

            # indices of continuations, to mask prompts in loss computation
            a_ixs = torch.arange(prompt_tok_len - 1, len(s_tok) - 1)
            # same continuations but for value computation, with the premise to eventually support interleaved dialog
            s_ixs = torch.arange(prompt_tok_len - 1, len(s_tok))
            # mask continuation's ending
            terminals = torch.ones_like(s_ixs)
            terminals[-1] = 0

            actions_ixs.append(a_ixs)
            states_ixs.append(s_ixs)
            dones.append(terminals)

        if self.trainer.tokenizer:
            prompt = self.trainer.tokenizer.decode(input_ids[0][: states_ixs[0][1]])
            response = self.trainer.tokenizer.decode(input_ids[0][states_ixs[0][1] :])
            print("[Sample example]")
            print("Prompt: ", prompt)
            print("Response: ", response)

        print(f"[Mean reward] {torch.Tensor(rewards).mean():.2f}")
        print(
            f"[Mean sample length] {torch.mean(torch.Tensor(list(map(len, input_ids)))):.2f}"
        )

        returns = torch.as_tensor(rewards, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-30)

        rewards = [torch.zeros(x.shape[0]) for x in actions_ixs]
        for rs, G in zip(rewards, returns):
            rs[-1] = G

        attention_mask = [torch.ones(x.shape[0], dtype=int) for x in input_ids]

        self.trainer.store = ILQLRolloutStorage(
            input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones
        )
