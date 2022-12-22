import torch
import numpy as np

from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline.offline_pipeline import ILQLRolloutStorage


def tokenize_dialogue(dialogue, tokenizer, max_length=2048):
    if isinstance(dialogue, str):
        dialogue = [tokenizer.bos_token, dialogue]

    dialogue[-1] += tokenizer.eos_token

    out = []
    ctx_length = max_length
    for phrase in reversed(dialogue):
        tokens = tokenizer(phrase).input_ids[-ctx_length:]
        ctx_length -= len(tokens)
        out.insert(0, tokens)
        if ctx_length == 0:
            break

    if len(out) & 1:
        if sum(map(len, out)) == max_length:
            out[0].pop(0)
        out.insert(0, [tokenizer.bos_token_id])

    return out


@register_orchestrator
class OfflineOrchestrator(Orchestrator):
    """
    Orchestrator that creates a static dataset for offline training
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def make_experience(self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """
        if self.trainer.tokenizer:
            samples = [
                tokenize_dialogue(s, self.trainer.tokenizer, max_length)
                for s in samples
            ]

        all_input_ids = []
        all_actions_ixs = []
        all_states_ixs = []
        all_dones = []
        for sample in samples:
            length = 0
            all_input_ids.append(torch.tensor(sum(sample, [])))
            isoutput = False
            actions_ixs, states_ixs, dones = [], [], []
            for phrase in sample:
                if isoutput:
                    actions_ixs.append(
                        torch.arange(length - 1, length + len(phrase) - 1)
                    )
                    states_ixs.append(
                        torch.arange(length - 1, length + len(phrase) - 1)
                    )

                length += len(phrase)
                isoutput = not isoutput

            states_ixs = torch.hstack((*states_ixs, torch.tensor(length - 1)))
            all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int))
            all_actions_ixs.append(torch.hstack(actions_ixs))
            all_states_ixs.append(states_ixs)

        if self.trainer.tokenizer:
            prompt = self.trainer.tokenizer.decode(
                all_input_ids[0][: all_states_ixs[0][1]]
            )
            response = self.trainer.tokenizer.decode(
                all_input_ids[0][all_states_ixs[0][1] :]
            )
            print("[Sample example]")
            print("Prompt: ", prompt)
            print("Response: ", response)

        lengths = list(map(len, all_input_ids))
        print(f"[Mean length] {np.mean(lengths):.2f} [{min(lengths)}, {max(lengths)}]")
        print(f"[Mean reward] {np.mean(rewards):.2f} [{min(rewards)}, {max(rewards)}]")

        returns = torch.as_tensor(rewards, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-30)

        rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
        for rs, ret in zip(rewards, returns):
            rs[-1] = ret

        attention_mask = [torch.ones(len(x), dtype=int) for x in all_input_ids]

        self.trainer.store = ILQLRolloutStorage(
            all_input_ids,
            attention_mask,
            rewards,
            all_states_ixs,
            all_actions_ixs,
            all_dones,
        )
