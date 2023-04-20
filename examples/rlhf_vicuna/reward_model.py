import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained("CarperAI/vicuna-13b-fine-tuned")
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained("CarperAI/vicuna-13b-fine-tuned")
        self.PAD_ID = self.tokenizer.pad_token_id

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = 0
        inference = False

        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }


rw_tokenizer = AutoTokenizer.from_pretrained("CarperAI/vicuna-13b-fine-tuned")
rw_tokenizer.padding_side = "right"
device = torch.cuda.device_count() - 1
rw_model = GPTRewardModel()
rw_model.load_state_dict(torch.load("/mnt/hdd/duyphung/oa_rm_llama.pt")["module"])
rw_model.half().to(device)
rw_model.requires_grad_(False)
rw_model.eval()


@torch.inference_mode()
def get_scores(samples: List[str], batch_size=16):
    scores_list = []
    batch_size = 2
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i : i + batch_size]
        sub_samples = [chosen for chosen in sub_samples]
        encodings_dict = rw_tokenizer(
            sub_samples,
            truncation=True,
            max_length=config.train.seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings_dict["input_ids"].to(rw_device)
        attn_masks = encodings_dict["attention_mask"].to(rw_device)
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
        scores_list.append(sub_scores["chosen_end_scores"])
    scores = torch.cat(scores_list, dim=0)
    return scores
