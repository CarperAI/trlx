from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from trlx.data.method_configs import MethodConfig, register_method
from trlx.utils.modeling import flatten_dict


@dataclass
@register_method
class MRTConfig(MethodConfig):
    """
    Config for MRT method

    :param num_rollouts: Number  of experiences to observe before learning
    :type num_rollouts: int

    :param gamma: Discount factor
    :type gamma: float

    :param gen_kwargs: Additioanl kwargs for the generation
    :type gen_kwargs: Dict[str, Any]

    :param gen_experience_kwargs: if this is not None, then the experience is generated using this
    :type gen_experience_kwargs: Dict[str, Any]
    """

    num_rollouts: int
    chunk_size: int
    num_candidates: int
    ce_loss_weight: float
    scale_reward: Optional[str]
    ref_mean: Optional[float]
    ref_std: Optional[float]
    cliprange_reward: float
    gen_kwargs: dict
    gen_experience_kwargs: Optional[dict] = None

    def loss(
        self,
        logprobs: TensorType["batch_size", "response_size"],
        rewards: TensorType["batch_size", "response_size"],
        mask: TensorType["batch_size", "response_size"],
    ):
        """MRT objective function."""

        # TODO: check if masking is correct

        n = mask.sum()

        loss = torch.tensor(0.0)

        # we make the assumption here that we only care about sequence level rewards only
        rewards = rewards.sum(dim=-1)
        costs = 1 - rewards

        # Reward component
        if self.ce_loss_weight < 1.0:  # if ce_loss_weight is 1.0, then we only use the ce loss
            # We make the assumption here that rewards are scaled to [0,1]
            # lengths = response_masks.sum(dim=-1).float()
            lengths = mask.sum(dim=-1).float()

            #             model_outputs = self.model(
            #                 input_ids=queries,
            #                 decoder_input_ids=responses, # response tokens are already shifted right
            #                 attention_mask=query_masks
            #                 return_dict=True)
            # ,
            avg_scores = logprobs.sum(dim=-1) / lengths

            # [batch_size, candidate_size]
            avg_scores = avg_scores.view(-1, self.num_candidates)
            costs = costs.view(-1, self.num_candidates)

            probs = F.softmax(avg_scores, dim=1).squeeze(-1)
            loss = (probs * costs).sum()

        # Cross entropy component
        ce_loss = torch.tensor(0.0)
        if self.ce_loss_weight > 0.0:
            # TODO: for this to work we need to have some sort of reference
            assert False, "ce_loss_weight should be 0.0"
            # if parallel_mask is not None:
            #     queries = queries[parallel_mask]
            #     query_masks = query_masks[parallel_mask]
            #     refs = refs[parallel_mask]
            #     ref_masks = ref_masks[parallel_mask]

            # # We should compute the cross entropy with the reference response and not with the generated response
            # model_outputs = self.model(
            #     input_ids=queries,
            #     decoder_input_ids=
            # shift_tokens_right(refs, self.model.config.pad_token_id, self.model.config.decoder_start_token_id),
            #     attention_mask=query_masks,
            #     return_dict=True)
            # ce_loss = F.cross_entropy(
            #     model_outputs.logits.reshape(-1, model_outputs.logits.size(-1)),
            #     refs.reshape(-1),
            #     ignore_index=self.model.config.pad_token_id
            # )

        combined_loss = self.ce_loss_weight * ce_loss + (1 - self.ce_loss_weight) * loss

        stats = dict(
            losses=dict(
                total_loss=combined_loss.item(),
                ce_loss=ce_loss.item(),
                mrt_loss=loss.item(),
            ),
            padding_percentage=n / mask.numel(),
        )

        return combined_loss, flatten_dict(stats)
