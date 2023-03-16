from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from trlx.data.method_configs import MethodConfig, register_method
from trlx.models.modeling_base import PreTrainedModelWrapper
from trlx.utils.modeling import (
    flatten_dict,
    get_tensor_stats,
)


@dataclass
@register_method
class MRTConfig(MethodConfig):
    """
    Config for MRT method

    :param ppo_epochs: Number of updates per batch
    :type ppo_epochs: int

    :param num_rollouts: Number  of experiences to observe before learning
    :type num_rollouts: int

    :param init_kl_coef: Initial value for KL coefficient
    :type init_kl_coef: float

    :param target: Target value for KL coefficient
    :type target: float

    :param horizon: Number of steps for KL coefficient to reach target
    :type horizon: int

    :param gamma: Discount factor
    :type gamma: float

    :param lam: GAE lambda
    :type lam: float

    :param cliprange: Clipping range for PPO policy loss (1 - cliprange, 1 + cliprange)
    :type cliprange: float

    :param cliprange_value: Clipping range for predicted values
                            (observed values - cliprange_value, observed values + cliprange_value)
    :type cliprange_value: float

    :param vf_coef: Value loss scale w.r.t policy loss
    :type vf_coef: float

    :param gen_kwargs: Additioanl kwargs for the generation
    :type gen_kwargs: Dict[str, Any]

    :param gen_experience_kwargs: if this is not None, then the experience is generated using this
    :type gen_experience_kwargs: Dict[str, Any]
    """

    ppo_epochs: int
    num_rollouts: int
    chunk_size: int
    # init_kl_coef: float
    # target: float
    # horizon: int
    # gamma: float
    # lam: float
    # cliprange: float
    # cliprange_value: float
    # vf_coef: float
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
        """MRT objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """

        loss = torch.tensor(0.0)
        costs = 1 - rewards

        # Reward component
        if self.ce_loss_weight < 1.0: # if ce_loss_weight is 1.0, then we only use the ce loss
            # We make the assumption here that rewards are scaled to [0,1]
            # lengths = response_masks.sum(dim=-1).float()
            lengths = mask.sum(dim=-1).float()

#             model_outputs = self.model(
#                 input_ids=queries,
#                 decoder_input_ids=responses, # response tokens are already shifted right (start token = pad token)
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
            assert False, 'ce_loss_weight should be 0.0'
            # if parallel_mask is not None:
            #     queries = queries[parallel_mask]
            #     query_masks = query_masks[parallel_mask]
            #     refs = refs[parallel_mask]
            #     ref_masks = ref_masks[parallel_mask]

            # # We should compute the cross entropy with the reference response and not with the generated response
            # model_outputs = self.model(
            #     input_ids=queries,
            #     decoder_input_ids=shift_tokens_right(refs, self.model.config.pad_token_id, self.model.config.decoder_start_token_id),
            #     attention_mask=query_masks,
            #     return_dict=True)
            # ce_loss = F.cross_entropy(
            #     model_outputs.logits.reshape(-1, model_outputs.logits.size(-1)),
            #     refs.reshape(-1),
            #     ignore_index=self.model.config.pad_token_id
            # )

        combined_loss = self.ce_loss_weight * ce_loss + (1 - self.ce_loss_weight) * loss

        stats = dict(
            loss=dict(combined_loss=combined_loss, ce_loss=ce_loss, loss=loss, costs=costs),
        )
        # stats = utils.apply_to_sample(lambda t: t.detach().cpu(), stats)

        return combined_loss, flatten_dict(stats)



"""
    def loss(self, rewards, queries, responses, query_masks, response_masks, refs, ref_masks, parallel_mask=None):
        loss = torch.tensor(0.0)
        costs = 1 - rewards

        # Reward component
        if self.params['ce_loss_weight'] < 1.0: # if ce_loss_weight is 1.0, then we only use the ce loss
            # We make the assumption here that rewards are scaled to [0,1]
            lengths = response_masks.sum(dim=-1).float()

            model_outputs = self.model(
                input_ids=queries,
                decoder_input_ids=responses, # response tokens are already shifted right (start token = pad token)
                attention_mask=query_masks,
                return_dict=True)

            logprobs = logprobs_from_logits(model_outputs.logits[:,:-1,:], responses[:, 1:], mask=response_masks[:, 1:])
            avg_scores = logprobs.sum(dim=-1) / lengths

            # [batch_size, candidate_size]
            avg_scores = avg_scores.view(-1, self.params['candidate_size'])
            costs = costs.view(-1, self.params['candidate_size'])

            probs = F.softmax(avg_scores, dim=1).squeeze(-1)
            loss = (probs * costs).sum()

        # Cross entropy component
        ce_loss = torch.tensor(0.0)
        if self.params['ce_loss_weight'] > 0.0:
            if parallel_mask is not None:
                queries = queries[parallel_mask]
                query_masks = query_masks[parallel_mask]
                refs = refs[parallel_mask]
                ref_masks = ref_masks[parallel_mask]

            # We should compute the cross entropy with the reference response and not with the generated response
            model_outputs = self.model(
                input_ids=queries,
                decoder_input_ids=shift_tokens_right(refs, self.model.config.pad_token_id, self.model.config.decoder_start_token_id),
                attention_mask=query_masks,
                return_dict=True)
            ce_loss = F.cross_entropy(
                model_outputs.logits.reshape(-1, model_outputs.logits.size(-1)),
                refs.reshape(-1),
                ignore_index=self.model.config.pad_token_id
            )

        combined_loss = self.params['ce_loss_weight'] * ce_loss + (1 - self.params['ce_loss_weight']) * loss

        stats = dict(
            loss=dict(combined_loss=combined_loss, ce_loss=ce_loss, loss=loss, costs=costs),
        )
        stats = utils.apply_to_sample(lambda t: t.detach().cpu(), stats)

        return combined_loss, flatten_dict(stats)
"""
