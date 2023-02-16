from transformers import LogitsProcessor, LogitsProcessorList
from transformers import AutoTokenizer
import torch
from typing import List, Optional

"""
We make a logits processor for generation
that will take a list of tokens that we want to allow, 
and sample as if all other logits were -inf.
"""

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# We want just digits and space+digit tokens
GOOD_TOKENS = [idx for token, idx in tokenizer.get_vocab().items() 
    if any(c in token for c in "0123456789") and len(token.replace('Ġ', '')) == 1]
# sorted(GOOD_TOKENS) = [' 0', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

class AllowListLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor`
    Only allow tokens in a list of allowed tokens.
    """

    def __init__(self, allowed_tokens_ids: Optional[List[int]] = None):
        if allowed_tokens_ids is None:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            allowed_tokens_ids = GOOD_TOKENS
        self.allowed_tokens_ids = allowed_tokens_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        #scores[~scores[:, self.allowed_tokens_ids].bool()] = -float("inf")
        mask = torch.zeros_like(scores[0], dtype=torch.bool)
        mask[self.allowed_tokens_ids] = True
        scores[:, ~mask] = -float("inf")
        # print how many scores are not inf
        #print(f"num_scores_not_inf = {torch.sum(scores != -float('inf'))}")   
        #import code; code.interact(local=dict(globals(), **locals()))
        return scores
    
    def __repr__(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return f"{self.__class__.__name__}(allowed_tokens_ids={self.allowed_tokens_ids})\n\n(allowed_tokens={tokenizer.decode(self.allowed_tokens_ids)})"


class AllowListLogitsProcessorList(LogitsProcessorList):
    r"""
    :class:`transformers.LogitsProcessorList`
    Only allow tokens in a list of allowed tokens.
    """

    # This class inherits from LogitsProcessorList, which inherits from list.
    # We override the __init__ method to add the allowed_tokens_ids argument.
    def __init__(self, allowed_tokens_ids: Optional[List[int]] = None):
        super().__init__()
        self.append(AllowListLogitsProcessor(allowed_tokens_ids))
    
        





