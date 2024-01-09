from dataclasses import dataclass

from transformers import BatchEncoding


@dataclass
class DPOElement:
    prompt_tokens: BatchEncoding
    chosen_tokens: BatchEncoding
    rejected_tokens: BatchEncoding


# TODO: Extend to include a concrete class for DPOPreferenceBatch
