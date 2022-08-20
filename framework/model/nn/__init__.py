import copy

import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoModel, AutoConfig

class QVModel(nn.Module):
    """
    Model that predicts Q values and V function of a state (context)
    """
    def __init__(self, path = "gpt2"):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(path)
        cfg = AutoConfig.from_pretrained(path)
        d_model = cfg.n_embd
        vocab = cfg.vocab_size

        def make_ffn(out_dim):
            return nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, out_dim)
            )

        self.q = make_ffn(vocab)
        self.target_q = make_ffn(vocab)
        self.v = make_ffn(1)

        self.lm_head = nn.Linear(d_model, vocab, bias = False)

    def forward(self, **x):
        h = self.transformer(**x)[0]
        return {
            "lm_out" : self.lm_head(h),
            "q_out" : self.q(h),
            "target_q_out" : self.target_q(h),
            "v_out" : self.v(h)
        }
    
    def sync_target(self, alpha):
        for target, src in zip(self.target_q.parameters(), self.q.parameters()):
            target.data.copy_((alpha * src.data) + (1.0 - alpha) * target.data)
            