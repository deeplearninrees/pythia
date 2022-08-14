import torch
import math

def dot_prod_attention(q, k, v, causal):
    attended = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])

    if causal:
        mask = torch.ones_like(attended).tril()
        attended = attended.masked_fill(mask == 0, -1e18)

    attended = torch.softmax(attended, -1)
    return torch.matmul(attended, v)

def multihead_attention(q, k, v, causal, nheads):
    """MultiheadAttention
    :param q: (batch_size, seqlen, nheads*hidden_size)
    :param k: (batch_size, seqlen, nheads*hidden_size)
    :param v: (batch_size, seqlen, nheads*hidden_size)
    :param causal: bool
    :param nheads: int
    :return: (batch_size, seqlen, nheads*hidden_size)
    """
    batch_size = q.shape[0]
    seq_len = q.shape[1]
    hidden_size = q.shape[2] // nheads

    q = q.view(batch_size, seq_len, nheads, hidden_size)
    k = k.view(batch_size, seq_len, nheads, hidden_size)
    v = v.view(batch_size, seq_len, nheads, hidden_size)

    attn = dot_prod_attention(q, k, v, causal=causal)
    return attn.view(batch_size, seq_len, nheads*hidden_size)

class TemporalAttention(torch.nn.Module):
    def __init__(self, nheads, causal):
        super().__init__()
        self.nheads = nheads
        self.causal = causal

    def forward(self, q, k, v):
        return multihead_attention(q, k, v, self.causal, self.nheads)