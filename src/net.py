import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if self.bias is not None and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent=False)

        return qk_dots + self.bias

class FrameAttention(torch.nn.Module):
    def __init__(self, nheads, inchans, kernel_size):
        super(FrameAttention, self).__init__()
        self.nheads = nheads
        self.kernel_size = kernel_size
        self.inchans = inchans

        self.q_linear = torch.nn.Conv3d(inchans, inchans*nheads, (kernel_size[0], kernel_size[1], 1), bias=False, padding='same')
        self.k_linear = torch.nn.Conv3d(inchans, inchans*nheads, (kernel_size[0], kernel_size[1], 1), bias=False, padding='same')
        self.v_linear = torch.nn.Conv3d(inchans, inchans*nheads, (kernel_size[0], kernel_size[1], 1), bias=False, padding='same')

        self.proj_linear = torch.nn.Conv3d(inchans*nheads, inchans, (kernel_size[0], kernel_size[1], 1), padding='same')

        self.alibi = AlibiPositionalBias(nheads)

        self.head_talk_bs = torch.nn.Linear(nheads*inchans, nheads*inchans, bias=False)
        self.head_talk_as = torch.nn.Linear(nheads*inchans, nheads*inchans, bias=False)

    def forward(self, x):
        #Attention across frames.
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        q = q.permute(0, 1, 4, 2, 3)
        k = k.permute(0, 1, 4, 2, 3)
        v = v.permute(0, 1, 4, 2, 3)

        q_shape = q.shape

        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        v = v.flatten(-2, -1)

        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = self.alibi(attn)
        attn_scores = attn / math.sqrt(q.shape[-1])
        attn_mask = torch.tril(torch.ones_like(attn_scores))
        attn_scores = attn_scores.masked_fill_(attn_mask, -1e18)
        
        attn = self.head_talk_bs(attn_scores.transpose(1, -1)).transpose(1, -1)
        attn = torch.softmax(attn, dim=-1)
        attn = self.head_talk_as(attn.transpose(1, -1)).transpose(1, -1)

        attn = torch.matmul(attn, v)

        attn = attn.unflatten(-1, q_shape[-2:])
        attn = attn.permute(0, 1, 3, 4, 2)

        out = self.proj_linear(attn)


        return out

class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return x * torch.sigmoid(x * self.beta)

class FrameFFN(torch.nn.Module):
    def __init__(self, expansion, inchans, kernel_size):
        super(FrameFFN, self).__init__()
        self.expansion = expansion
        self.kernel_size = kernel_size
        self.inchans = inchans

        self.ffn1_linear = torch.nn.Conv3d(inchans, inchans*expansion, (kernel_size[0], kernel_size[1], 1), padding='same')
        self.ffn2_linear = torch.nn.Conv3d(inchans*expansion, inchans, (kernel_size[0], kernel_size[1], 1), padding='same')
        self.relu = Swish()
    def forward(self, x):
        x = self.ffn1_linear(x)
        x = self.relu(x)
        x = self.ffn2_linear(x)
        return x

class VideoNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x = self.layernorm(x.transpose(1, -1)).transpose(-1, 1)
        return x

class Block(torch.nn.Module):
    #Alibi + Macaron + Talking Heads
    def __init__(self, nheads, expansion, inchans, kernel_size, size):
        super(Block, self).__init__()
        self.nheads = nheads
        self.expansion = expansion
        self.inchans = inchans
        self.kernel_size = kernel_size
        self.size = size
        self.inchans_size = size + (inchans,)

        self.attn = FrameAttention(self.nheads, self.inchans, self.kernel_size)
        self.ffn1 = FrameFFN(self.expansion, self.inchans, self.kernel_size)
        self.ffn2 = FrameFFN(self.expansion, self.inchans, self.kernel_size)

        self.prenorm1 = VideoNorm(self.inchans_size)
        self.postnorm1 = VideoNorm(self.inchans_size)
        self.prenorm2 = VideoNorm(self.inchans_size)
        self.postnorm2 = VideoNorm(self.inchans_size)
        self.prenorm3 = VideoNorm(self.inchans_size)
        self.postnorm3 = VideoNorm(self.inchans_size)

    def forward(self, x):
        res = x

        x = self.prenorm1(x)
        x = self.ffn1(x)
        x = self.prenorm2(x)
        x = x + res; res = x

        x = self.prenorm2(x)
        x = self.attn(x)
        x = self.postnorm2(x)
        x = x + res; res = x

        x = self.prenorm3(x)
        x = self.ffn2(x)
        x = self.postnorm3(x)
        x = x + res; res = x

        return x

class VAEBlock(torch.nn.Module):
    def __init__(self, inchans, kernel_size, size):
        super(VAEBlock, self).__init__()
        self.conv_mu = Block(1,1, inchans, kernel_size, size)
        self.conv_logvar = Block(1,1,inchans,kernel_size, size)

    def forward(self, x):
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)

        eps = torch.randn(*mu.shape)
        stddev = torch.exp(logvar * .5)
        return mu + eps*stddev, -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class PythianEngine(torch.nn.Module):
    def __init__(self,  nheads,
                        expansion,
                        nlayers,
                        inchans=3,
                        outchans=3,
                        kernel_size=(3,3),
                        max_length=16,
                        no_vae=False,
                        size=(256, 256),
                        ):

        super(PythianEngine, self).__init__()
        self.nheads = nheads
        self.expansion = expansion
        self.nlayers = nlayers
        self.inchans = inchans
        self.outchans = outchans
        self.kernel_size = kernel_size
        self.max_length = max_length
        self.no_vae = no_vae
        self.size = size

        assert self.nlayers % 2 == 0, "Number of layers must be even because the layers are split into the encoder and decoder. :)"

        self.encoder = torch.nn.ModuleList([Block(nheads, expansion, inchans, kernel_size, size) for _ in range(self.nlayers // 2)])
        
        if not self.no_vae:
            self.vae_uncertainty = VAEBlock(inchans, kernel_size, size)

        self.decoder = torch.nn.ModuleList([Block(nheads, expansion, inchans, kernel_size, size) for _ in range(self.nlayers // 2)])

        self.head = torch.nn.Conv3d(inchans, inchans, 1)
    def forward(self, x):
        y = x

        for i in range(self.nlayers//2):
            y = self.encoder[i](y)

        if not self.no_vae:
            y, kl = self.vae_uncertainty(y)
        
        for i in range(self.nlayers//2):
            y = self.decoder[i](y)

        y = self.head(y)
        y = torch.sigmoid(y)
        return (y, kl) if not self.no_vae else y

# class PatchAttentionBlock(torch.nn.Module):
#     def __init__(self, inchans, nheads, expansion, patch_size=8):
#         super(PatchAttentionBlock, self).__init__()
#         self.inchans = inchans
#         self.patch_size = patch_size
#         self.expansion = expansion
#         self.nheads = nheads

#         self.spatial_qkv = torch.nn.Linear(self.inchans*patch_size*patch_size, self.inchans*patch_size*patch_size*3*nheads, bias=False)
#         self.temporal_qkv = torch.nn.Linear(self.inchans*patch_size*patch_size, self.inchans*patch_size*patch_size*3*nheads, bias=False)
#         self.spatial_projection = torch.nn.Linear(self.inchans*patch_size*patch_size*nheads, self.inchans*patch_size*patch_size)
#         self.temporal_projection = torch.nn.Linear(self.inchans*patch_size*patch_size*nheads, self.inchans*patch_size*patch_size)

#         self.layer_norm0 = torch.nn.LayerNorm((self.inchans*patch_size*patch_size,))
#         self.layer_norm1 = torch.nn.LayerNorm((self.inchans*patch_size*patch_size,))
#         self.layer_norm2 = torch.nn.LayerNorm((self.inchans*patch_size*patch_size,))
#         self.layer_norm3 = torch.nn.LayerNorm((self.inchans*patch_size*patch_size,))
        
#         self.sffn1 = torch.nn.Linear(self.inchans*patch_size*patch_size, self.inchans*patch_size*patch_size*expansion)
#         self.sswish = Swish(self.inchans*patch_size*patch_size*expansion)
#         self.sffn2 = torch.nn.Linear(self.inchans*patch_size*patch_size*expansion, self.inchans*patch_size*patch_size)

#         self.tffn1 = torch.nn.Linear(self.inchans*patch_size*patch_size, self.inchans*patch_size*patch_size*expansion)
#         self.tswish = Swish(self.inchans*patch_size*patch_size*expansion)
#         self.tffn2 = torch.nn.Linear(self.inchans*patch_size*patch_size*expansion, self.inchans*patch_size*patch_size)
#     def forward(self, x):
#         B, C, H, W, T = x.shape

#         time_splitted = []
#         for i in x.split(1, -1):
#             i = torch.nn.functional.unfold(i.squeeze(-1), (self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size)).transpose(1, 2)
#             time_splitted.append(i)
#         x = torch.stack(time_splitted, dim=2)

#         q, k, v = torch.chunk(self.spatial_qkv(x), 3, -1)

#         q = q.view(B, q.shape[1], T, self.nheads, q.shape[-1]//self.nheads)
#         k = k.view(B, k.shape[1], T, self.nheads, k.shape[-1]//self.nheads)
#         v = v.view(B, v.shape[1], T, self.nheads, v.shape[-1]//self.nheads)

#         attention_scores = torch.matmul(q, k.transpose(-1, -2))
#         attention_scores = attention_scores / (q.shape[-1] ** .5)
#         attention_scores = torch.softmax(attention_scores, dim=-1)
#         out = torch.matmul(attention_scores, v)
#         out = out.view(B, q.shape[1], T, -1)

#         out = self.spatial_projection(out)

#         out_x = self.layer_norm0(out + x)

#         out = self.sffn1(out_x)
#         out = self.sswish(out)
#         out = self.sffn2(out)

#         out = self.layer_norm1(out + out_x)

#         x_out = out.transpose(1, 2)
        
#         q, k, v = torch.chunk(self.temporal_qkv(x_out), 3, -1)

#         q = q.view(B, T, q.shape[2], self.nheads, q.shape[-1]//self.nheads)
#         k = k.view(B, T, q.shape[2], self.nheads, k.shape[-1]//self.nheads)
#         v = v.view(B, T, q.shape[2], self.nheads, v.shape[-1]//self.nheads)

#         attention_scores = torch.matmul(q, k.transpose(-1, -2))
#         attention_scores = attention_scores / (q.shape[-1] ** .5)
#         mask = torch.ones_like(attention_scores).tril()
#         attention_scores = attention_scores.masked_fill(mask == 0, -1e18)
#         attention_scores = torch.softmax(attention_scores, dim=-1)
#         out = torch.matmul(attention_scores, v)
#         out = out.view(B, T, q.shape[2], -1)


#         out = self.temporal_projection(out)

#         out_x = self.layer_norm2(out + x_out)

#         out = self.tffn1(out_x)
#         out = self.tswish(out)
#         out = self.tffn2(out)

#         out = self.layer_norm3(out + out_x)


#         out = out.transpose(1, 2)
#         out = torch.split(out, 1, 2)
#         images = []
#         for i in out:
#             i = torch.nn.functional.fold(i.squeeze(2).transpose(1, 2), (H,  W), (self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
#             images.append(i)
        
#         out = torch.stack(images, dim=-1)
#         return out

if __name__ == '__main__':
    patch_attn = PythianEngine(4, 4, 4)
    print(patch_attn(torch.randn((1,3,256,256,16)))[0].shape)