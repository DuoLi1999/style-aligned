# Currently only sd15

import torch
import einops
from torch import einsum

import torch.nn.functional as F
import torch.nn as nn


def exists(val):
    return val is not None


def optimized_attention(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head ** -0.5

    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    _ATTN_PRECISION = "fp32"
    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION == "fp32":
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if exists(mask):
        if mask.dtype == torch.bool:
            mask = einops.rearrange(mask, 'b ... -> b (...)') #TODO: check if this bool part matches pytorch attention
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = einops.repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        else:
            sim += mask

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return out


class LoRALinearLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 256, org=None):
        super().__init__()
        self.down = torch.nn.Linear(in_features, rank, bias=False)
        self.up = torch.nn.Linear(rank, out_features, bias=False)
        self.org = [org]

    def forward_old(self, h):
        # breakpoint()
        org_weight = self.org[0].weight.to(h)
        org_bias = self.org[0].bias.to(h) if self.org[0].bias is not None else None
        down_weight = self.down.weight
        up_weight = self.up.weight
        final_weight = org_weight + torch.mm(up_weight, down_weight)
        return torch.nn.functional.linear(h, final_weight, org_bias)

    def forward(self, h):
        # breakpoint()
        org_weight = self.org[0].weight.to(h)
        org_bias = self.org[0].bias.to(h) if self.org[0].bias is not None else None
        down_weight = self.down.weight
        up_weight = self.up.weight
        # final_weight = org_weight + torch.mm(up_weight, down_weight)
        # return torch.nn.functional.linear(h, final_weight, org_bias)
        result = torch.nn.functional.linear(h, org_weight, org_bias)
        result = result + torch.nn.functional.linear(torch.nn.functional.linear(h, down_weight, None), up_weight, None)
        return result


class AttentionSharingUnit(torch.nn.Module):
    def __init__(self, module, frames=2, use_control=False, rank=256):
        super().__init__()

        self.heads = module.heads
        self.frames = frames
        self.original_module = [module]
        q_in_channels, q_out_channels = module.to_q.in_features, module.to_q.out_features
        k_in_channels, k_out_channels = module.to_k.in_features, module.to_k.out_features
        v_in_channels, v_out_channels = module.to_v.in_features, module.to_v.out_features
        o_in_channels, o_out_channels = module.to_out[0].in_features, module.to_out[0].out_features  # [0] Linear, [1] Dropout

        hidden_size = k_out_channels


        self.to_q_lora = [LoRALinearLayer(q_in_channels, q_out_channels, rank, module.to_q) for _ in range(self.frames)]
        self.to_k_lora = [LoRALinearLayer(k_in_channels, k_out_channels, rank, module.to_k) for _ in range(self.frames)]
        self.to_v_lora = [LoRALinearLayer(v_in_channels, v_out_channels, rank, module.to_v) for _ in range(self.frames)]
        self.to_out_lora = [LoRALinearLayer(o_in_channels, o_out_channels, rank, module.to_out[0]) for _ in range(self.frames)]

        self.to_q_lora = torch.nn.ModuleList(self.to_q_lora)
        self.to_k_lora = torch.nn.ModuleList(self.to_k_lora)
        self.to_v_lora = torch.nn.ModuleList(self.to_v_lora)
        self.to_out_lora = torch.nn.ModuleList(self.to_out_lora)

        self.temporal_i = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_n = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.temporal_q = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_k = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_v = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_o = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.control_convs = None

        if use_control:
            self.control_convs = [torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(256, hidden_size, kernel_size=1),
            ) for _ in range(self.frames)]
            self.control_convs = torch.nn.ModuleList(self.control_convs)

        self.control_signals = None

    def forward(self, h, encoder_hidden_states=None, value=None, attention_mask=None, cross_attention_kwargs={}):
        context = encoder_hidden_states
        transformer_options = cross_attention_kwargs

        modified_hidden_states = einops.rearrange(h, '(b f) d c -> f b d c', f=self.frames)

        if self.control_convs is not None:
            context_dim = int(modified_hidden_states.shape[2])
            control_outs = []
            for f in range(self.frames):
                control_signal = self.control_signals[context_dim].to(modified_hidden_states)
                control = self.control_convs[f](control_signal)
                control = einops.rearrange(control, 'b c h w -> b (h w) c')
                control_outs.append(control)
            control_outs = torch.stack(control_outs, dim=0)
            modified_hidden_states = modified_hidden_states + control_outs.to(modified_hidden_states)

        if context is None:
            framed_context = modified_hidden_states
        else:
            framed_context = einops.rearrange(context, '(b f) d c -> f b d c', f=self.frames)

        # framed_cond_mark = einops.rearrange(transformer_options['cond_mark'], '(b f) -> f b', f=self.frames).to(modified_hidden_states)
        # framed_cond_mark = einops.rearrange(torch.Tensor([1., 0.], device=h.device), '(b f) -> f b', f=self.frames).to(modified_hidden_states)
        framed_cond_mark = einops.rearrange(torch.Tensor([1., 0.]), '(b f) -> f b', f=self.frames).to(modified_hidden_states)

        attn_outs = []
        for f in range(self.frames):
            fcf = framed_context[f]

            if context is not None:
                cond_overwrite = transformer_options.get('cond_overwrite', [])
                if len(cond_overwrite) > f:
                    cond_overwrite = cond_overwrite[f]
                else:
                    cond_overwrite = None
                if cond_overwrite is not None:
                    cond_mark = framed_cond_mark[f][:, None, None]
                    fcf = cond_overwrite.to(fcf) * (1.0 - cond_mark) + fcf * cond_mark

            q = self.to_q_lora[f](modified_hidden_states[f])
            k = self.to_k_lora[f](fcf)
            v = self.to_v_lora[f](fcf)
            o = optimized_attention(q, k, v, self.heads)
            o = self.to_out_lora[f](o)
            o = self.original_module[0].to_out[1](o)
            attn_outs.append(o)

        attn_outs = torch.stack(attn_outs, dim=0)
        modified_hidden_states = modified_hidden_states + attn_outs.to(modified_hidden_states)
        modified_hidden_states = einops.rearrange(modified_hidden_states, 'f b d c -> (b f) d c', f=self.frames)

        x = modified_hidden_states
        x = self.temporal_n(x)
        x = self.temporal_i(x)
        d = x.shape[1]

        x = einops.rearrange(x, "(b f) d c -> (b d) f c", f=self.frames)

        q = self.temporal_q(x)
        k = self.temporal_k(x)
        v = self.temporal_v(x)

        x = optimized_attention(q, k, v, self.heads)
        x = self.temporal_o(x)
        x = einops.rearrange(x, "(b d) f c -> (b f) d c", d=d)

        modified_hidden_states = modified_hidden_states + x

        return modified_hidden_states - h


class AttentionSharingUnit_woLoRA(torch.nn.Module):
    def __init__(self, module, frames=2, use_control=False, rank=256):
        super().__init__()

        self.heads = module.heads
        self.frames = frames
        self.original_module = [module]
        q_in_channels, q_out_channels = module.to_q.in_features, module.to_q.out_features
        k_in_channels, k_out_channels = module.to_k.in_features, module.to_k.out_features
        v_in_channels, v_out_channels = module.to_v.in_features, module.to_v.out_features
        o_in_channels, o_out_channels = module.to_out[0].in_features, module.to_out[0].out_features  # [0] Linear, [1] Dropout

        hidden_size = k_out_channels


        self.to_q = [torch.nn.Linear(q_in_channels, q_out_channels, bias=(module.to_q.bias != None)) for _ in range(self.frames)]
        self.to_k = [torch.nn.Linear(k_in_channels, k_out_channels, bias=(module.to_k.bias != None)) for _ in range(self.frames)]
        self.to_v = [torch.nn.Linear(v_in_channels, v_out_channels, bias=(module.to_v.bias != None)) for _ in range(self.frames)]
        self.to_out = [torch.nn.Linear(o_in_channels, o_out_channels, bias=(module.to_out[0].bias != None)) for _ in range(self.frames)]

        # init the weights wity module
        with torch.no_grad():
            for idx_frame in range(self.frames):
                self.to_q[idx_frame].weight.copy_(module.to_q.weight.clone().detach())
                self.to_k[idx_frame].weight.copy_(module.to_k.weight.clone().detach())
                self.to_v[idx_frame].weight.copy_(module.to_v.weight.clone().detach())
                self.to_out[idx_frame].weight.copy_(module.to_out[0].weight.clone().detach())
                if module.to_q.bias != None:
                    self.to_q[idx_frame].bias.copy_(module.to_q.bias.clone().detach())
                if module.to_k.bias != None:
                    self.to_k[idx_frame].bias.copy_(module.to_k.bias.clone().detach())
                if module.to_v.bias != None:
                    self.to_v[idx_frame].bias.copy_(module.to_v.bias.clone().detach())
                if module.to_out[0].bias != None:
                    self.to_out[idx_frame].bias.copy_(module.to_out[0].bias.clone().detach())

        self.to_q = torch.nn.ModuleList(self.to_q)
        self.to_k = torch.nn.ModuleList(self.to_k)
        self.to_v = torch.nn.ModuleList(self.to_v)
        self.to_out = torch.nn.ModuleList(self.to_out)

        self.temporal_i = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_n = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.temporal_q = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_k = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_v = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_o = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.control_convs = None

        if use_control:
            self.control_convs = [torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(256, hidden_size, kernel_size=1),
            ) for _ in range(self.frames)]
            self.control_convs = torch.nn.ModuleList(self.control_convs)

        self.control_signals = None

    def forward(self, h, encoder_hidden_states=None, value=None, attention_mask=None, cross_attention_kwargs={}):
        context = encoder_hidden_states
        transformer_options = cross_attention_kwargs

        modified_hidden_states = einops.rearrange(h, '(b f) d c -> f b d c', f=self.frames)

        if self.control_convs is not None:
            context_dim = int(modified_hidden_states.shape[2])
            control_outs = []
            for f in range(self.frames):
                control_signal = self.control_signals[context_dim].to(modified_hidden_states)
                control = self.control_convs[f](control_signal)
                control = einops.rearrange(control, 'b c h w -> b (h w) c')
                control_outs.append(control)
            control_outs = torch.stack(control_outs, dim=0)
            modified_hidden_states = modified_hidden_states + control_outs.to(modified_hidden_states)

        if context is None:
            framed_context = modified_hidden_states
        else:
            framed_context = einops.rearrange(context, '(b f) d c -> f b d c', f=self.frames)

        # framed_cond_mark = einops.rearrange(transformer_options['cond_mark'], '(b f) -> f b', f=self.frames).to(modified_hidden_states)
        # framed_cond_mark = einops.rearrange(torch.Tensor([1., 0.], device=h.device), '(b f) -> f b', f=self.frames).to(modified_hidden_states)
        framed_cond_mark = einops.rearrange(torch.Tensor([1., 0.]), '(b f) -> f b', f=self.frames).to(modified_hidden_states)

        attn_outs = []
        for f in range(self.frames):
            fcf = framed_context[f]

            if context is not None:
                cond_overwrite = transformer_options.get('cond_overwrite', [])
                if len(cond_overwrite) > f:
                    cond_overwrite = cond_overwrite[f]
                else:
                    cond_overwrite = None
                if cond_overwrite is not None:
                    cond_mark = framed_cond_mark[f][:, None, None]
                    fcf = cond_overwrite.to(fcf) * (1.0 - cond_mark) + fcf * cond_mark

            q = self.to_q[f](modified_hidden_states[f])
            k = self.to_k[f](fcf)
            v = self.to_v[f](fcf)
            o = optimized_attention(q, k, v, self.heads)
            o = self.to_out[f](o)
            o = self.original_module[0].to_out[1](o)
            attn_outs.append(o)

        attn_outs = torch.stack(attn_outs, dim=0)
        modified_hidden_states = modified_hidden_states + attn_outs.to(modified_hidden_states)
        modified_hidden_states = einops.rearrange(modified_hidden_states, 'f b d c -> (b f) d c', f=self.frames)

        x = modified_hidden_states
        x = self.temporal_n(x)
        x = self.temporal_i(x)
        d = x.shape[1]

        x = einops.rearrange(x, "(b f) d c -> (b d) f c", f=self.frames)

        q = self.temporal_q(x)
        k = self.temporal_k(x)
        v = self.temporal_v(x)

        x = optimized_attention(q, k, v, self.heads)
        x = self.temporal_o(x)
        x = einops.rearrange(x, "(b d) f c -> (b f) d c", d=d)

        modified_hidden_states = modified_hidden_states + x

        return modified_hidden_states - h


class AdditionalAttentionCondsEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks_0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 64*64*256

        self.blocks_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 32*32*256

        self.blocks_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 16*16*256

        self.blocks_3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 8*8*256

        self.blks = [self.blocks_0, self.blocks_1, self.blocks_2, self.blocks_3]

    def __call__(self, h):
        results = {}
        for b in self.blks:
            h = b(h)
            results[int(h.shape[2]) * int(h.shape[3])] = h
        return results


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj

