"""NCU profiling script for NEW flash_mla_sm120 decode kernel (V32, h=128, bs=1)."""
import torch
import sys
sys.path.insert(0, '.')

def quantize_kv_v32(kv_bf16):
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    nb, bs, hk, d = kv_bf16.shape
    kv = kv_bf16.squeeze(2)
    bpt = d_nope + num_tiles * 4 + d_rope * 2  # 656
    result = torch.zeros(nb, bs, bpt, dtype=torch.uint8, device=kv.device)
    for ti in range(num_tiles):
        tile = kv[..., ti*tile_size:(ti+1)*tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = torch.pow(2, torch.clamp_min(amax / 448.0, 1e-4).log2().ceil())
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[..., ti*tile_size:(ti+1)*tile_size] = fp8.view(torch.uint8)
        sb = scale.to(torch.float32).contiguous().view(torch.uint8).reshape(nb, bs, 4)
        result[..., d_nope + ti*4 : d_nope + (ti+1)*4] = sb
    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(nb, bs, d_rope*2)
    result[..., d_nope + num_tiles*4:] = rope
    return result.view(nb, bs, 1, bpt)

d_qk, d_v, topk = 576, 512, 2048
num_blocks, block_size = 64, 64
num_heads, batch_size = 128, 1
sm_scale = d_qk ** -0.5

torch.manual_seed(42)
kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk, device='cuda', dtype=torch.bfloat16) / 10).clamp(-1, 1)
kv_packed = quantize_kv_v32(kv_bf16)

q = torch.randn(batch_size, num_heads, d_qk, device='cuda', dtype=torch.bfloat16) / 10
indices = torch.randint(0, num_blocks * block_size, (batch_size, topk), device='cuda', dtype=torch.int32)
indices[:, -10:] = -1

import flash_mla_sm120

# Warmup
for _ in range(3):
    flash_mla_sm120.sparse_mla_decode_fwd(q, kv_packed, indices, sm_scale, d_v)
torch.cuda.synchronize()

# Profiled iteration
flash_mla_sm120.sparse_mla_decode_fwd(q, kv_packed, indices, sm_scale, d_v)
torch.cuda.synchronize()
print("Done (new kernel)")
