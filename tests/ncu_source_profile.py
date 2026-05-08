"""
NCU source-level warp stall analysis for prefill MG kernel.
Collects per-source-line stall samples to identify exact hotspots.
"""

import subprocess
import sys
import os

def run_ncu_source(model_type, d_qk, d_v, topk, num_heads, chunk):
    """Run NCU with source-level sampling."""

    script = f"""
import torch, sys
sys.path.insert(0, '.')
from test_decode import quantize_kv_model1, quantize_kv_v32
import flash_mla_sm120
torch.manual_seed(42)
kv_bf16 = (torch.randn(64,64,1,{d_qk},device='cuda',dtype=torch.bfloat16)/10).clamp(-1,1)
kv = {'quantize_kv_model1' if model_type == 'MODEL1' else 'quantize_kv_v32'}(kv_bf16)
q = torch.randn({chunk},{num_heads},{d_qk},device='cuda',dtype=torch.bfloat16)/10
idx = torch.randint(0,4096,({chunk},{topk}),device='cuda',dtype=torch.int32); idx[:,-10:]=-1
for _ in range(3): flash_mla_sm120.sparse_mla_prefill_fwd(q,kv,idx,{d_qk}**-0.5,{d_v})
torch.cuda.synchronize()
flash_mla_sm120.sparse_mla_prefill_fwd(q,kv,idx,{d_qk}**-0.5,{d_v})
torch.cuda.synchronize()
"""

    print(f"\n{'='*70}")
    print(f"NCU source profile: {model_type} h={num_heads} topk={topk} chunk={chunk}")
    print(f"{'='*70}")

    # Write temp script
    with open('/tmp/ncu_prof.py', 'w') as f:
        f.write(script)

    cmd = [
        'ncu', '--set', 'source',
        '--kernel-name', 'regex:prefill_mg',
        '--launch-count', '1',
        '--source-folders', '/home/scratch.yuasun/codes/sparse_mla_sm120/csrc',
        'python', '/tmp/ncu_prof.py'
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '3'

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)

    # Print key output
    for line in result.stdout.split('\n'):
        if line.strip():
            print(line)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)


if __name__ == '__main__':
    run_ncu_source('MODEL1', 512, 512, 512, 64, 128)
