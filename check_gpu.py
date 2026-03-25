import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpus",
    type=int,
    nargs="+",
    default=[0, 1, 3, 4, 5],
    help="GPU indices to test, e.g. --gpus 0 1 3. Defaults to all.",
)
args = parser.parse_args()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version:    {torch.version.cuda}")

try:
    count = torch.cuda.device_count()
    print(f"Device count:    {count}")
except Exception as e:
    print(f"Device count:    FAILED - {e}")
    count = 0

print()

gpu_ids = args.gpus if args.gpus is not None else list(range(count))

for gpu_id in gpu_ids:
    print(f"=== GPU {gpu_id} ===")

    try:
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"  Name:         {props.name}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
    except Exception as e:
        print(f"  get_device_properties FAILED: {e}")
        print("  Status: FAIL")
        print()
        continue

    try:
        t = torch.randn(2000, 2000, device=f"cuda:{gpu_id}")
        r = t @ t
        print(f"  Matrix multiply OK, shape: {r.shape}")
        del t, r
        torch.cuda.empty_cache()
        print("  Status: PASS")
    except Exception as e:
        print(f"  Matrix multiply FAILED: {e}")
        print("  Status: FAIL")

    print()
