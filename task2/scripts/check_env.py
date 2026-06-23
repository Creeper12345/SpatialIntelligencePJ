#!/usr/bin/env python3
import importlib.util
import json
import sys

REQUIRED = ["torch", "numpy", "wandb", "huggingface_hub", "pyarrow"]
OPTIONAL = ["lerobot", "calvin_env", "calvin_agent"]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    result = {"python": sys.executable, "required": {}, "optional": {}}
    for name in REQUIRED:
        result["required"][name] = has_module(name)
    for name in OPTIONAL:
        result["optional"][name] = has_module(name)

    try:
        import torch
        result["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        result["torch_error"] = repr(exc)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    missing = [name for name, ok in result["required"].items() if not ok]
    if missing:
        print("ERROR missing required packages: " + ", ".join(missing), file=sys.stderr)
        return 1
    if not result["optional"].get("lerobot", False):
        print("ERROR missing lerobot; install it before smoke/train.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
