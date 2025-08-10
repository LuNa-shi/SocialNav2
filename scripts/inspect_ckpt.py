#!/usr/bin/env python3
import argparse
import io
import os
import re
import sys
from collections import OrderedDict
from typing import Any, Iterable, Tuple

try:
    import torch
except Exception as exc:  # pragma: no cover
    print("ERROR: This script requires PyTorch to be installed.")
    print(f"Import error: {exc}")
    sys.exit(1)


def format_tensor_info(t: "torch.Tensor") -> str:
    try:
        shape_str = "x".join(str(s) for s in list(t.shape))
    except Exception:
        shape_str = str(tuple(t.shape))
    dtype = str(t.dtype).replace("torch.", "")
    device = str(t.device)
    numel = t.numel() if hasattr(t, "numel") else "?"
    return f"Tensor(shape=[{shape_str}], dtype={dtype}, device={device}, numel={numel})"


def short_value_repr(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return format_tensor_info(value)
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is not None and isinstance(value, np.ndarray):  # type: ignore
        return f"ndarray(shape={value.shape}, dtype={value.dtype}, size={value.size})"
    if isinstance(value, (str, bytes)):
        s = value if isinstance(value, str) else value.decode(errors="ignore")
        if len(s) > 80:
            s = s[:77] + "..."
        return f"{type(value).__name__}({repr(s)})"
    if isinstance(value, (int, float, bool)):
        return f"{type(value).__name__}({value})"
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, (dict, OrderedDict)):
        return f"{type(value).__name__}(keys={len(value)})"
    return f"{type(value).__name__}"


def is_mapping(obj: Any) -> bool:
    return isinstance(obj, (dict, OrderedDict))


def is_sequence(obj: Any) -> bool:
    return isinstance(obj, (list, tuple))


def should_expand(obj: Any) -> bool:
    return is_mapping(obj) or is_sequence(obj)


def print_line(prefix: str, key: str, value: Any) -> None:
    print(f"{prefix}{key}: {short_value_repr(value)}")


def describe(
    obj: Any,
    prefix: str = "",
    max_depth: int = 4,
    max_items_per_level: int = 50,
    key_filter_regex: str = "",
    current_depth: int = 0,
    path: str = "",
) -> None:
    if current_depth > max_depth:
        return

    key_filter = re.compile(key_filter_regex) if key_filter_regex else None

    def pass_filter(k: str) -> bool:
        if key_filter is None:
            return True
        try:
            return bool(key_filter.search(k))
        except re.error:
            return True

    if is_mapping(obj):
        items = list(obj.items())
        print(f"{prefix}{path or '<root>'}: dict(keys={len(items)})")
        shown = 0
        for k, v in items:
            k_str = str(k)
            if not pass_filter(k_str):
                continue
            if shown >= max_items_per_level:
                print(f"{prefix}  ... ({len(items) - shown} more keys not shown)")
                break
            if should_expand(v) and current_depth < max_depth:
                describe(
                    v,
                    prefix=prefix + "  ",
                    max_depth=max_depth,
                    max_items_per_level=max_items_per_level,
                    key_filter_regex=key_filter_regex,
                    current_depth=current_depth + 1,
                    path=f"{k_str}",
                )
            else:
                print_line(prefix + "  ", k_str, v)
            shown += 1
    elif is_sequence(obj):
        print(f"{prefix}{path or '<root>'}: {type(obj).__name__}(len={len(obj)})")
        count = 0
        for idx, v in enumerate(obj):
            if count >= max_items_per_level:
                print(f"{prefix}  ... ({len(obj) - count} more items not shown)")
                break
            k_str = f"[{idx}]"
            if should_expand(v) and current_depth < max_depth:
                describe(
                    v,
                    prefix=prefix + "  ",
                    max_depth=max_depth,
                    max_items_per_level=max_items_per_level,
                    key_filter_regex=key_filter_regex,
                    current_depth=current_depth + 1,
                    path=f"{path}{k_str}",
                )
            else:
                print_line(prefix + "  ", k_str, v)
            count += 1
    else:
        print_line(prefix, path or "<root>", obj)


def try_load_checkpoint(ckpt_path: str) -> Any:
    # Try with safest options first
    load_kwargs = {"map_location": "cpu"}
    try:
        # PyTorch >= 2.1 supports weights_only
        return torch.load(ckpt_path, weights_only=True, **load_kwargs)  # type: ignore
    except TypeError:
        # Fall back if weights_only not supported
        return torch.load(ckpt_path, **load_kwargs)


def summarize_checkpoint(obj: Any) -> Tuple[int, int]:
    total_tensors = 0
    total_numel = 0

    def visit(x: Any) -> None:
        nonlocal total_tensors, total_numel
        if isinstance(x, torch.Tensor):
            total_tensors += 1
            try:
                total_numel += int(x.numel())
            except Exception:
                pass
            return
        if is_mapping(x):
            for _, v in x.items():
                visit(v)
        elif is_sequence(x):
            for v in x:
                visit(v)

    visit(obj)
    return total_tensors, total_numel


def detect_common_keys(obj: Any) -> Iterable[str]:
    if not is_mapping(obj):
        return []
    keys = set(obj.keys())
    common = []
    for k in [
        "state_dict",
        "model",
        "encoder",
        "policy",
        "optimizer",
        "scheduler",
        "epoch",
        "iter",
        "step",
        "global_step",
        "config",
        "args",
        "metadata",
        "trainer_state",
    ]:
        if k in keys:
            common.append(k)
    return common


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and print the structure of a PyTorch .pth checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint file")
    parser.add_argument("--max-depth", type=int, default=4, help="Max recursion depth")
    parser.add_argument(
        "--max-items", type=int, default=50, help="Max items to show per level"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Regex to filter keys to display (applies to mapping keys)",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default="",
        help="Comma-separated top-level keys to focus on (if present)",
    )
    args = parser.parse_args()

    ckpt_path = os.path.expanduser(args.checkpoint)
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(ckpt_path)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: File not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    try:
        obj = try_load_checkpoint(ckpt_path)
    except Exception as exc:
        print("Failed to load checkpoint with torch.load.\n")
        print(f"Exception: {exc}")
        print("\nIf this is due to missing modules/classes, consider re-running in the original environment.")
        sys.exit(2)

    print("\nTop-level object:")
    print(f"- Type: {type(obj).__name__}")
    if is_mapping(obj):
        print(f"- Keys: {len(obj.keys())}")
        common = list(detect_common_keys(obj))
        if common:
            print(f"- Notable keys present: {', '.join(common)}")
    elif is_sequence(obj):
        print(f"- Length: {len(obj)}")

    tensors, numel = summarize_checkpoint(obj)
    print(f"- Tensors found: {tensors}")
    if tensors > 0:
        print(f"- Total parameters/elements: {numel}")

    print("\nStructure:\n")
    focus_keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if focus_keys and is_mapping(obj):
        subset = {k: obj[k] for k in focus_keys if k in obj}
        if not subset:
            print("Requested focus keys not found. Showing full structure instead.\n")
            describe(
                obj,
                max_depth=args.max_depth,
                max_items_per_level=args.max_items,
                key_filter_regex=args.filter,
            )
        else:
            describe(
                subset,
                max_depth=args.max_depth,
                max_items_per_level=args.max_items,
                key_filter_regex=args.filter,
            )
    else:
        describe(
            obj,
            max_depth=args.max_depth,
            max_items_per_level=args.max_items,
            key_filter_regex=args.filter,
        )


if __name__ == "__main__":
    main()

