from collections.abc import Mapping, Sequence
import math
import torch



def _is_tensor(v) -> bool:
    return torch.is_tensor(v)

def _safe_item(t: torch.Tensor):
    try:
        return t.item()
    except Exception:
        return None

def _fmt_num(n: float) -> str:
    if n is None:
        return "None"
    if isinstance(n, float) and (math.isnan(n) or math.isinf(n)):
        return str(n)
    # Compact float formatting
    if isinstance(n, (float, int)):
        return f"{n:.6g}" if isinstance(n, float) else str(n)
    return str(n)


def tdebug(
    x,
    *,
    name: str = "x",
    max_list_items: int = 50,
    show_values: bool = False,
    max_values: int = 8,
    stats: str = "basic",  # "off" | "basic" | "full"
) -> str:
    """
    Debug formatter for torch tensors and 1-level containers.

    Supported inputs (1-level):
      - torch.Tensor
      - Mapping[str, Tensor]
      - Sequence[Tensor] (list/tuple)

    Returns:
      - A single string (multi-line) suitable for loguru lazy debug usage.
    """

    def _tensor_line(k: str, t: torch.Tensor) -> str:
        try:
            shape = tuple(t.shape)
            dtype = str(t.dtype).replace("torch.", "")
            device = str(t.device)
            numel = int(t.numel())
            elem = int(t.element_size())
            nbytes = numel * elem

            parts = [
                f"{k}: shape={shape}",
                f"dtype={dtype}",
                f"device={device}",
                f"numel={numel}",
                f"bytes={nbytes}",
            ]

            # Cheap-ish extras
            try:
                parts.append(f"requires_grad={bool(t.requires_grad)}")
            except Exception:
                pass
            try:
                parts.append(f"contig={bool(t.is_contiguous())}")
            except Exception:
                pass

            # Optional stats
            if stats != "off":
                with torch.no_grad():
                    if numel == 0:
                        parts.append("min=None max=None")
                    else:
                        if t.is_complex():
                            v = t.abs()
                            parts.append(f"abs_min={_fmt_num(v.min().item())}")
                            parts.append(f"abs_max={_fmt_num(v.max().item())}")
                            if stats == "full":
                                parts.append(f"abs_mean={_fmt_num(v.mean().item())}")
                                parts.append(
                                    f"abs_std={_fmt_num(v.std(unbiased=False).item())}"
                                )
                        elif t.is_floating_point():
                            finite = torch.isfinite(t)
                            n_finite = int(finite.sum().item())
                            n_nan = int(torch.isnan(t).sum().item())
                            n_inf = int(torch.isinf(t).sum().item())
                            parts.append(f"finite={n_finite}/{numel}")
                            if n_nan or n_inf:
                                parts.append(f"nan={n_nan} inf={n_inf}")

                            # Avoid crashing on all-non-finite tensors
                            if n_finite > 0:
                                tf = t[finite]
                                parts.append(f"min={_fmt_num(tf.min().item())}")
                                parts.append(f"max={_fmt_num(tf.max().item())}")
                                if stats == "full":
                                    parts.append(f"mean={_fmt_num(tf.mean().item())}")
                                    parts.append(
                                        f"std={_fmt_num(tf.std(unbiased=False).item())}"
                                    )
                            else:
                                parts.append("min=None max=None")
                        else:
                            # integer / bool
                            try:
                                parts.append(f"min={_fmt_num(t.min().item())}")
                                parts.append(f"max={_fmt_num(t.max().item())}")
                            except Exception:
                                pass

            # Optional values peek
            if show_values:
                with torch.no_grad():
                    if numel == 1:
                        parts.append(f"value={_fmt_num(_safe_item(t))}")
                    elif numel > 1:
                        flat = t.flatten()
                        n = min(int(flat.numel()), int(max_values))
                        # Use a small slice only; avoids huge prints
                        vals = flat[:n].detach()
                        try:
                            vals_cpu = vals.to("cpu")
                            parts.append(f"values[:{n}]={vals_cpu.tolist()}")
                        except Exception:
                            parts.append(f"values[:{n}]=<unprintable>")

            return " | ".join(parts)
        except Exception as e:
            return f"{k}: <tdebug failed: {type(e).__name__}: {e}>"

    def _format_one_level(obj, obj_name: str) -> str:
        # Tensor
        if _is_tensor(obj):
            return _tensor_line(obj_name, obj)

        # Dict-like
        if isinstance(obj, Mapping):
            items = list(obj.items())
            if len(items) > max_list_items:
                items = items[:max_list_items]
                trunc = True
            else:
                trunc = False

            lines = [f"{obj_name}: Mapping(len={len(obj)})"]
            for k, v in items:
                kk = f"{obj_name}[{k!r}]"
                if _is_tensor(v):
                    lines.append("  " + _tensor_line(kk, v))
                else:
                    lines.append(f"  {kk}: <non-tensor: {type(v).__name__}>")
            if trunc:
                lines.append(f"  ... truncated to first {max_list_items} items")
            return "\n".join(lines)

        # Sequence-like (but avoid treating strings/bytes as sequences)
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            xs = list(obj)
            if len(xs) > max_list_items:
                xs = xs[:max_list_items]
                trunc = True
            else:
                trunc = False

            lines = [f"{obj_name}: Sequence(len={len(obj)})"]
            for i, v in enumerate(xs):
                ii = f"{obj_name}[{i}]"
                if _is_tensor(v):
                    lines.append("  " + _tensor_line(ii, v))
                else:
                    lines.append(f"  {ii}: <non-tensor: {type(v).__name__}>")
            if trunc:
                lines.append(f"  ... truncated to first {max_list_items} items")
            return "\n".join(lines)

        return f"{obj_name}: <unsupported type: {type(obj).__name__}>"

    return _format_one_level(x, name)
