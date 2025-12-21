from __future__ import annotations

import typing as t
import click
from pydantic import BaseModel


NoneType = type(None)


def _unwrap_annotated(tp: t.Any) -> t.Any:
    origin = t.get_origin(tp)
    if origin is t.Annotated:
        return t.get_args(tp)[0]
    return tp


def _strip_optional(tp: t.Any) -> t.Any:
    tp = _unwrap_annotated(tp)
    origin = t.get_origin(tp)
    if origin is t.Union:
        args = tuple(a for a in t.get_args(tp) if a is not NoneType)
        if len(args) == 1:
            return args[0]
    return tp


def _click_param_for_type(tp: t.Any) -> dict:
    tp = _strip_optional(tp)
    origin = t.get_origin(tp)
    args = t.get_args(tp)

    # Literal -> click.Choice
    if origin is t.Literal:
        return {"type": click.Choice(list(args), case_sensitive=False)}

    if origin in (list, t.Sequence, tuple):
        raise ValueError("Sequence overridable support is not build yet. Build it..")

    if tp is bool:
        return {"type": click.BOOL}
    if tp in (int, float, str):
        return {"type": tp}

    return {"type": str}


def add_click_overides(
    model: type[BaseModel],
    overridables: t.Sequence[str],
):
    """
    Decorator factory:
        @click.command()
        @add_click_overrides(Config, OVERRIDABLES)
        def main(...): ...
    """

    def _decorator(func: t.Callable[..., t.Any]):
        for name in reversed(overridables):
            # always assumed to be in pydantic v2
            anno = model.model_fields[name].annotation
            opt = f"--{name.replace("_", "-")}"
            kwargs = {
                "required": False,
                "default": None,
                "show_default": True,
            }
            kwargs.update(_click_param_for_type(anno))

            func = click.option(opt, name, **kwargs)(func)

        return func

    return _decorator
