"""`panel_from` — turn a drawing function into a Pydantic Panel class.

A drawing function `fn(X, Y, ..., ax, ...)` stays first-class Python.
`panel_from(fn)` introspects its signature and produces a `PlotPanel` subclass
whose Pydantic fields mirror the kwargs and whose `draw(self, ax)` forwards
`{field: getattr(self, field)}` (plus per-`PlotData` slot wiring).

Conventions
-----------
- `ax`, `self` are skipped.
- Parameters whose names appear in `plot_data_keys` (default: `X`, `Y`,
  `input_names`, `output_name`) are routed from `self.plot_data.{x,y,
  input_names,output_name}` at draw time, *not* exposed as Panel fields.
- Parameters whose names already exist on the base (`PlotPanel`) class
  inherit those fields; `draw` forwards `getattr(self, name)`.
- `rescaler=None` falls back to `IdentityRescaler()` exactly once, here.
"""

import inspect
import re
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

import numpy as np
import pydantic

from jeanplot.data import IdentityRescaler, PlotFunctionResult
from jeanplot.panels.base import PlotPanel


def _is_arraylike_annot(annot: Any) -> bool:
    if annot is np.ndarray:
        return True
    origin = get_origin(annot)
    if origin in (Union, types.UnionType):
        return any(a is np.ndarray for a in get_args(annot))
    return False


def _normalize_annot(annot: Any) -> Any:
    return Any if _is_arraylike_annot(annot) else annot


_SENTINEL = {"self", "ax"}
_DEFAULT_PLOT_DATA_KEYS = ("X", "Y", "input_names", "output_name")
_PLOT_DATA_SOURCE = {
    "X": "x",
    "Y": "y",
    "input_names": "input_names",
    "output_name": "output_name",
}


def _derive_name(fn_name: str) -> str:
    parts = re.split(r"[_\W]+", fn_name)
    camel = "".join(p[:1].upper() + p[1:] for p in parts if p)
    if camel.endswith(("1D", "2D", "3D")) or "Panel" in camel:
        return camel + "Panel" if "Panel" not in camel else camel
    return camel + "Panel"


def _check_param_kinds(sig: inspect.Signature, fn_name: str) -> None:
    for pname, p in sig.parameters.items():
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(f"panel_from({fn_name}): *args is not supported (got *{pname})")
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            raise TypeError(f"panel_from({fn_name}): **kwargs is not supported (got **{pname})")
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"panel_from({fn_name}): positional-only params are not supported (got {pname}, /)"
            )


def panel_from(
    fn,
    *,
    name: str | None = None,
    base: type = PlotPanel,
    plot_data_keys: tuple[str, ...] = _DEFAULT_PLOT_DATA_KEYS,
    field_overrides: dict[str, Any] | None = None,
    txt_fn=None,
) -> type:
    sig = inspect.signature(fn)
    _check_param_kinds(sig, fn.__name__)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    base_fields = set(base.model_fields)
    pd_routed = {k for k in plot_data_keys if k in sig.parameters}
    own_fields: dict[str, tuple[Any, Any]] = {}
    inherited: set[str] = set()

    for pname, param in sig.parameters.items():
        if pname in _SENTINEL or pname in pd_routed:
            continue
        if pname in base_fields:
            inherited.add(pname)
            continue
        annot = _normalize_annot(hints.get(pname, Any))
        default = param.default if param.default is not inspect.Parameter.empty else ...
        own_fields[pname] = (annot, default)

    if field_overrides:
        for k, v in field_overrides.items():
            if not isinstance(v, tuple):
                annot = own_fields.get(k, (Any, ...))[0]
                own_fields[k] = (annot, v)
            else:
                own_fields[k] = v

    cls_name = name or _derive_name(fn.__name__)
    cls = pydantic.create_model(cls_name, __base__=base, **own_fields)  # pyright: ignore[reportArgumentType, reportCallIssue]

    fn_param_names = set(sig.parameters) - _SENTINEL
    own_keys = set(own_fields)
    has_rescaler_param = "rescaler" in fn_param_names

    def _gather_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {}
        for pname in fn_param_names:
            if pname in pd_routed:
                pd = self.plot_data
                assert pd is not None, f"{cls_name}: plot_data is required for {fn.__name__}"
                kwargs[pname] = getattr(pd, _PLOT_DATA_SOURCE[pname])
            elif pname in own_keys or pname in inherited:
                kwargs[pname] = getattr(self, pname)
        if has_rescaler_param and kwargs.get("rescaler") is None:
            kwargs["rescaler"] = IdentityRescaler()
        return kwargs

    def draw(self, ax) -> PlotFunctionResult | None:
        if not self.is_drawable:
            return None
        kwargs = _gather_kwargs(self)
        result = fn(ax=ax, **kwargs)
        if isinstance(result, PlotFunctionResult) and result.mappable is not None:
            self._mappable = result.mappable
        return result

    cls.draw = draw

    if txt_fn is not None:
        txt_sig = inspect.signature(txt_fn)
        txt_params = set(txt_sig.parameters) - _SENTINEL
        txt_pd_routed = {k for k in plot_data_keys if k in txt_params}

        def render_txt(self):
            kwargs: dict[str, Any] = {}
            for pname in txt_params:
                if pname in txt_pd_routed:
                    pd = self.plot_data
                    if pd is None:
                        continue
                    kwargs[pname] = getattr(pd, _PLOT_DATA_SOURCE[pname])
                elif hasattr(self, pname):
                    kwargs[pname] = getattr(self, pname)
            return str(txt_fn(**kwargs))

        cls.render_txt = render_txt

    cls.__panel_fn__ = fn
    if txt_fn is not None:
        cls.__panel_txt_fn__ = txt_fn

    cls.model_rebuild(force=True)
    return cls
