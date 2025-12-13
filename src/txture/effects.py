from typing import Callable, Mapping, Any, Dict
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class EffectContext:
    mode: str
    outline: bool
    color: bool
    face_label: str | None
    face_conf: float
    gesture_label: str | None
    gesture_conf: float
    flags: Mapping[str, bool] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EffectLayer:
    name: str
    priority: int = 0
    enabled: bool = True
    apply_frame: Callable[[np.ndarray, EffectContext], np.ndarray] | None = (
        None
    )
    apply_ascii: (
        Callable[
            [
                list[str],
                list[list[tuple[int, int, int]]] | None,
                EffectContext,
            ],
            tuple[list[str], list[list[tuple[int, int, int]]] | None],
        ]
        | None
    ) = None


class EffectStack:
    def __init__(self) -> None:
        self._layers: list[EffectLayer] = []

    def add_layer(self, layer: EffectLayer) -> None:
        self._layers.append(layer)
        self._layers.sort(key=lambda layer: layer.priority)

    def apply_frame(self, frame: np.ndarray, ctx: EffectContext) -> np.ndarray:
        out = frame
        for layer in self._layers:
            if (not layer.enabled) or (layer.apply_frame is None):
                continue
            try:
                out = layer.apply_frame(out, ctx)
            except Exception:
                continue
        return out

    def apply_ascii(
        self,
        lines: list[str],
        colors: list[list[tuple[int, int, int]]] | None,
        ctx: EffectContext,
    ) -> tuple[list[str], list[list[tuple[int, int, int]]] | None]:
        out_lines = lines
        out_colors = colors
        for layer in self._layers:
            if (not layer.enabled) or (layer.apply_ascii is None):
                continue
            try:
                out_lines, out_colors = layer.apply_ascii(
                    out_lines, out_colors, ctx
                )
            except Exception:
                continue
        return out_lines, out_colors
