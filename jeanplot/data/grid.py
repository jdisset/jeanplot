from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class GridData:
    """Raw grid data from a 2D smoothing pass.

    `values[yi, xi]` = value at `(x_coords[xi], y_coords[yi])`.
    """

    x_coords: np.ndarray
    y_coords: np.ndarray
    values: np.ndarray
    xlims: tuple[float, float]
    ylims: tuple[float, float]
    resolution: int
    input_names: list[str]
    output_name: str
    z_value: float | None = None


def extract_grid_data(
    output_values: np.ndarray,
    xlims: tuple[float, float],
    ylims: tuple[float, float],
    resolution: int,
    input_names: Sequence[str],
    output_name: str,
    z_value: float | None = None,
) -> GridData:
    return GridData(
        x_coords=np.linspace(xlims[0], xlims[1], resolution),
        y_coords=np.linspace(ylims[0], ylims[1], resolution),
        values=output_values.reshape(resolution, resolution),
        xlims=tuple(xlims),
        ylims=tuple(ylims),
        resolution=resolution,
        input_names=list(input_names),
        output_name=output_name,
        z_value=z_value,
    )


def grid_data_to_b64(grids: list[GridData]) -> str:
    """Serialize list of GridData to base64-encoded compressed npz."""
    import base64
    import io
    import json

    arrays: dict[str, np.ndarray] = {}
    meta: list[dict[str, Any]] = []
    for i, gd in enumerate(grids):
        p = f"t{i}_"
        arrays[f"{p}x"] = gd.x_coords.astype(np.float32)
        arrays[f"{p}y"] = gd.y_coords.astype(np.float32)
        arrays[f"{p}v"] = gd.values.astype(np.float32)
        meta.append(
            {
                "xlims": [float(gd.xlims[0]), float(gd.xlims[1])],
                "ylims": [float(gd.ylims[0]), float(gd.ylims[1])],
                "resolution": int(gd.resolution),
                "input_names": list(gd.input_names),
                "output_name": str(gd.output_name),
                "z_value": float(gd.z_value) if gd.z_value is not None else None,
            }
        )
    buf = io.BytesIO()
    np.savez_compressed(buf, _meta=np.array(json.dumps(meta)), _n=np.array(len(grids)), **arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def grid_data_from_b64(b64_string: str) -> list[GridData]:
    """Deserialize list of GridData from base64-encoded npz."""
    import base64
    import io
    import json

    data = np.load(io.BytesIO(base64.b64decode(b64_string)), allow_pickle=False)
    n = int(data["_n"])
    meta: list[dict[str, Any]] = json.loads(str(data["_meta"]))
    return [
        GridData(
            x_coords=data[f"t{i}_x"],
            y_coords=data[f"t{i}_y"],
            values=data[f"t{i}_v"],
            xlims=tuple(m["xlims"]),
            ylims=tuple(m["ylims"]),
            resolution=m["resolution"],
            input_names=m["input_names"],
            output_name=m["output_name"],
            z_value=m.get("z_value"),
        )
        for i, m in enumerate(meta[:n])
    ]
