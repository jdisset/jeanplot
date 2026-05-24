"""Overlay primitives (diagonal paths, slice overlays, chords, addition-vs-removal).

Verbatim copy from biocomp `plotutils.py` (lines 952-1203) with biocomp imports
replaced by jeanplot equivalents.
"""

import numpy as np

from jeanplot.plots.smooth_kernel import build_tree, knn_stats


def diagonal_xy(X, angle_deg=45.0):
    X = np.asarray(X)
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    return np.column_stack([c * X[:, 1] - s * X[:, 0], s * X[:, 1] + c * X[:, 0]])


def diagonal_xy_raw(X_lat, rescaler, angle_deg=45.0):
    X_raw = np.asarray(rescaler.inv(X_lat))
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    s_raw = c * X_raw[:, 1] - s * X_raw[:, 0]
    t_raw = s * X_raw[:, 1] + c * X_raw[:, 0]
    return np.column_stack([rescaler.fwd(s_raw), rescaler.fwd(t_raw)])


def diagonal_slice_path_latent(t_raw, s_raw_arr, rescaler, angle_deg=45.0):
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    s_arr = np.asarray(s_raw_arr)
    x1_raw = c * t_raw - s * s_arr
    x2_raw = c * s_arr + s * t_raw
    return np.asarray(rescaler.fwd(x1_raw)), np.asarray(rescaler.fwd(x2_raw))


def plot_diagonal_paths(
    ax, t_raw_values, s_raw_range, rescaler, colors=None, n=400, angle_deg=45.0, line_props=None
):
    s_arr = np.linspace(s_raw_range[0], s_raw_range[1], n)
    line_props = dict(line_props or {})
    for i, t_raw in enumerate(t_raw_values):
        x1_lat, x2_lat = diagonal_slice_path_latent(t_raw, s_arr, rescaler, angle_deg)
        kw = dict(line_props)
        if colors is not None:
            kw["color"] = colors[i]
        ax.plot(x1_lat, x2_lat, **kw)


_SLICE_AXES = ("x", "y", "s", "t")


def slice_panel_args(slice_axis, X_lat, rescaler, slice_values_raw, input_names=None):
    if slice_axis not in _SLICE_AXES:
        raise ValueError(f"slice_axis must be one of {_SLICE_AXES}, got {slice_axis!r}")
    X_lat = np.asarray(X_lat)
    slices_latent = [[float(v)] for v in rescaler.fwd(np.asarray(slice_values_raw))]
    n0, n1 = input_names or ["x", "y"]

    if slice_axis == "x":
        X = X_lat
        names = [n0, n1]
    elif slice_axis == "y":
        X = X_lat[:, [1, 0]]
        names = [n1, n0]
    elif slice_axis == "s":
        X = diagonal_xy_raw(X_lat, rescaler)
        names = [f"({n0} − {n1}) / √2", f"({n0} + {n1}) / √2"]
    else:
        X = diagonal_xy_raw(X_lat, rescaler)[:, [1, 0]]
        names = [f"({n0} + {n1}) / √2", f"({n0} − {n1}) / √2"]
    return {"X": X, "slices_latent": slices_latent, "input_names": names}


def plot_slice_overlay(
    ax,
    slice_axis,
    slice_values_raw,
    rescaler,
    var_range_raw=None,
    colors=None,
    n=400,
    line_props=None,
):
    if slice_axis not in _SLICE_AXES:
        raise ValueError(f"slice_axis must be one of {_SLICE_AXES}, got {slice_axis!r}")
    line_props = dict(line_props or {})

    if slice_axis in ("x", "y"):
        slice_lat = rescaler.fwd(np.asarray(slice_values_raw))
        draw = ax.axhline if slice_axis == "x" else ax.axvline
        for i, v in enumerate(slice_lat):
            kw = dict(line_props)
            if colors is not None:
                kw["color"] = colors[i]
            draw(float(v), **kw)
        return

    if var_range_raw is None:
        raise ValueError(f"var_range_raw required for slice_axis={slice_axis!r}")
    cos45 = sin45 = np.cos(np.deg2rad(45.0))
    var_arr = np.linspace(var_range_raw[0], var_range_raw[1], n)
    for i, fixed_raw in enumerate(slice_values_raw):
        if slice_axis == "s":
            x1_raw = cos45 * fixed_raw - sin45 * var_arr
            x2_raw = cos45 * var_arr + sin45 * fixed_raw
        else:
            x1_raw = cos45 * var_arr - sin45 * fixed_raw
            x2_raw = cos45 * fixed_raw + sin45 * var_arr
        x1_lat = rescaler.fwd(x1_raw)
        x2_lat = rescaler.fwd(x2_raw)
        kw = dict(line_props)
        if colors is not None:
            kw["color"] = colors[i]
        ax.plot(x1_lat, x2_lat, **kw)


def plot_slice_chords(
    ax,
    X,
    Y,
    slices,
    xlims,
    rescaler=None,
    colors=None,
    knn_stats_params=None,
    res=100,
    n_curve=200,
    chord_props=None,
    **_kw,
):
    X = np.asarray(X)
    Y = np.asarray(Y)
    slices = np.asarray(slices)
    knn_stats_params = dict(knn_stats_params or {})
    knn_radius = float(knn_stats_params.get("radius", 0.075))

    xmin = float(X[:, 0].min() if xlims[0] is None else xlims[0])
    xmax = float(X[:, 0].max() if xlims[1] is None else xlims[1])
    xquery_min = max(xmin, float(X[:, 0].min()) + knn_radius * 0.5)
    xquery_max = min(xmax, float(X[:, 0].max()) - knn_radius)
    xq = np.linspace(xquery_min, xquery_max, res)

    tree = build_tree(X)
    nslices = slices.shape[0]
    n_input = X.shape[1]
    chord_props = dict(chord_props or {})

    for i in range(nslices):
        query = xq.reshape(-1, 1)
        if n_input > 1:
            query = np.hstack([query, np.tile(slices[i], (query.shape[0], 1))])
        knn_mean = np.asarray(
            knn_stats(query, Y, tree=tree, stats=["mean"], **knn_stats_params)
        ).reshape(-1)
        finite = np.isfinite(knn_mean)
        if not finite.any():
            continue
        idx = np.where(finite)[0]
        x_lo_lat, x_hi_lat = float(xq[idx[0]]), float(xq[idx[-1]])
        y_lo_lat, y_hi_lat = float(knn_mean[idx[0]]), float(knn_mean[idx[-1]])

        kw = dict(chord_props)
        if colors is not None:
            kw["color"] = colors[i]

        if rescaler is None:
            ax.plot([x_lo_lat, x_hi_lat], [y_lo_lat, y_hi_lat], **kw)
            continue

        x_lo_raw, x_hi_raw = float(rescaler.inv(x_lo_lat)), float(rescaler.inv(x_hi_lat))
        y_lo_raw, y_hi_raw = float(rescaler.inv(y_lo_lat)), float(rescaler.inv(y_hi_lat))
        x_raw = np.linspace(x_lo_raw, x_hi_raw, n_curve)
        t = (x_raw - x_lo_raw) / (x_hi_raw - x_lo_raw)
        y_raw = y_lo_raw + t * (y_hi_raw - y_lo_raw)
        ax.plot(rescaler.fwd(x_raw), rescaler.fwd(y_raw), **kw)


def plot_addition_vs_removal_overlay(
    ax,
    X_lat,
    Y_lat,
    slice_values_raw,
    anchor_raw_values,
    rescaler,
    colors=None,
    knn_stats_params=None,
    max_centroid_offset_frac=0.0,
    line_props=None,
    res=200,
    **_kw,
):
    X_lat = np.asarray(X_lat)
    Y_lat = np.asarray(Y_lat)
    anchor_raw_values = list(anchor_raw_values)
    if not anchor_raw_values:
        return

    knn_stats_params = dict(knn_stats_params or {})
    knn_stats_params.pop("avg_method", None)
    knn_radius = float(knn_stats_params.get("radius", 0.075))
    knn_stats_params["radius"] = knn_radius
    sigma_in_radius = float(knn_stats_params.get("sigma_in_radius", 3.0))
    offset_cutoff = (
        max_centroid_offset_frac * (knn_radius / sigma_in_radius)
        if max_centroid_offset_frac > 0.0
        else None
    )

    line_props = dict(line_props or {})
    tree = build_tree(X_lat)

    for a, anchor_raw in enumerate(anchor_raw_values):
        anchor_lat = float(rescaler.fwd(float(anchor_raw)))
        kw_base = dict(line_props)
        if colors is not None:
            kw_base["color"] = colors[a]
        for slice_raw in slice_values_raw:
            slice_lat = float(rescaler.fwd(float(slice_raw)))
            delta_lat = np.linspace(0.0, slice_lat, res)
            x2_lat = slice_lat - delta_lat
            plot_x_lat = anchor_lat + delta_lat
            query = np.column_stack([np.full(res, anchor_lat), x2_lat])

            requested = ["mean", "variance"]
            if offset_cutoff is not None:
                requested.append("centroid_offset")
            knn_result = knn_stats(query, Y_lat, tree=tree, stats=requested, **knn_stats_params)
            if offset_cutoff is not None:
                knn_mean, _knn_var, knn_offset = knn_result
                boundary = np.asarray(knn_offset) > offset_cutoff
                y_lat = np.where(boundary, np.nan, np.asarray(knn_mean).reshape(-1))
            else:
                knn_mean, _knn_var = knn_result
                y_lat = np.asarray(knn_mean).reshape(-1)

            ax.plot(plot_x_lat, y_lat, **kw_base)


_HEAD_DEFAULT = {"linestyle": "--", "lw": 0.9}
_TAIL_DEFAULT = {"linestyle": "-.", "lw": 0.9}
_CHORD_DEFAULT = {"linestyle": ":", "lw": 1.1}


def plot_linearity_reference(
    ax,
    X,
    Y,
    slices,
    rescaler=None,
    xlims=(0.0, 1.0),
    colors=None,
    knn_stats_params=None,
    head_frac=0.1,
    tail_frac=0.1,
    show_head=True,
    show_tail=True,
    show_chord=True,
    line_props=None,
    head_props=None,
    tail_props=None,
    chord_props=None,
    res=200,
    n_curve=200,
):
    X = np.asarray(X)
    Y = np.asarray(Y)
    slices = np.asarray(slices)
    knn_stats_params = dict(knn_stats_params or {})
    knn_radius = float(knn_stats_params.get("radius", 0.075))
    knn_stats_params["radius"] = knn_radius

    xmin = float(X[:, 0].min() if xlims[0] is None else xlims[0])
    xmax = float(X[:, 0].max() if xlims[1] is None else xlims[1])
    xquery_min = max(xmin, float(X[:, 0].min()) + knn_radius * 0.5)
    xquery_max = min(xmax, float(X[:, 0].max()) - knn_radius)
    xq = np.linspace(xquery_min, xquery_max, res)

    tree = build_tree(X)
    nslices = slices.shape[0] if slices.ndim else 0
    n_input = X.shape[1]

    base = {"alpha": 0.85, "zorder": 3.0}
    base.update(line_props or {})

    def _plot_raw(x_raw, y_raw, kw):
        if rescaler is None:
            ax.plot(x_raw, y_raw, **kw)
        else:
            ax.plot(rescaler.fwd(x_raw), rescaler.fwd(y_raw), **kw)

    for i in range(nslices):
        query = xq.reshape(-1, 1)
        if n_input > 1:
            query = np.hstack([query, np.tile(slices[i], (query.shape[0], 1))])
        knn_mean = np.asarray(
            knn_stats(query, Y, tree=tree, stats=["mean"], **knn_stats_params)
        ).reshape(-1)
        finite = np.isfinite(knn_mean)
        if finite.sum() < 2:
            continue
        xs, ys = xq[finite], knn_mean[finite]
        lo, hi = float(xs[0]), float(xs[-1])
        span = hi - lo
        if span <= 0:
            continue
        xr = xs if rescaler is None else np.asarray(rescaler.inv(xs))
        yr = ys if rescaler is None else np.asarray(rescaler.inv(ys))
        ok = np.isfinite(xr) & np.isfinite(yr)
        if ok.sum() < 2:
            continue
        xd_lat = np.linspace(lo, hi, n_curve)
        xd_raw = xd_lat if rescaler is None else np.asarray(rescaler.inv(xd_lat))
        kw_color = {"color": colors[i]} if colors is not None else {}
        xr_ok, yr_ok = xr[ok], yr[ok]

        if show_chord and xr_ok[-1] != xr_ok[0]:
            slope = (yr_ok[-1] - yr_ok[0]) / (xr_ok[-1] - xr_ok[0])
            y_chord = yr_ok[0] + slope * (xd_raw - xr_ok[0])
            _plot_raw(xd_raw, y_chord, {**_CHORD_DEFAULT, **base, **kw_color, **(chord_props or {})})
        for show, frac, defaults, props, mask in (
            (show_head, head_frac, _HEAD_DEFAULT, head_props, xs <= lo + (head_frac or 0) * span),
            (show_tail, tail_frac, _TAIL_DEFAULT, tail_props, xs >= hi - (tail_frac or 0) * span),
        ):
            if not (show and frac and frac > 0 and (mask & ok).sum() >= 2):
                continue
            sel = mask & ok
            a, b = np.polyfit(xr[sel], yr[sel], 1)
            _plot_raw(xd_raw, a * xd_raw + b, {**defaults, **base, **kw_color, **(props or {})})
