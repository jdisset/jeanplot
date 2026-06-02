import numpy as np

from jeanplot.plots.heatmap import make_xy_grid
from jeanplot.splat import ConditionalSplat, SplatField


BOUNDS = [(0.0, 1.0), (0.0, 1.0)]
RES = 60
RADIUS = 0.15
SIR = 3.0


def _grid_points(field):
    mesh = np.stack(np.meshgrid(*field._axes, indexing="ij"), axis=-1)
    return mesh.reshape(-1, field.ndim)


def _brute_mean(X, Y, q, radius, sigma):
    d = np.linalg.norm(X[None] - q[:, None], axis=-1)
    K = np.exp(-0.5 * (d / sigma) ** 2)
    K[d > radius] = 0.0
    num = K @ Y
    den = K.sum(1)
    out = np.full((len(q), Y.shape[1]), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok, None]
    return out, den


def test_constant_mean_is_exact():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (4000, 2))
    Y = np.full((4000, 1), 2.718)
    f = SplatField.fit(
        X, Y, bounds=BOUNDS, resolution=RES, radius=RADIUS, sigma_in_radius=SIR, stats=["mean"]
    )
    m = f.lattice("mean")[..., 0]
    sup = np.isfinite(m)
    assert sup.sum() > 0.8 * m.size
    assert np.allclose(m[sup], 2.718, atol=1e-9)


def test_variance_of_constant_is_zero():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (4000, 2))
    Y = np.full((4000, 1), -0.5)
    f = SplatField.fit(
        X,
        Y,
        bounds=BOUNDS,
        resolution=RES,
        radius=RADIUS,
        sigma_in_radius=SIR,
        stats=["std", "variance"],
    )
    v = f.lattice("variance")[..., 0]
    sup = np.isfinite(v)
    assert np.all(v[sup] < 1e-9)


def test_mean_matches_brute_force_nw():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (8000, 2))
    Y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1]
    f = SplatField.fit(
        X, Y, bounds=BOUNDS, resolution=RES, radius=RADIUS, sigma_in_radius=SIR, stats=["mean"]
    )
    q = _grid_points(f)
    m_lat = f.lattice("mean")[..., 0]
    bf, den = _brute_mean(X, Y[:, None], q, RADIUS, RADIUS / SIR)
    bf = bf.reshape(RES, RES, 1)[..., 0]  # q is ij order over axes
    den = den.reshape(RES, RES)
    # interior, well-supported cells only (edge truncation + discretisation)
    interior = np.zeros((RES, RES), bool)
    m = int(RES * 0.2)
    interior[m:-m, m:-m] = True
    good = interior & np.isfinite(m_lat) & (den > den.max() * 0.3)
    assert good.sum() > 100
    err = np.abs(m_lat[good] - bf[good])
    assert np.median(err) < 0.02
    assert np.percentile(err, 95) < 0.05


def test_grad_recovers_linear_slope():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (12000, 2))
    a = np.array([1.7, -0.9])
    Y = X @ a + 0.3
    f = SplatField.fit(
        X, Y, bounds=BOUNDS, resolution=RES, radius=RADIUS, sigma_in_radius=SIR, stats=["grad"]
    )
    g = f.lattice("grad")
    m = int(RES * 0.25)
    sub = g[m:-m, m:-m]
    sub = sub[np.isfinite(sub).all(-1)]
    assert sub.shape[0] > 100
    assert np.allclose(np.median(sub, axis=0), a, atol=0.05)


def test_centroid_offset_small_in_interior():
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, (8000, 2))
    Y = np.zeros((8000, 1))
    f = SplatField.fit(
        X,
        Y,
        bounds=BOUNDS,
        resolution=RES,
        radius=RADIUS,
        sigma_in_radius=SIR,
        stats=["centroid_offset"],
    )
    off = f.lattice("centroid_offset")
    m = int(RES * 0.25)
    sub = off[m:-m, m:-m]
    sub = sub[np.isfinite(sub)]
    # uniform cloud, interior: centroid ~ query point
    assert np.median(sub) < 0.02


def test_zslice_band_matches_3d_brute():
    rng = np.random.default_rng(5)
    X = rng.uniform(0, 1, (20000, 3))
    Y = X[:, 0] - X[:, 1] + 0.2 * X[:, 2]
    z0 = np.array([0.5])
    f = SplatField.fit(
        X,
        Y,
        bounds=BOUNDS,
        resolution=RES,
        radius=RADIUS,
        sigma_in_radius=SIR,
        zslice=z0,
        stats=["mean"],
    )
    m_lat = f.lattice("mean")[..., 0]
    q2 = _grid_points(f)
    q3 = np.hstack([q2, np.full((len(q2), 1), 0.5)])
    bf, den = _brute_mean(X, Y[:, None], q3, RADIUS, RADIUS / SIR)
    bf = bf.reshape(RES, RES)
    den = den.reshape(RES, RES)
    mm = int(RES * 0.25)
    interior = np.zeros((RES, RES), bool)
    interior[mm:-mm, mm:-mm] = True
    good = interior & np.isfinite(m_lat) & (den > den.max() * 0.3)
    assert good.sum() > 50
    assert np.median(np.abs(m_lat[good] - bf[good])) < 0.03


def test_flat_xy_matches_make_xy_grid_order():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, (6000, 2))
    Y = (X[:, 0] + 1)[:, None]
    f = SplatField.fit(
        X, Y, bounds=BOUNDS, resolution=RES, radius=RADIUS, sigma_in_radius=SIR, stats=["mean"]
    )
    flat = f.flat_xy("mean").reshape(-1)
    xy = make_xy_grid(0.0, 1.0, 0.0, 1.0, xres=RES, yres=RES)
    # at each make_xy_grid point the flat value must equal the lattice cell
    # indexed [x-bin, y-bin]; sample a few interior points
    lat = f.lattice("mean")[..., 0]
    idx = np.linspace(0, len(xy) - 1, 50).astype(int)
    for p in idx:
        ix = int(round(xy[p, 0] * (RES - 1)))
        iy = int(round(xy[p, 1] * (RES - 1)))
        a, b = flat[p], lat[ix, iy]
        assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b)


def test_conditional_splat_recovers_distribution():
    rng = np.random.default_rng(8)
    cond = rng.uniform(0, 1, (60000, 1))
    val = cond[:, 0] + rng.normal(0, 0.05, 60000)
    cs = ConditionalSplat.fit(
        cond,
        val,
        bounds=[(0.0, 1.0)],
        resolution=60,
        radius=0.08,
        value_range=(-0.2, 1.2),
        n_value_bins=80,
        min_points=10,
    )
    q = np.array([[0.2], [0.5], [0.8]])
    assert np.allclose(cs.mean_at(q), [0.2, 0.5, 0.8], atol=0.02)
    qt = cs.quantiles_at(q, [0.1, 0.5, 0.9])
    assert np.allclose(qt[:, 1], [0.2, 0.5, 0.8], atol=0.03)
    spread = qt[:, 2] - qt[:, 0]  # ~ 2*1.28*sigma, widened by bandwidth
    assert np.all((spread > 0.1) & (spread < 0.22))
    _, pdf = cs.pdf_at(q)
    assert np.allclose(pdf.sum(1), 1.0)
    assert not np.isfinite(cs.mean_at(np.array([[5.0]])))[0]


def test_multi_output_one_pass():
    rng = np.random.default_rng(6)
    X = rng.uniform(0, 1, (8000, 2))
    Y = np.column_stack([np.full(8000, 1.0), 2.0 * X[:, 0]])
    f = SplatField.fit(
        X, Y, bounds=BOUNDS, resolution=RES, radius=RADIUS, sigma_in_radius=SIR, stats=["mean"]
    )
    m = f.lattice("mean")
    assert m.shape == (RES, RES, 2)
    sup = np.isfinite(m[..., 0])
    assert np.allclose(m[..., 0][sup], 1.0, atol=1e-9)
