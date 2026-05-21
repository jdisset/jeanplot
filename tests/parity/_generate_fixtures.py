"""Generate fixture JSON files for parity tests. Run once, results check in.

Usage:
    python -m tests.parity._generate_fixtures

These fixtures are intentionally tiny (60-200 samples) so the rendered
images stay small and parity assertions stay fast.
"""

import json
from pathlib import Path

import numpy as np


FIXTURES = Path(__file__).parent / "fixtures"


def _save(name: str, x, y, input_names, output_name="output", metadata=None):
    payload = {
        "x": np.asarray(x, dtype=float).tolist(),
        "y": np.asarray(y, dtype=float).tolist(),
        "input_names": list(input_names),
        "output_name": output_name,
        "metadata": metadata or {},
    }
    (FIXTURES / f"{name}.json").write_text(json.dumps(payload))


def _make_1d():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(120, 1))
    y = 0.6 * np.sin(2 * np.pi * x[:, 0]) + 0.3 + 0.05 * rng.normal(size=120)
    return x, y.reshape(-1, 1)


def _make_2d():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, size=(200, 2))
    y = (x[:, 0] * x[:, 1]) + 0.05 * rng.normal(size=200)
    return x, y.reshape(-1, 1)


def _make_3d():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, size=(300, 3))
    y = (x[:, 0] * x[:, 1] - x[:, 2]) * 0.5 + 0.5
    y = y + 0.05 * rng.normal(size=300)
    return x, y.reshape(-1, 1)


def _make_mvp_pair():
    rng = np.random.default_rng(4)
    measured = rng.uniform(0, 0.8, size=(250,))
    predicted = measured + 0.05 * rng.normal(size=250)
    return measured, predicted


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)

    x, y = _make_1d()
    _save("1d_smooth", x, y, ["x0"], metadata={"network_name": "fix1d"})

    x, y = _make_2d()
    _save("2d_smooth", x, y, ["x0", "x1"], metadata={"network_name": "fix2d"})

    x, y = _make_3d()
    _save("3d_smooth", x, y, ["x0", "x1", "x2"], metadata={"network_name": "fix3d"})

    m, p = _make_mvp_pair()
    payload = {
        "measured": np.asarray(m, dtype=float).tolist(),
        "predicted": np.asarray(p, dtype=float).tolist(),
        "metadata": {"network_name": "fixmvp"},
    }
    (FIXTURES / "mvp_pair.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
