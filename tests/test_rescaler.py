import numpy as np
import pytest

from jeanplot import Rescaler, IdentityRescaler


def test_identity_satisfies_protocol():
    r = IdentityRescaler()
    assert isinstance(r, Rescaler)


def test_identity_round_trip():
    r = IdentityRescaler()
    x = np.linspace(0, 1, 5)
    np.testing.assert_array_equal(r.fwd(x), x)
    np.testing.assert_array_equal(r.inv(x), x)


def test_protocol_accepts_duck_typed_object():
    class Logish:
        def fwd(self, x):
            return np.log(x)

        def inv(self, x):
            return np.exp(x)

    assert isinstance(Logish(), Rescaler)


def test_protocol_rejects_missing_methods():
    class Half:
        def fwd(self, x):
            return x

    assert not isinstance(Half(), Rescaler)


def test_biocomp_datarescaler_satisfies_protocol():
    biocomp_datautils = pytest.importorskip("biocomp.datautils")
    assert hasattr(biocomp_datautils.DataRescaler, "fwd")
    assert hasattr(biocomp_datautils.DataRescaler, "inv")
