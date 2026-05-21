from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Rescaler(Protocol):
    """Bijective transform between a raw value space and a display space.

    `fwd(raw) -> display` projects into plot coordinates.
    `inv(display) -> raw` labels ticks with raw values.

    Implementations must accept numpy arrays of any shape and broadcast.
    """

    def fwd(self, x: np.ndarray) -> np.ndarray: ...
    def inv(self, x: np.ndarray) -> np.ndarray: ...


class IdentityRescaler:
    def fwd(self, x):
        return np.asarray(x)

    def inv(self, x):
        return np.asarray(x)
