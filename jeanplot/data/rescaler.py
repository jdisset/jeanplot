from typing import Protocol, runtime_checkable
import numpy as np
from pydantic import BaseModel, ConfigDict


@runtime_checkable
class Rescaler(Protocol):
    """Bijective transform between a raw value space and a display space.

    `fwd(raw) -> display` projects into plot coordinates.
    `inv(display) -> raw` labels ticks with raw values.

    Implementations must accept numpy arrays of any shape and broadcast.
    """

    def fwd(self, x: np.ndarray) -> np.ndarray: ...
    def inv(self, x: np.ndarray) -> np.ndarray: ...


class DataRescaler(BaseModel):
    """Pydantic-backed base for stateful rescalers used in plotting and data IO.

    Subclasses override ``fwd`` / ``inv`` to define the bijection between a raw
    measurement space and a display space (typically ``[0, 1]``). The base
    implementation is the identity transform.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def fwd(self, x):
        return x

    def inv(self, y):
        return y


class IdentityRescaler(DataRescaler):
    def fwd(self, x):
        return np.asarray(x)

    def inv(self, y):
        return np.asarray(y)
