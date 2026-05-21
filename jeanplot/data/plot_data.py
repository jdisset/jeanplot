from typing import Self, Annotated, Any, TypeVar, TypeAlias
from collections.abc import Callable
import numpy as np
from pydantic import BaseModel, ConfigDict, BeforeValidator
from jeanplot.core.debug import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
NdArray: TypeAlias = np.ndarray


def _asarray(x):
    return np.asarray(x, dtype=np.float32) if x is not None else None


class DataDimensions(BaseModel):
    input: int = 0
    output: int = 0


class PlotData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    xval: Annotated[NdArray | None, BeforeValidator(_asarray)]
    yval: Annotated[NdArray | None, BeforeValidator(_asarray)]

    input_names: list[str] = []
    output_name: str | list[str] = "output"

    column_names: list[str] | None = None

    metadata: dict[str, Any] = {}

    force_single_output: bool = True
    disable_check_shapes: bool = False

    @property
    def column_proteins(self) -> list[str] | None:
        return self.column_names

    @column_proteins.setter
    def column_proteins(self, value: list[str] | None):
        self.column_names = value

    @property
    def x(self) -> NdArray:
        assert self.xval is not None
        self.check_shapes()
        return self.xval

    @property
    def y(self) -> NdArray:
        assert self.yval is not None
        self.check_shapes()
        return self.yval

    @property
    def dimensions(self) -> DataDimensions:
        self.check_shapes()
        if not isinstance(self.input_names, list):
            logger.warning(f"Input names are not a list: {self.input_names}")
            return DataDimensions()
        if len(self.input_names) > 0:
            return DataDimensions(input=len(self.input_names), output=1)
        return DataDimensions(input=0, output=1)

    def check_shapes(self) -> Self:
        if self.disable_check_shapes:
            return self
        assert self.xval is not None
        assert self.yval is not None

        if self.xval.ndim == 1:
            self.xval = self.xval.reshape(-1, 1)
        if self.yval.ndim == 1:
            self.yval = self.yval.reshape(-1, 1)

        if self.xval.shape[0] != self.yval.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples. "
                f"Shapes are {self.xval.shape} and {self.yval.shape}"
            )

        if self.yval.shape[1] > 1:
            assert len(self.output_name) == self.yval.shape[1], (
                f"Output name {self.output_name} does not match "
                f"the number of outputs {self.yval.shape[1]}"
            )
            if self.force_single_output:
                logger.warning(
                    f"Y has {self.yval.shape[1]} outputs, but only 1 output is expected. "
                    f"Folding extra outputs into inputs."
                )
                newxval = np.concatenate([self.xval, self.yval[:, 1:]], axis=1)
                self.xval = newxval
                self.yval = self.yval[:, :1]
                self.input_names.extend(self.output_name[1:])
                self.output_name = self.output_name[0]
        return self

    def __deepcopy__(self, memo):
        return self


class LazyPlotData(PlotData):
    get_xy: Callable[[PlotData], tuple[NdArray, NdArray]]

    xval: Annotated[NdArray | None, BeforeValidator(_asarray)] = None
    yval: Annotated[NdArray | None, BeforeValidator(_asarray)] = None

    @property
    def x(self) -> NdArray:
        self.set_xy()
        assert self.xval is not None
        return _asarray(self.xval)

    @property
    def y(self) -> NdArray:
        self.set_xy()
        assert self.yval is not None
        return _asarray(self.yval)

    def set_xy(self):
        if self.xval is None:
            self.xval, self.yval = self.get_xy(self)
        self.check_shapes()

    @property
    def dimensions(self) -> DataDimensions:
        self.set_xy()
        if not isinstance(self.input_names, list):
            logger.warning(f"Input names are not a list: {self.input_names}")
            return DataDimensions()
        if len(self.input_names) > 0:
            return DataDimensions(input=len(self.input_names), output=1)
        return DataDimensions(input=0, output=1)

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        if self.xval is None:
            return f"LazyPlotData[not loaded, get_xy={self.get_xy}]"
        return f"LazyPlotData[loaded with {len(self.xval)} samples]"

    def __str__(self):
        return self.__repr__()
