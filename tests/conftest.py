"""Shared fixtures for jeanplot tests."""
import pytest
import matplotlib.pyplot as plt

from jeanplot import (
    Container,
    Size,
    BoxStyle,
    LayoutConstraints,
    jstyle,
    MockRenderer,
)
from jeanplot.gene.data import TUData, PartData, CircuitData, InteractionData, SourceData


@pytest.fixture(autouse=True)
def reset_jstyle():
    """Resets jstyle before/after each test."""
    original = jstyle._cascade
    jstyle._cascade = None
    yield
    jstyle._cascade = original


@pytest.fixture(autouse=True)
def cleanup_plt():
    """Close matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def mock_renderer():
    """Renderer without matplotlib dependency."""
    return MockRenderer()


@pytest.fixture
def row_container():
    """Container with row layout and 3 children."""
    parent = Container(
        id="row",
        min_dimensions=Size(200, 100),
        layout=LayoutConstraints(direction="row", gap=10),
        style=BoxStyle(padding=(10, 10, 10, 10)),
    )
    for i in range(3):
        parent.add_child(Container(id=f"c{i}", min_dimensions=Size(50, 50)))
    return parent


@pytest.fixture
def column_container():
    """Container with column layout and 3 children."""
    parent = Container(
        id="col",
        min_dimensions=Size(100, 200),
        layout=LayoutConstraints(direction="column", gap=10),
        style=BoxStyle(padding=(10, 10, 10, 10)),
    )
    for i in range(3):
        parent.add_child(Container(id=f"c{i}", min_dimensions=Size(50, 50)))
    return parent


@pytest.fixture
def simple_tu_data():
    """Basic transcription unit data."""
    return TUData(
        id="test_tu",
        name="TestTU",
        parts=[
            PartData(id="p1", name="pCMV", role="promoter"),
            PartData(id="p2", name="GFP", role="reporter"),
            PartData(id="p3", name="pA", role="terminator"),
        ],
    )


@pytest.fixture
def simple_circuit_data():
    """Basic circuit with two TUs and an interaction."""
    return CircuitData(
        transcription_units=[
            TUData(
                id="tu1",
                name="Reporter",
                parts=[
                    PartData(id="p1", name="pCMV", role="promoter"),
                    PartData(id="p2", name="GFP", role="reporter"),
                    PartData(id="p3", name="pA", role="terminator"),
                ],
            ),
            TUData(
                id="tu2",
                name="Regulator",
                parts=[
                    PartData(id="p4", name="pEF1a", role="promoter"),
                    PartData(id="p5", name="CasE", role="regulator"),
                    PartData(id="p6", name="pA2", role="terminator"),
                ],
            ),
        ],
        sources=[
            SourceData(id="src1", name="Plasmid1", tu_ids=["tu1"]),
            SourceData(id="src2", name="Plasmid2", tu_ids=["tu2"]),
        ],
        interactions=[
            InteractionData(
                id="int1",
                source_tu="tu2",
                source_part="p5",
                target_tu="tu1",
                target_part="p2",
                interaction_type="inhibition",
            ),
        ],
    )
