import warnings

warnings.warn(
    "jeanplot._deprecated contains biocomp-specific code that will be moved to biocomptools. "
    "Do not depend on these modules.",
    DeprecationWarning,
    stacklevel=2,
)

from jeanplot._deprecated.network_utils import TUInfo as TUInfo, Interaction as Interaction  # noqa: E402
from jeanplot._deprecated.network_diagram import (  # noqa: E402
    NetworkDiagram as NetworkDiagram,
    ComputeNode as ComputeNode,
    TranscriptionNode as TranscriptionNode,
    TranslationNode as TranslationNode,
    AggregationNode as AggregationNode,
    ERNNode as ERNNode,
    InvNode as InvNode,
    FluoNode as FluoNode,
    DeadEndNode as DeadEndNode,
    TUNode as TUNode,
)
from jeanplot._deprecated.network_schematic import (  # noqa: E402
    NetworkGeneticSchematic as NetworkGeneticSchematic,
)
