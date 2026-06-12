import tempfile
import matplotlib

matplotlib.use("Agg")

import dracon as dr
import jeanplot
from jeanplot import Figure, PlotPanel, Size, LayoutConstraints
from jeanplot.core.style import jstyle
from jeanplot.core.style_engine import merge_jstyle_rules
from pydantic import PrivateAttr


class Recorder(PlotPanel):
    _draw_calls: int = PrivateAttr(default=0)

    def draw(self, ax):
        self._draw_calls += 1


def test_figure_theme_applied_during_render():
    with tempfile.TemporaryDirectory() as td:
        fig = Figure(
            id="fig",
            output_dir=td,
            output_file="t.png",
            min_dimensions=Size(2.0, 2.0),
            layout=LayoutConstraints(direction="row"),
            dpi=50,
            theme={"[id=tinted]": {"title": "themed"}},
        )
        panel = Recorder(id="tinted", min_dimensions=Size(2.0, 2.0))
        fig.add_child(panel)
        mfig = jeanplot.render(fig)
        assert panel.title == "themed"
        import matplotlib.pyplot as plt

        plt.close(mfig)


def test_merge_overrides_same_selector_field_not_shadowed():
    # A YAML-loaded cascade keeps RAW string selector keys until applied. An override on
    # the SAME selector + SAME field must merge (override wins) rather than land as a second
    # same-specificity rule that the base shadows under fill. Regression: the base tree was
    # not parsed to Locator keys, so a string `"Box"` key never matched `Locator("Box")`.
    base = dr.loads("rules: !cascade:jstyle_fill\n  Box:\n    title: base\n    font_size: 9\n")[
        "rules"
    ]
    merged = merge_jstyle_rules(base, {"Box": {"title": "override"}})
    box_rules = [dict(v) for k, v in merged._rule_tree.items() if "Box" in str(k)]
    assert len(box_rules) == 1  # one Box rule, not a base + a shadowing duplicate
    assert box_rules[0]["title"] == "override"  # override wins
    assert box_rules[0]["font_size"] == 9  # untouched sibling survives the merge


def test_theme_overrides_same_selector_field_wins_at_render():
    # End-to-end of the above: base theme (loaded cascade) sets Recorder.title; theme_overrides
    # sets the SAME field on the SAME type selector. The override must win on the drawn panel.
    with tempfile.TemporaryDirectory() as td:
        base = dr.loads("rules: !cascade:jstyle_fill\n  Recorder:\n    title: base\n")["rules"]
        fig = Figure(
            id="fig",
            output_dir=td,
            output_file="o.png",
            min_dimensions=Size(2.0, 2.0),
            layout=LayoutConstraints(direction="row"),
            dpi=50,
            theme=base,
            theme_overrides={"Recorder": {"title": "override"}},
        )
        panel = Recorder(id="r", min_dimensions=Size(2.0, 2.0))
        fig.add_child(panel)
        mfig = jeanplot.render(fig)
        assert panel.title == "override"
        import matplotlib.pyplot as plt

        plt.close(mfig)


def test_figure_no_theme_leaves_ambient_jstyle():
    with tempfile.TemporaryDirectory() as td:
        jstyle.clear()
        jstyle.update({"[id=x]": {"title": "ambient"}})
        ambient = jstyle._cascade
        fig = Figure(
            id="fig",
            output_dir=td,
            output_file="n.png",
            min_dimensions=Size(2.0, 2.0),
            dpi=50,
        )
        panel = Recorder(id="x", min_dimensions=Size(2.0, 2.0))
        fig.add_child(panel)
        mfig = jeanplot.render(fig)
        assert jstyle._cascade is ambient
        assert panel.title == "ambient"
        import matplotlib.pyplot as plt

        plt.close(mfig)
