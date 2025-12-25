"""Tests for core data models."""
from jeanplot import Size, Offset, BoxStyle, LayoutConstraints, Transform


class TestSize:
    """Size model."""

    def test_default_zero(self):
        """Default size is zero."""
        s = Size()
        assert s.width == 0 and s.height == 0

    def test_size_values(self):
        """Size stores width/height."""
        s = Size(width=100, height=50)
        assert s.width == 100
        assert s.height == 50

    def test_size_equality(self):
        """Size equality comparison."""
        s1 = Size(100, 50)
        s2 = Size(100, 50)
        assert s1 == s2


class TestOffset:
    """Offset model."""

    def test_default_zero(self):
        """Default offset is zero."""
        o = Offset()
        dx, dy = o.compute(Size(100, 100), Size(200, 200))
        assert dx == 0 and dy == 0

    def test_absolute_offset(self):
        """Absolute offset applied directly."""
        o = Offset(absolute=(10, 20))
        dx, dy = o.compute(Size(100, 100), Size(200, 200))
        assert dx == 10 and dy == 20

    def test_relative_offset(self):
        """Relative offset based on self size."""
        o = Offset(relative=(0.5, 0.25))
        dx, dy = o.compute(Size(100, 80), Size(200, 200))
        assert dx == 50  # 0.5 * 100
        assert dy == 20  # 0.25 * 80

    def test_parent_relative_offset(self):
        """Parent-relative offset based on parent size."""
        o = Offset(parent_relative=(0.1, 0.2))
        dx, dy = o.compute(Size(50, 50), Size(200, 100))
        assert dx == 20  # 0.1 * 200
        assert dy == 20  # 0.2 * 100

    def test_combined_offsets(self):
        """All offset types combine additively."""
        o = Offset(
            absolute=(5, 5),
            relative=(0.1, 0.1),
            parent_relative=(0.05, 0.05),
        )
        dx, dy = o.compute(Size(100, 100), Size(200, 200))
        # 5 + 10 + 10 = 25
        assert dx == 25
        assert dy == 25


class TestBoxStyle:
    """BoxStyle model."""

    def test_default_transparent(self):
        """Default has no background."""
        s = BoxStyle()
        assert s.background_color is None

    def test_padding_tuple(self):
        """Padding stored as 4-tuple."""
        s = BoxStyle(padding=(10, 20, 30, 40))
        assert s.padding == (10, 20, 30, 40)

    def test_content_box(self):
        """Content box subtracts padding."""
        s = BoxStyle(padding=(10, 20, 10, 20))
        w, h = s.content_box(Size(100, 80))
        assert w == 60  # 100 - 20 - 20
        assert h == 60  # 80 - 10 - 10

    def test_border_properties(self):
        """Border properties stored."""
        s = BoxStyle(border_color="#ff0000", border_width=2)
        assert s.border_color == "#ff0000ff"  # normalized with alpha
        assert s.border_width == 2


class TestLayoutConstraints:
    """LayoutConstraints model."""

    def test_default_row(self):
        """Default direction is row."""
        lc = LayoutConstraints()
        assert lc.direction == "row"

    def test_column_direction(self):
        """Column direction stored."""
        lc = LayoutConstraints(direction="column")
        assert lc.direction == "column"

    def test_alignment_options(self):
        """Alignment options stored."""
        lc = LayoutConstraints(align_items="center", justify_content="end")
        assert lc.align_items == "center"
        assert lc.justify_content == "end"


class TestTransform:
    """Transform model."""

    def test_default_identity(self):
        """Default is identity transform."""
        t = Transform()
        assert t.translate == (0, 0)
        assert t.rotate == 0
        assert t.scale == (1, 1)

    def test_translate(self):
        """Translation stored."""
        t = Transform(translate=(10, 20))
        assert t.translate == (10, 20)

    def test_rotate(self):
        """Rotation stored."""
        t = Transform(rotate=45)
        assert t.rotate == 45

    def test_scale_uniform(self):
        """Uniform scale stored."""
        t = Transform(scale=(2, 2))
        assert t.scale == (2, 2)

    def test_to_matrix_identity(self):
        """Identity transform is identity matrix."""
        import numpy as np
        from numpy.testing import assert_allclose
        t = Transform()
        mat = t.to_matrix(Size(100, 100))
        assert_allclose(mat, np.eye(3))

    def test_to_matrix_translate(self):
        """Translation in matrix."""
        import numpy as np
        t = Transform(translate=(10, 20))
        mat = t.to_matrix(Size(100, 100))
        assert mat[0, 2] == 10
        assert mat[1, 2] == 20

    def test_to_matrix_scale(self):
        """Scale in matrix."""
        t = Transform(scale=(2, 3))
        mat = t.to_matrix(Size(100, 100))
        assert mat[0, 0] == 2
        assert mat[1, 1] == 3

    def test_to_matrix_rotate(self):
        """Rotation in matrix."""
        import numpy as np
        from numpy.testing import assert_allclose
        t = Transform(rotate=90)
        mat = t.to_matrix(Size(100, 100))
        # 90 degree rotation swaps axes
        assert_allclose(abs(mat[0, 1]), 1, atol=0.01)
        assert_allclose(abs(mat[1, 0]), 1, atol=0.01)

    def test_to_matrix_skew(self):
        """Skew in matrix."""
        import numpy as np
        t = Transform(skew_x=45)  # 45 degrees
        mat = t.to_matrix(Size(100, 100))
        assert_allclose = __import__('numpy.testing', fromlist=['assert_allclose']).assert_allclose
        assert_allclose(mat[0, 1], 1.0, atol=0.01)  # tan(45) = 1

    def test_repr(self):
        """Transform repr includes non-default values."""
        t = Transform(translate=(10, 20), rotate=45)
        r = repr(t)
        assert "t=" in r
        assert "r=" in r


class TestSizeOperations:
    """Size class methods."""

    def test_union(self):
        """Union takes max of both."""
        s1 = Size(100, 50)
        s2 = Size(50, 100)
        u = s1.union(s2)
        assert u.width == 100
        assert u.height == 100

    def test_min(self):
        """Min takes minimum of both."""
        s1 = Size(100, 50)
        s2 = Size(50, 100)
        m = Size.min(s1, s2)
        assert m.width == 50
        assert m.height == 50

    def test_max(self):
        """Max takes maximum of both."""
        s1 = Size(100, 50)
        s2 = Size(50, 100)
        m = Size.max(s1, s2)
        assert m.width == 100
        assert m.height == 100

    def test_repr(self):
        """Size repr shows values."""
        s = Size(100, 50)
        assert "100" in repr(s)
        assert "50" in repr(s)


class TestOffsetRepr:
    """Offset repr and str."""

    def test_empty_offset_repr(self):
        """Empty offset has minimal repr."""
        o = Offset()
        assert repr(o) == "Offset()"

    def test_absolute_offset_repr(self):
        """Absolute offset in repr."""
        o = Offset(absolute=(10, 20))
        assert "abs=" in repr(o)

    def test_relative_offset_repr(self):
        """Relative offset in repr."""
        o = Offset(relative=(0.5, 0.5))
        assert "rel=" in repr(o)


class TestMarginPadding:
    """MarginPadding properties."""

    def test_margin_properties(self):
        """Individual margin accessors."""
        s = BoxStyle(margin=(10, 20, 30, 40))
        assert s.margin_top == 10
        assert s.margin_right == 20
        assert s.margin_bottom == 30
        assert s.margin_left == 40

    def test_padding_properties(self):
        """Individual padding accessors."""
        s = BoxStyle(padding=(5, 10, 15, 20))
        assert s.padding_top == 5
        assert s.padding_right == 10
        assert s.padding_bottom == 15
        assert s.padding_left == 20


class TestColorNormalization:
    """normalize_color function."""

    def test_none_color(self):
        """None returns None."""
        from jeanplot.core.models import normalize_color
        assert normalize_color(None) is None

    def test_none_string(self):
        """'none' string returns None."""
        from jeanplot.core.models import normalize_color
        assert normalize_color("none") is None
        assert normalize_color("NONE") is None

    def test_hex_normalized(self):
        """Hex color normalized."""
        from jeanplot.core.models import normalize_color
        result = normalize_color("#ff0000")
        assert result.startswith("#ff0000")

    def test_rgb_tuple(self):
        """RGB tuple normalized to hex."""
        from jeanplot.core.models import normalize_color
        result = normalize_color((1.0, 0.0, 0.0))
        assert result.startswith("#ff0000")

    def test_invalid_color(self):
        """Invalid color returns None."""
        from jeanplot.core.models import normalize_color
        result = normalize_color("not_a_color")
        assert result is None


class TestLayoutConstraintsRepr:
    """LayoutConstraints repr."""

    def test_repr(self):
        """Repr shows key values."""
        lc = LayoutConstraints(direction="column", gap=20)
        r = repr(lc)
        assert "column" in r
        assert "20" in r


class TestShadow:
    """Shadow model."""

    def test_default_shadow(self):
        """Default shadow values."""
        from jeanplot.core.models import Shadow
        s = Shadow()
        assert s.blur_radius == 3.0
        assert s.offset_x == 0.0

    def test_box_with_shadow(self):
        """BoxStyle can have shadow."""
        from jeanplot.core.models import Shadow
        shadow = Shadow(offset_x=5, offset_y=5)
        box = BoxStyle(shadow=shadow)
        assert box.shadow.offset_x == 5
