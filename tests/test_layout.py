from jeanplot.component import Component
import numpy as np
import matplotlib.pyplot as plt
from jeanplot.models import Transform, Size, BoxStyle, LayoutConstraints, Offset
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.container import Container
from jeanplot.text import Text
from jeanplot.component import Overlay, AnchorComponent
from numpy.testing import assert_allclose
from jeanplot.debug import set_debug
from jeanplot.connector import (
    Connection,
    StraightCurve,
    SimpleBezierCurve,
    OrthogonalCurve,
)
from jeanplot.path_utils import find_component_by_path
import pytest


set_debug(False)


def get_world_origin(component: Component):
    return component.get_world_origin()


@pytest.fixture(autouse=True)
def setup_test():
    set_debug(True)
    yield
    set_debug(False)


def test_row_layout():
    """test row layout with different alignment options"""

    PAD = 10
    GAP = 5
    CWIDTH = 400
    CHEIGHT = 200
    container = Container(
        id="row-container",
        min_dimensions=Size(width=CWIDTH, height=CHEIGHT),
        layout=LayoutConstraints(
            direction="row",
            align_items="start",
            justify_content="space-between",
            gap=GAP,
        ),
        style=BoxStyle(
            background_color="#f0f0f0",
            border_color="black",
            border_width=1,
            padding=(PAD, PAD, PAD, PAD),  # t, r, b, l
        ),
    )
    B1W, B1H = 80, 50
    B2W, B2H = 100, 80
    B3W, B3H = 120, 30

    box1 = Container(
        id="box1",
        min_dimensions=Size(width=B1W, height=B1H),
        style=BoxStyle(background_color="red"),
    )
    box2 = Container(
        id="box2",
        min_dimensions=Size(width=B2W, height=B2H),
        style=BoxStyle(background_color="green"),
    )
    box3 = Container(
        id="box3",
        min_dimensions=Size(width=B3W, height=B3H),
        style=BoxStyle(background_color="blue"),
    )

    container.add_children([box1, box2, box3])
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=500, height=300)
    container.measure_and_layout(renderer)
    renderer.render_component(ax, container)

    cont_pos = container.get_world_origin()
    # CORRECTION: Use err_msg parameter
    assert_allclose(cont_pos, (0, 0), err_msg="Container origin")
    assert container._dimensions == Size(width=CWIDTH, height=CHEIGHT), "Container dims"
    assert len(container.children) == 3, "Child count"

    # box1:
    box1_pos = box1.get_world_origin()
    exp_b1_x = PAD  # parent left padding
    exp_b1_y = PAD  # parent top padding
    # CORRECTION: Use err_msg parameter
    assert_allclose(
        box1_pos,
        (exp_b1_x, exp_b1_y),
        err_msg=f"Box1 position exp=({exp_b1_x},{exp_b1_y}) got={box1_pos}",
    )
    assert box1._dimensions == Size(width=B1W, height=B1H), "Box1 dims"

    # box2:
    box2_pos = box2.get_world_origin()
    assert box2._dimensions == Size(width=B2W, height=B2H), "Box2 dims"

    # box3:
    box3_pos = box3.get_world_origin()
    assert box3._dimensions == Size(width=B3W, height=B3H), "Box3 dims"

    # calculate space between for checks
    content_w = CWIDTH - PAD - PAD  # left/right padding
    total_child_w = B1W + B2W + B3W
    total_gap_w = 2 * GAP
    total_required = total_child_w + total_gap_w
    extra_space = content_w - total_required
    spacing = extra_space / 2.0 if extra_space > 0 else 0

    # check box2 position relative to box1 + spacing
    exp_b2_x = exp_b1_x + B1W + GAP + spacing
    exp_b2_y = PAD  # align_items=start
    # CORRECTION: Use err_msg parameter
    assert_allclose(
        box2_pos,
        (exp_b2_x, exp_b2_y),
        err_msg=f"Box2 position exp=({exp_b2_x:.1f},{exp_b2_y:.1f}) got={box2_pos}",
    )

    # check box3 position relative to box2 + spacing
    exp_b3_x = exp_b2_x + B2W + GAP + spacing
    exp_b3_y = PAD  # align_items=start
    # CORRECTION: Use err_msg parameter
    assert_allclose(
        box3_pos,
        (exp_b3_x, exp_b3_y),
        err_msg=f"Box3 position exp=({exp_b3_x:.1f},{exp_b3_y:.1f}) got={box3_pos}",
    )


def test_basic_row_space_between_center():
    """test row layout: space-between justification, center alignment, no child offsets"""

    PAD = 20.0
    GAP = 10.0
    BOXW = 50.0
    PWIDTH = 500.0
    PHEIGHT = 200.0

    parent = Container(
        id="basic-row-container",
        min_dimensions=Size(width=PWIDTH, height=PHEIGHT),
        style=BoxStyle(background_color="#f0f0f0", padding=(PAD, PAD, PAD, PAD)),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",  # vertical center
            justify_content="space-between",  # horizontal distribution
            gap=GAP,
        ),
    )
    box1 = Container(
        id="b1",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="red"),
    )
    box2 = Container(
        id="b2",
        min_dimensions=Size(width=BOXW, height=BOXW * 0.5),
        style=BoxStyle(background_color="green"),
    )  # different height
    box3 = Container(
        id="b3",
        min_dimensions=Size(width=BOXW, height=BOXW * 1.5),
        style=BoxStyle(background_color="blue"),
    )  # different height

    parent.add_children([box1, box2, box3])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    ax = renderer.create_context(width=600, height=300)
    renderer.render_component(ax, parent)

    # calculations
    content_w = PWIDTH - 2 * PAD
    content_h = PHEIGHT - 2 * PAD
    content_x = PAD
    content_y = PAD
    num_layout_children = 3
    total_layout_children_width = num_layout_children * BOXW
    total_gap_width = (num_layout_children - 1) * GAP
    total_required_width = total_layout_children_width + total_gap_width
    extra_space = content_w - total_required_width
    spacing = extra_space / (num_layout_children - 1) if num_layout_children > 1 else 0

    # --- assertions ---
    pos_p = get_world_origin(parent)
    assert_allclose(pos_p, (0, 0), err_msg="Parent origin")
    assert parent._dimensions == Size(width=PWIDTH, height=PHEIGHT), "Parent dims"

    # box1
    pos_b1 = get_world_origin(box1)
    exp_b1_x = content_x
    exp_b1_y = content_y + (content_h - box1._dimensions.height) / 2.0  # center align
    assert_allclose(pos_b1, (exp_b1_x, exp_b1_y), err_msg="Box1 position")
    assert box1._dimensions == Size(width=BOXW, height=BOXW), "Box1 dims"

    # box2
    pos_b2 = get_world_origin(box2)
    # corrected expected x: start after box1 + GAP + spacing
    exp_b2_x = exp_b1_x + BOXW + GAP + spacing
    exp_b2_y = content_y + (content_h - box2._dimensions.height) / 2.0  # center align
    assert_allclose(pos_b2, (exp_b2_x, exp_b2_y), err_msg="Box2 position")
    assert box2._dimensions == Size(width=BOXW, height=BOXW * 0.5), "Box2 dims"

    # box3
    pos_b3 = get_world_origin(box3)
    # corrected expected x: start after box2 + GAP + spacing
    exp_b3_x = exp_b2_x + BOXW + GAP + spacing
    exp_b3_y = content_y + (content_h - box3._dimensions.height) / 2.0  # center align
    assert_allclose(pos_b3, (exp_b3_x, exp_b3_y), err_msg="Box3 position")
    assert box3._dimensions == Size(width=BOXW, height=BOXW * 1.5), "Box3 dims"


def test_layout_children_offsets():
    """test children positioned by layout but with added offsets"""

    PAD = 20.0
    GAP = 10.0
    BOXW = 50.0
    PWIDTH = 600.0  # wider to give space
    PHEIGHT = 200.0

    parent = Container(
        id="offset-layout-container",
        min_dimensions=Size(width=PWIDTH, height=PHEIGHT),
        style=BoxStyle(background_color="#f0f0f0", padding=(PAD, PAD, PAD, PAD)),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="start",  # use start justify for simpler base positions
            gap=GAP,
        ),
    )

    # box1 (reference, no offset)
    box1 = Container(id="b1", min_dimensions=Size(width=BOXW, height=BOXW))

    # box2 (relative offset)
    B2_R_OFF_X, B2_R_OFF_Y = 0.5, 0.25
    box2 = Container(
        id="b2",
        min_dimensions=Size(width=BOXW, height=BOXW),
        offset=Offset(relative=(B2_R_OFF_X, B2_R_OFF_Y)),
        style=BoxStyle(background_color="lightgreen"),
    )

    # box3 (absolute offset)
    B3_ABS_OFF_X, B3_ABS_OFF_Y = -10.0, 25.0
    box3 = Container(
        id="b3",
        min_dimensions=Size(width=BOXW, height=BOXW),
        offset=Offset(absolute=(B3_ABS_OFF_X, B3_ABS_OFF_Y)),
        style=BoxStyle(background_color="lightyellow"),
    )

    # box4 (parent relative offset)
    B4_PR_OFF_X, B4_PR_OFF_Y = 0.01, -0.2
    box4 = Container(
        id="b4",
        min_dimensions=Size(width=BOXW, height=BOXW),
        offset=Offset(parent_relative=(B4_PR_OFF_X, B4_PR_OFF_Y)),
        style=BoxStyle(background_color="lightpink"),
    )

    # box5 (mixed offset)
    B5_R_OFF_X, B5_R_OFF_Y = 0.5, 0.3
    B5_PR_OFF_X, B5_PR_OFF_Y = 0.02, -0.2
    B5_ABS_OFF_X, B5_ABS_OFF_Y = -20.0, 15.0
    box5 = Container(
        id="b5",
        min_dimensions=Size(width=BOXW, height=BOXW),
        offset=Offset(
            relative=(B5_R_OFF_X, B5_R_OFF_Y),
            parent_relative=(B5_PR_OFF_X, B5_PR_OFF_Y),
            absolute=(B5_ABS_OFF_X, B5_ABS_OFF_Y),
        ),
        style=BoxStyle(background_color="lightgrey"),
    )

    parent.add_children([box1, box2, box3, box4, box5])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    ax = renderer.create_context(width=700, height=300)
    renderer.render_component(ax, parent)

    # calculations
    content_h = PHEIGHT - 2 * PAD
    content_x = PAD
    content_y = PAD
    y_align_offset = (content_h - BOXW) / 2.0  # center alignment offset

    # expected base positions (layout only, before user offset applied)
    exp_base_b1_x = content_x
    exp_base_b1_y = content_y + y_align_offset
    exp_base_b2_x = exp_base_b1_x + BOXW + GAP
    exp_base_b2_y = exp_base_b1_y
    exp_base_b3_x = exp_base_b2_x + BOXW + GAP
    exp_base_b3_y = exp_base_b2_y
    exp_base_b4_x = exp_base_b3_x + BOXW + GAP
    exp_base_b4_y = exp_base_b3_y
    exp_base_b5_x = exp_base_b4_x + BOXW + GAP
    exp_base_b5_y = exp_base_b4_y

    # --- assertions ---
    pos_p = get_world_origin(parent)
    assert_allclose(pos_p, (0, 0))

    # box1 (no offset)
    pos_b1 = get_world_origin(box1)
    assert_allclose(pos_b1, (exp_base_b1_x, exp_base_b1_y), err_msg="Box1 base pos")

    # box2 (relative offset)
    pos_b2 = get_world_origin(box2)
    # calculate only the *user-defined* delta
    user_offset_b2 = Offset(relative=(B2_R_OFF_X, B2_R_OFF_Y))

    auto_delta_b2_x, auto_delta_b2_y = user_offset_b2.compute(box2._dimensions, parent._dimensions)

    delta_b2_x = (box2._dimensions.width * B2_R_OFF_X) + (parent._dimensions.width * 0.0) + 0.0
    delta_b2_y = (box2._dimensions.height * B2_R_OFF_Y) + (parent._dimensions.height * 0.0) + 0.0

    assert_allclose(
        (auto_delta_b2_x, auto_delta_b2_y), (delta_b2_x, delta_b2_y), err_msg="Box2 offset calc"
    )

    exp_b2_x = exp_base_b2_x + delta_b2_x
    exp_b2_y = exp_base_b2_y + delta_b2_y
    assert_allclose(pos_b2, (exp_b2_x, exp_b2_y), err_msg="Box2 final pos")

    # box3 (absolute offset)
    pos_b3 = get_world_origin(box3)
    # calculate only the *user-defined* delta
    user_offset_b3 = Offset(absolute=(B3_ABS_OFF_X, B3_ABS_OFF_Y))
    delta_b3_x, delta_b3_y = user_offset_b3.compute(box3._dimensions, parent._dimensions)
    exp_b3_x = exp_base_b3_x + delta_b3_x
    exp_b3_y = exp_base_b3_y + delta_b3_y
    assert_allclose(pos_b3, (exp_b3_x, exp_b3_y), err_msg="Box3 final pos")

    # box4 (parent relative offset)
    pos_b4 = get_world_origin(box4)
    # calculate only the *user-defined* delta
    user_offset_b4 = Offset(parent_relative=(B4_PR_OFF_X, B4_PR_OFF_Y))
    delta_b4_x, delta_b4_y = user_offset_b4.compute(box4._dimensions, parent._dimensions)
    exp_b4_x = exp_base_b4_x + delta_b4_x
    exp_b4_y = exp_base_b4_y + delta_b4_y
    assert_allclose(pos_b4, (exp_b4_x, exp_b4_y), err_msg="Box4 final pos")

    # box5 (mixed offset)
    pos_b5 = get_world_origin(box5)
    # calculate only the *user-defined* delta
    user_offset_b5 = Offset(
        relative=(B5_R_OFF_X, B5_R_OFF_Y),
        parent_relative=(B5_PR_OFF_X, B5_PR_OFF_Y),
        absolute=(B5_ABS_OFF_X, B5_ABS_OFF_Y),
    )
    delta_b5_x, delta_b5_y = user_offset_b5.compute(box5._dimensions, parent._dimensions)
    exp_b5_x = exp_base_b5_x + delta_b5_x
    exp_b5_y = exp_base_b5_y + delta_b5_y
    assert_allclose(pos_b5, (exp_b5_x, exp_b5_y), err_msg="Box5 final pos")


def test_overlay_positioning():
    """test overlay components without and with offsets"""

    PAD = 20.0
    BOXW = 50.0
    PWIDTH = 300.0
    PHEIGHT = 200.0

    parent = Container(
        id="overlay-container",
        min_dimensions=Size(width=PWIDTH, height=PHEIGHT),
        style=BoxStyle(background_color="#f0f0f0", padding=(PAD, PAD, PAD, PAD)),
        layout=LayoutConstraints(direction="row", gap=10),  # layout doesn't affect overlays
    )

    # plain overlay, no offset - change type to Container
    box6 = Container(  # *** CHANGED from Overlay to Container ***
        id="overlay-no-offset",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="lightyellow"),  # Use a visible color
        is_overlay=True,  # Keep marked as overlay
    )

    # overlay with offset
    B7_R_OFF_X, B7_R_OFF_Y = 0.3, 0.4
    B7_PR_OFF_X, B7_PR_OFF_Y = 0.1, -0.1
    B7_ABS_OFF_X, B7_ABS_OFF_Y = 30.0, 20.0
    box7 = Container(
        id="overlay-with-offset",
        min_dimensions=Size(width=BOXW, height=BOXW),
        is_overlay=True,
        style=BoxStyle(background_color="lightblue"),
        offset=Offset(
            relative=(B7_R_OFF_X, B7_R_OFF_Y),
            parent_relative=(B7_PR_OFF_X, B7_PR_OFF_Y),
            absolute=(B7_ABS_OFF_X, B7_ABS_OFF_Y),
        ),
    )

    parent.add_children([box6, box7])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    # Use subplots for easier display control if running interactively
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    ax.set_aspect("equal")

    renderer.render_component(ax, parent, adjust_lims=True)

    # calculations
    # parent origin is (0,0)

    # --- assertions ---
    pos_p = get_world_origin(parent)
    assert_allclose(pos_p, (0, 0))

    # box6 (overlay, no offset) - should be at parent's origin (0,0)
    pos_b6 = get_world_origin(box6)
    exp_b6_x, exp_b6_y = (0, 0)
    assert_allclose(pos_b6, (exp_b6_x, exp_b6_y), err_msg="Box6 overlay pos")
    assert box6._dimensions == Size(width=BOXW, height=BOXW), "Box6 dims"

    # box7 (overlay, with offset)
    pos_b7 = get_world_origin(box7)
    # offset is calculated relative to parent origin (0,0) using parent's overall dimensions
    offset_b7_x, offset_b7_y = box7.offset.compute(box7._dimensions, parent._dimensions)
    # final position = parent origin (0,0) + calculated offset
    exp_b7_x = offset_b7_x
    exp_b7_y = offset_b7_y
    assert_allclose(pos_b7, (exp_b7_x, exp_b7_y), err_msg="Box7 overlay w/ offset pos")
    assert box7._dimensions == Size(width=BOXW, height=BOXW), "Box7 dims"


def test_nested_layout_and_overlay():
    """test internal layout and overlay positioning within a nested container"""

    PAD = 20.0
    GAP = 10.0  # internal gap for box8
    BOXW = 50.0
    PWIDTH = 200.0
    PHEIGHT = 200.0
    IBOXW = BOXW / 3.0
    IBOXH = BOXW / 5.0  # = 10.0

    parent = Container(  # outer container to hold box8
        id="nested-parent",
        min_dimensions=Size(width=PWIDTH, height=PHEIGHT),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        layout=LayoutConstraints(
            align_items="start", justify_content="start"
        ),  # simpler parent layout
    )

    # Inner boxes for box8
    inner_box1 = Container(  # nested overlay inside box8
        id="inner-box1-overlay",
        min_dimensions=Size(width=IBOXW, height=IBOXH),
        style=BoxStyle(background_color="darkblue"),
        is_overlay=True,
        offset=Offset(relative=(0.1, 0.25), parent_relative=(0.05, -0.1), absolute=(-10.0, 5.0)),
    )
    inner_box2 = Container(  # nested layout child inside box8
        id="inner-box2-layout",
        min_dimensions=Size(width=IBOXW, height=IBOXH),
        style=BoxStyle(background_color="darkgreen"),
    )
    inner_box3 = Container(  # nested layout child inside box8
        id="inner-box3-layout",
        min_dimensions=Size(width=IBOXW, height=IBOXH),
        style=BoxStyle(background_color="darkorange"),
    )

    # Box8 itself, with its own offset relative to 'parent'
    box8 = Container(
        id="box8-nested",
        min_dimensions=Size(width=BOXW, height=BOXW),  # 50x50
        style=BoxStyle(background_color="#dddddd", border_width=1),  # no padding for easier calc
        layout=LayoutConstraints(
            direction="column",
            align_items="center",  # horizontal center inner_box2/3
            justify_content="space-between",  # affects inner_box2, inner_box3
            gap=GAP,
        ),
        children=[inner_box1, inner_box2, inner_box3],  # inner_box1 is overlay
        offset=Offset(absolute=(10, 5)),  # simple absolute offset for box8
    )

    parent.add_child(box8)

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    ax = renderer.create_context(width=600, height=300)  # make context large enough
    renderer.render_component(ax, parent)

    # --- Calculations for Box8 internal layout ---
    b8_content_w = box8._dimensions.width  # should be BOXW=50
    b8_content_h = box8._dimensions.height  # should be BOXW=50
    b8_num_layout_children = 2  # inner_box2, inner_box3
    b8_total_children_height = b8_num_layout_children * IBOXH  # 2 * 10 = 20
    b8_total_gap_height = (b8_num_layout_children - 1) * GAP  # 1 * 10 = 10
    b8_total_required_height = b8_total_children_height + b8_total_gap_height  # 20 + 10 = 30
    b8_extra_space = b8_content_h - b8_total_required_height  # 50 - 30 = 20
    # space-between: extra space is the gap between first and last
    b8_spacing = (
        b8_extra_space / (b8_num_layout_children - 1) if b8_num_layout_children > 1 else 0
    )  # 20 / 1 = 20

    # align_items='center' horizontal offset within b8_content_w for IBOXW width items
    b8_x_align_offset = (b8_content_w - IBOXW) / 2.0  # (50 - 16.67) / 2 = 16.67

    # --- Assertions ---
    pos_p = get_world_origin(parent)
    assert_allclose(pos_p, (0, 0))

    # Box8 position (relative to parent)
    pos_b8 = get_world_origin(box8)
    # parent layout is start/start
    exp_b8_x = PAD + box8.offset.absolute[0]  # parent content start + box8 offset
    exp_b8_y = PAD + box8.offset.absolute[1]
    assert_allclose(pos_b8, (exp_b8_x, exp_b8_y), err_msg="Box8 position")
    assert box8._dimensions == Size(width=BOXW, height=BOXW), "Box8 dims"

    # Inner_box1 (overlay inside Box8)
    pos_ib1 = get_world_origin(inner_box1)
    # offset relative to box8 origin (pos_b8) using box8 dimensions (BOXW, BOXW)
    offset_ib1_x, offset_ib1_y = inner_box1.offset.compute(inner_box1._dimensions, box8._dimensions)
    exp_ib1_x = pos_b8[0] + offset_ib1_x
    exp_ib1_y = pos_b8[1] + offset_ib1_y
    assert_allclose(pos_ib1, (exp_ib1_x, exp_ib1_y), err_msg="Inner Box 1 overlay pos")
    assert inner_box1._dimensions == Size(width=IBOXW, height=IBOXH), "Inner Box 1 dims"

    # Inner_box2 (layout child 1 inside Box8)
    pos_ib2 = get_world_origin(inner_box2)
    # local position within box8 (origin at 0,0)
    local_ib2_x = b8_x_align_offset  # centered horizontally
    local_ib2_y = 0  # first item in space-between starts at 0
    # world position = box8 world position + local position
    exp_ib2_x = pos_b8[0] + local_ib2_x
    exp_ib2_y = pos_b8[1] + local_ib2_y
    assert_allclose(pos_ib2, (exp_ib2_x, exp_ib2_y), err_msg="Inner Box 2 layout pos")
    assert inner_box2._dimensions == Size(width=IBOXW, height=IBOXH), "Inner Box 2 dims"

    # Inner_box3 (layout child 2 inside Box8)
    pos_ib3 = get_world_origin(inner_box3)
    # local position within box8 (origin at 0,0)
    local_ib3_x = b8_x_align_offset  # centered horizontally
    # starts after inner_box2 height + the internal GAP + calculated space-between spacing
    # Corrected calculation: include GAP
    local_ib3_y = local_ib2_y + IBOXH + GAP + b8_spacing  # Corrected line
    # world position = box8 world position + local position
    exp_ib3_x = pos_b8[0] + local_ib3_x
    exp_ib3_y = pos_b8[1] + local_ib3_y  # expected: 25 + 40 = 65
    assert_allclose(pos_ib3, (exp_ib3_x, exp_ib3_y), err_msg="Inner Box 3 layout pos")
    assert inner_box3._dimensions == Size(width=IBOXW, height=IBOXH), "Inner Box 3 dims"


def test_complex_layout():
    """test row layout with complex offsets and overlays"""

    PAD = 20.0
    GAP = 10.0
    BOXW = 50.0
    PWIDTH = 500.0
    PHEIGHT = 200.0
    IBOXW = BOXW / 3.0
    IBOXH = BOXW / 5.0

    parent = Container(
        id="offset-examples",
        min_dimensions=Size(width=PWIDTH, height=PHEIGHT),
        style=BoxStyle(
            background_color="#f0f0f0",
            border_color="black",
            border_width=1,
            padding=(PAD, PAD, PAD, PAD),
        ),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="space-between",
            gap=GAP,
        ),
        offset=Offset(),
    )
    box1 = Container(
        id="no-offset",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="lightblue", border_color="blue", border_width=2),
    )

    B2_R_OFF_X = 0.5
    B2_R_OFF_Y = 0.25
    box2 = Container(
        id="relative-offset",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="lightgreen", border_color="green", border_width=2),
        offset=Offset(relative=(B2_R_OFF_X, B2_R_OFF_Y)),
    )

    B3_ABS_OFF_X = -10.0
    B3_ABS_OFF_Y = 25.0
    box3 = Container(
        id="absolute-offset",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="lightyellow", border_color="orange", border_width=2),
        offset=Offset(absolute=(B3_ABS_OFF_X, B3_ABS_OFF_Y)),
    )

    B4_PR_OFF_X = 0.1
    B4_PR_OFF_Y = -0.2
    box4 = Container(
        id="parent-relative-offset",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="lightpink", border_color="red", border_width=2),
        offset=Offset(parent_relative=(B4_PR_OFF_X, B4_PR_OFF_Y)),
    )

    B5_R_OFF_X = 0.2
    B5_R_OFF_Y = 0.3
    B5_PR_OFF_X = 0.05
    B5_PR_OFF_Y = -0.1
    B5_ABS_OFF_X = -20.0
    B5_ABS_OFF_Y = 15.0
    box5 = Container(
        id="mixed-offset",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="lightgrey", border_color="grey", border_width=2),
        offset=Offset(
            relative=(B5_R_OFF_X, B5_R_OFF_Y),
            parent_relative=(B5_PR_OFF_X, B5_PR_OFF_Y),
            absolute=(B5_ABS_OFF_X, B5_ABS_OFF_Y),
        ),
    )

    box6 = Overlay(  # overlay 1
        id="overlay",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="lightpurple", border_color="purple", border_width=2),
    )

    B7_R_OFF_X = 0.3
    B7_R_OFF_Y = 0.4
    B7_PR_OFF_X = 0.1
    B7_PR_OFF_Y = -0.2
    B7_ABS_OFF_X = 130.0
    B7_ABS_OFF_Y = 20.0
    box7 = Container(  # overlay 2 with offset
        id="overlay2",
        min_dimensions=Size(width=BOXW, height=BOXW),
        is_overlay=True,
        style=BoxStyle(background_color="lightgrey", border_color="grey", border_width=2),
        offset=Offset(
            relative=(B7_R_OFF_X, B7_R_OFF_Y),
            parent_relative=(B7_PR_OFF_X, B7_PR_OFF_Y),
            absolute=(B7_ABS_OFF_X, B7_ABS_OFF_Y),
        ),
    )

    IB1_R_OFF_X = 0.1
    IB1_R_OFF_Y = 0.25
    IB1_PR_OFF_X = 0.05
    IB1_PR_OFF_Y = -0.1
    IB1_ABS_OFF_X = -10.0
    IB1_ABS_OFF_Y = 5.0
    inner_box1 = Container(  # nested overlay inside box8
        id="inner-box1",
        min_dimensions=Size(width=IBOXW, height=IBOXH),
        style=BoxStyle(background_color="darkblue", border_color="blue", border_width=1),
        is_overlay=True,
        offset=Offset(
            relative=(IB1_R_OFF_X, IB1_R_OFF_Y),
            parent_relative=(IB1_PR_OFF_X, IB1_PR_OFF_Y),
            absolute=(IB1_ABS_OFF_X, IB1_ABS_OFF_Y),
        ),
    )

    inner_box2 = Container(  # nested layout child inside box8
        id="inner-box2",
        min_dimensions=Size(width=IBOXW, height=IBOXH),
        style=BoxStyle(background_color="darkgreen", border_color="green", border_width=1),
    )

    inner_box3 = Container(  # nested layout child inside box8
        id="inner-box3",
        min_dimensions=Size(width=IBOXW, height=IBOXH),
        style=BoxStyle(background_color="darkorange", border_color="orange", border_width=1),
    )

    B8_R_OFF_X = 0.2
    B8_R_OFF_Y = 0.3
    B8_PR_OFF_X = 0.05
    B8_PR_OFF_Y = -0.05
    B8_ABS_OFF_X = -20.0
    B8_ABS_OFF_Y = 15.0
    box8 = Container(  # container with nested children and its own offset
        id="box8-nested",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="#dddddd", border_color="black", border_width=1),
        layout=LayoutConstraints(
            direction="column",
            align_items="center",
            justify_content="space-between",  # affects inner_box2, inner_box3
            gap=GAP,  # gap between inner_box2 and inner_box3
        ),
        children=[inner_box1, inner_box2, inner_box3],  # inner_box1 is overlay!
        offset=Offset(
            relative=(B8_R_OFF_X, B8_R_OFF_Y),
            parent_relative=(B8_PR_OFF_X, B8_PR_OFF_Y),
            absolute=(B8_ABS_OFF_X, B8_ABS_OFF_Y),
        ),
    )

    parent.children = [box1, box2, box3, box4, box5, box8, box6, box7]  # box8 before overlays

    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=600, height=400)
    parent.measure_and_layout(renderer)
    renderer.render_component(ax, parent)
    # plt.title("Complex Offset Examples") # Keep if showing plot interactively

    # --- Calculations ---
    content_w = PWIDTH - 2 * PAD
    content_h = PHEIGHT - 2 * PAD
    content_x = PAD
    content_y = PAD
    num_layout_children = 6  # box1, box2, box3, box4, box5, box8
    total_layout_children_width = num_layout_children * BOXW
    total_gap_width = (num_layout_children - 1) * GAP
    total_required_width = total_layout_children_width + total_gap_width
    extra_space = content_w - total_required_width
    spacing = extra_space / (num_layout_children - 1) if num_layout_children > 1 else 0
    # align_items='center' vertical offset within content_h for BOXW height items
    y_align_offset = (content_h - BOXW) / 2.0

    # --- Corrected Expected base positions (layout only) ---
    exp_base_b1_x = content_x  # 20
    exp_base_b1_y = content_y + y_align_offset  # 20 + 55 = 75
    exp_base_b2_x = exp_base_b1_x + BOXW + GAP + spacing  # 20 + 50 + 10 + 22 = 102
    exp_base_b2_y = exp_base_b1_y  # 75
    exp_base_b3_x = exp_base_b2_x + BOXW + GAP + spacing  # 102 + 50 + 10 + 22 = 184
    exp_base_b3_y = exp_base_b2_y  # 75
    exp_base_b4_x = exp_base_b3_x + BOXW + GAP + spacing  # 184 + 50 + 10 + 22 = 266
    exp_base_b4_y = exp_base_b3_y  # 75
    exp_base_b5_x = exp_base_b4_x + BOXW + GAP + spacing  # 266 + 50 + 10 + 22 = 348
    exp_base_b5_y = exp_base_b4_y  # 75
    exp_base_b8_x = exp_base_b5_x + BOXW + GAP + spacing  # 348 + 50 + 10 + 22 = 430
    exp_base_b8_y = exp_base_b5_y  # 75
    # --- End Corrected Expected base positions ---

    # --- Parent Assertions ---
    pos_parent = get_world_origin(parent)
    assert_allclose(pos_parent, (0, 0), err_msg="Parent should be at origin")
    assert parent._dimensions == Size(width=PWIDTH, height=PHEIGHT), "Parent dimensions mismatch"

    # --- Box1 Assertions (No Offset) ---
    pos_b1 = get_world_origin(box1)
    assert_allclose(pos_b1, (exp_base_b1_x, exp_base_b1_y), err_msg="Box1 position mismatch")
    assert box1._dimensions == Size(width=BOXW, height=BOXW), "Box1 dimensions mismatch"

    # --- Box2 Assertions (Relative Offset) ---
    pos_b2 = get_world_origin(box2)
    user_offset_b2 = Offset(relative=(B2_R_OFF_X, B2_R_OFF_Y))
    delta_b2_x, delta_b2_y = user_offset_b2.compute(box2._dimensions, parent._dimensions)
    exp_b2_x_final = exp_base_b2_x + delta_b2_x  # Use corrected base
    exp_b2_y_final = exp_base_b2_y + delta_b2_y
    assert_allclose(pos_b2, (exp_b2_x_final, exp_b2_y_final), err_msg="Box2 position mismatch")
    assert box2._dimensions == Size(width=BOXW, height=BOXW), "Box2 dimensions mismatch"

    # --- Box3 Assertions (Absolute Offset) ---
    pos_b3 = get_world_origin(box3)
    user_offset_b3 = Offset(absolute=(B3_ABS_OFF_X, B3_ABS_OFF_Y))
    delta_b3_x, delta_b3_y = user_offset_b3.compute(box3._dimensions, parent._dimensions)
    exp_b3_x_final = exp_base_b3_x + delta_b3_x  # Use corrected base
    exp_b3_y_final = exp_base_b3_y + delta_b3_y
    assert_allclose(pos_b3, (exp_b3_x_final, exp_b3_y_final), err_msg="Box3 position mismatch")
    assert box3._dimensions == Size(width=BOXW, height=BOXW), "Box3 dimensions mismatch"

    # --- Box4 Assertions (Parent Relative Offset) ---
    pos_b4 = get_world_origin(box4)
    user_offset_b4 = Offset(parent_relative=(B4_PR_OFF_X, B4_PR_OFF_Y))
    delta_b4_x, delta_b4_y = user_offset_b4.compute(box4._dimensions, parent._dimensions)
    exp_b4_x_final = exp_base_b4_x + delta_b4_x  # Use corrected base
    exp_b4_y_final = exp_base_b4_y + delta_b4_y
    assert_allclose(pos_b4, (exp_b4_x_final, exp_b4_y_final), err_msg="Box4 position mismatch")
    assert box4._dimensions == Size(width=BOXW, height=BOXW), "Box4 dimensions mismatch"

    # --- Box5 Assertions (Mixed Offset) ---
    pos_b5 = get_world_origin(box5)
    user_offset_b5 = Offset(
        relative=(B5_R_OFF_X, B5_R_OFF_Y),
        parent_relative=(B5_PR_OFF_X, B5_PR_OFF_Y),
        absolute=(B5_ABS_OFF_X, B5_ABS_OFF_Y),
    )
    delta_b5_x, delta_b5_y = user_offset_b5.compute(box5._dimensions, parent._dimensions)
    exp_b5_x_final = exp_base_b5_x + delta_b5_x  # Use corrected base
    exp_b5_y_final = exp_base_b5_y + delta_b5_y
    assert_allclose(pos_b5, (exp_b5_x_final, exp_b5_y_final), err_msg="Box5 position mismatch")
    assert box5._dimensions == Size(width=BOXW, height=BOXW), "Box5 dimensions mismatch"

    # --- Box8 Assertions (Container with Offset) ---
    pos_b8 = get_world_origin(box8)
    user_offset_b8 = Offset(
        relative=(B8_R_OFF_X, B8_R_OFF_Y),
        parent_relative=(B8_PR_OFF_X, B8_PR_OFF_Y),
        absolute=(B8_ABS_OFF_X, B8_ABS_OFF_Y),
    )
    delta_b8_x, delta_b8_y = user_offset_b8.compute(box8._dimensions, parent._dimensions)
    exp_b8_x_final = exp_base_b8_x + delta_b8_x  # Use corrected base
    exp_b8_y_final = exp_base_b8_y + delta_b8_y
    assert_allclose(pos_b8, (exp_b8_x_final, exp_b8_y_final), err_msg="Box8 position mismatch")
    assert box8._dimensions == Size(width=BOXW, height=BOXW), "Box8 dimensions mismatch"

    # --- Box6 Assertions (Overlay, No Offset) ---
    pos_b6 = get_world_origin(box6)
    # overlays without offset position themselves at the parent's origin (0,0)
    exp_b6_x, exp_b6_y = (0, 0)
    assert_allclose(pos_b6, (exp_b6_x, exp_b6_y), err_msg="Box6 (Overlay) position mismatch")
    assert box6._dimensions == Size(width=BOXW, height=BOXW), "Box6 dimensions mismatch"

    # --- Box7 Assertions (Overlay, Mixed Offset) ---
    pos_b7 = get_world_origin(box7)
    # overlay offset is calculated relative to parent origin (0,0)
    offset_b7_x, offset_b7_y = box7.offset.compute(box7._dimensions, parent._dimensions)
    exp_b7_x = 0 + offset_b7_x  # relative to parent origin
    exp_b7_y = 0 + offset_b7_y
    assert_allclose(
        pos_b7, (exp_b7_x, exp_b7_y), err_msg="Box7 (Overlay w/ Offset) position mismatch"
    )
    assert box7._dimensions == Size(width=BOXW, height=BOXW), "Box7 dimensions mismatch"

    # --- Box8 Internal Layout Calculations ---
    b8_content_w = box8._dimensions.width
    b8_content_h = box8._dimensions.height
    b8_num_layout_children = 2
    b8_total_children_height = b8_num_layout_children * IBOXH
    b8_total_gap_height = (b8_num_layout_children - 1) * GAP
    b8_total_required_height = b8_total_children_height + b8_total_gap_height
    b8_extra_space = b8_content_h - b8_total_required_height
    b8_spacing = b8_extra_space / (b8_num_layout_children - 1) if b8_num_layout_children > 1 else 0
    b8_x_align_offset = (b8_content_w - IBOXW) / 2.0

    # --- Inner_box1 Assertions (Overlay inside Box8) ---
    pos_ib1 = get_world_origin(inner_box1)
    # offset relative to box8 origin (pos_b8) using box8 dimensions (BOXW, BOXW)
    offset_ib1_x, offset_ib1_y = inner_box1.offset.compute(inner_box1._dimensions, box8._dimensions)
    # world position = box8 world position + overlay offset
    exp_ib1_x = pos_b8[0] + offset_ib1_x  # Use calculated final pos_b8
    exp_ib1_y = pos_b8[1] + offset_ib1_y
    assert_allclose(
        pos_ib1, (exp_ib1_x, exp_ib1_y), err_msg="Inner Box 1 (Overlay) position mismatch"
    )
    assert inner_box1._dimensions == Size(
        width=IBOXW, height=IBOXH
    ), "Inner Box 1 dimensions mismatch"

    # --- Inner_box2 Assertions (Layout Child inside Box8) ---
    pos_ib2 = get_world_origin(inner_box2)
    # local position within box8
    local_ib2_x = b8_x_align_offset
    local_ib2_y = 0  # first item in space-between
    # world position = box8 world position + local position
    exp_ib2_x = pos_b8[0] + local_ib2_x  # Use calculated final pos_b8
    exp_ib2_y = pos_b8[1] + local_ib2_y
    assert_allclose(pos_ib2, (exp_ib2_x, exp_ib2_y), err_msg="Inner Box 2 position mismatch")
    assert inner_box2._dimensions == Size(
        width=IBOXW, height=IBOXH
    ), "Inner Box 2 dimensions mismatch"

    # --- Inner_box3 Assertions (Layout Child inside Box8) ---
    pos_ib3 = get_world_origin(inner_box3)
    # local position within box8
    local_ib3_x = b8_x_align_offset
    local_ib3_y = local_ib2_y + IBOXH + GAP + b8_spacing  # below ib2 + gap + calculated space
    # world position = box8 world position + local position
    exp_ib3_x = pos_b8[0] + local_ib3_x  # Use calculated final pos_b8
    exp_ib3_y = pos_b8[1] + local_ib3_y
    assert_allclose(pos_ib3, (exp_ib3_x, exp_ib3_y), err_msg="Inner Box 3 position mismatch")
    assert inner_box3._dimensions == Size(
        width=IBOXW, height=IBOXH
    ), "Inner Box 3 dimensions mismatch"


def test_simple_attachment_no_offset():
    """
    Test attaching one component to another without offset.
    Expected Visual: A small green square with its top-left corner aligned
                     with the top-left corner of a larger red-border square.
    """

    PAD = 10.0
    ROOT_SIZE = 200.0
    TARGET_SIZE = 100.0
    ATTACHED_SIZE = 20.0

    target = Container(
        id="target",
        min_dimensions=Size(width=TARGET_SIZE, height=TARGET_SIZE),
        style=BoxStyle(border_color="red", border_width=2),  # No background
        offset=Offset(absolute=(50, 50)),  # Position target within root
    )
    # --- FIX: Change type from Component to Container ---
    attached = Container(  # Use Container so background is rendered
        id="attached",
        min_dimensions=Size(width=ATTACHED_SIZE, height=ATTACHED_SIZE),
        style=BoxStyle(background_color="lightgreen"),  # Container renders its style
        attached_to=target,
        # No attachment_offset means default (0,0) relative to target origin
    )
    # -----------------------------------------------------

    root = Container(
        id="attach-test-1",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        children=[target, attached],
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    # Render for visual check if needed
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, root)

    # --- Assertions ---
    pos_target = get_world_origin(target)
    exp_target_x = PAD + 50
    exp_target_y = PAD + 50
    assert_allclose(pos_target, (exp_target_x, exp_target_y), err_msg="Target position")

    pos_attached = get_world_origin(attached)
    # Default attachment offset (0,0) means attached origin aligns with target origin
    exp_attached_x = exp_target_x
    exp_attached_y = exp_target_y
    assert_allclose(
        pos_attached, (exp_attached_x, exp_attached_y), err_msg="Attached position (no offset)"
    )

    # Verify _resolved_attach_target
    assert attached._resolved_attach_target is target, "Attachment target resolution failed"


def test_attachment_with_offset():
    """
    Test attaching one component relative to another using attachment_offset.
    Expected Visual: A small green square positioned relative to the top-right corner of a larger red-border square.
    """
    PAD = 10.0
    ROOT_SIZE = 200.0
    TARGET_SIZE = 100.0
    ATTACHED_SIZE = 20.0

    target = Container(
        id="target",
        min_dimensions=Size(width=TARGET_SIZE, height=TARGET_SIZE),
        style=BoxStyle(border_color="red", border_width=2),  # No background
        offset=Offset(absolute=(50, 50)),  # Position target within root
    )
    attach_offset = Offset(parent_relative=(1.0, 1.0), absolute=(5, -10), relative=(-0.5, -0.5))
    # --- FIX: Change type from Component to Container ---
    attached = Container(  # Use Container so background is rendered
        id="attached",
        min_dimensions=Size(width=ATTACHED_SIZE, height=ATTACHED_SIZE),
        style=BoxStyle(background_color="lightgreen"),
        attached_to=target,
        attachment_offset=attach_offset,
    )
    # -----------------------------------------------------

    root = Container(
        id="attach-test-2",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        children=[target, attached],
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, root)

    # --- Assertions ---
    pos_target = get_world_origin(target)
    exp_target_x = PAD + 50
    exp_target_y = PAD + 50
    assert_allclose(pos_target, (exp_target_x, exp_target_y), err_msg="Target position")

    pos_attached = get_world_origin(attached)
    exp_offset_x = 95.0
    exp_offset_y = 80.0
    exp_attached_x = exp_target_x + exp_offset_x
    exp_attached_y = exp_target_y + exp_offset_y
    assert_allclose(
        pos_attached, (exp_attached_x, exp_attached_y), err_msg="Attached position (with offset)"
    )
    assert attached._resolved_attach_target is target, "Attachment target resolution failed"


def test_attachment_string_path_nested():
    """
    Test attaching a component to a nested child using a string path reference.
    Expected Visual: A small green square attached to the center of a blue square,
                     where the blue square is a child of a red square.
    """
    PAD = 10.0
    ROOT_SIZE = 300.0
    PARENT_SIZE = 150.0
    CHILD_SIZE = 50.0
    ATTACHED_SIZE = 20.0

    child = Container(
        id="child-target",
        min_dimensions=Size(width=CHILD_SIZE, height=CHILD_SIZE),
        style=BoxStyle(background_color="blue"),
        offset=Offset(absolute=(25, 25)),
    )
    parent = Container(
        id="parent",
        min_dimensions=Size(width=PARENT_SIZE, height=PARENT_SIZE),
        style=BoxStyle(background_color="red"),
        offset=Offset(absolute=(50, 50)),
        children=[child],
    )

    # --- FIX: Change type from Component to Container ---
    attached = Container(  # Use Container so background is rendered
        id="attached",
        min_dimensions=Size(width=ATTACHED_SIZE, height=ATTACHED_SIZE),
        style=BoxStyle(background_color="lightgreen"),
        attached_to="parent/child-target",  # Corrected path
        attachment_offset=Offset(
            relative=(0.5, 0.5), parent_relative=(0.5, 0.5)
        ),  # Center on target center
    )
    # -----------------------------------------------------

    root = Container(
        id="attach-test-3",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        children=[parent, attached],
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, root)

    # --- Assertions ---
    resolved_child = find_component_by_path(root, "parent/child-target")
    assert resolved_child is child, "Child target not found by path"

    pos_parent = get_world_origin(parent)
    exp_parent_x = PAD + 50
    exp_parent_y = PAD + 50
    assert_allclose(pos_parent, (exp_parent_x, exp_parent_y), err_msg="Parent position")

    pos_child = get_world_origin(child)
    exp_child_x = exp_parent_x + 25
    exp_child_y = exp_parent_y + 25
    assert_allclose(pos_child, (exp_child_x, exp_child_y), err_msg="Child position")

    pos_attached = get_world_origin(attached)
    offset_x, offset_y = attached.attachment_offset.compute(attached._dimensions, child._dimensions)
    exp_offset_x = 0.5 * ATTACHED_SIZE + 0.5 * CHILD_SIZE  # = 10 + 25 = 35
    exp_offset_y = 0.5 * ATTACHED_SIZE + 0.5 * CHILD_SIZE  # = 10 + 25 = 35
    assert_allclose(
        (offset_x, offset_y), (exp_offset_x, exp_offset_y), err_msg="Attachment offset calculation"
    )

    exp_attached_x = exp_child_x + offset_x
    exp_attached_y = exp_child_y + offset_y
    assert_allclose(
        pos_attached, (exp_attached_x, exp_attached_y), err_msg="Attached position (string path)"
    )

    assert attached._resolved_attach_target is child, "Attachment path resolution failed"


def test_simple_connection_centers():
    """
    Test a basic straight connection between the centers of two components.
    """
    PAD = 10.0
    ROOT_SIZE = 300.0
    BOX_SIZE = 50.0
    GAP = 30.0

    box_a = Container(
        id="box_a",
        min_dimensions=Size(width=BOX_SIZE, height=BOX_SIZE),
        style=BoxStyle(background_color="red"),
    )
    box_b = Container(
        id="box_b",
        min_dimensions=Size(width=BOX_SIZE, height=BOX_SIZE),
        style=BoxStyle(background_color="blue"),
    )

    conn = Connection(
        id="conn_ab",
        start_component=box_a,
        end_component=box_b,
        curve_type=StraightCurve(),
        color="black",
        line_width=2,
        # Default start/end offsets connect centers
    )

    root = Container(
        id="conn-test-1",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        layout=LayoutConstraints(
            direction="row", align_items="center", justify_content="start", gap=GAP
        ),
        children=[box_a, box_b, conn],
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)  # Keep commented unless debugging
    renderer.render_component(ax, root)  # Keep commented unless debugging

    # --- Assertions ---
    pos_a = box_a.get_world_origin()
    exp_a_x = PAD
    exp_a_y = PAD + (ROOT_SIZE - 2 * PAD - BOX_SIZE) / 2
    assert_allclose(pos_a, (exp_a_x, exp_a_y), err_msg="Box A position")

    pos_b = box_b.get_world_origin()
    exp_b_x = exp_a_x + BOX_SIZE + GAP
    exp_b_y = exp_a_y
    assert_allclose(pos_b, (exp_b_x, exp_b_y), err_msg="Box B position")

    # --- CORRECTION: Call internal method to get calculated points ---
    calculated_points = conn._get_world_connection_points()
    assert calculated_points is not None, "Failed to calculate connection points"
    conn_world_start, conn_world_end = calculated_points
    # --- END CORRECTION ---

    # Expected start = center of box_a
    exp_start_x = exp_a_x + BOX_SIZE / 2
    exp_start_y = exp_a_y + BOX_SIZE / 2
    assert_allclose(conn_world_start, (exp_start_x, exp_start_y), err_msg="Connection start point")

    # Expected end = center of box_b
    exp_end_x = exp_b_x + BOX_SIZE / 2
    exp_end_y = exp_b_y + BOX_SIZE / 2
    assert_allclose(conn_world_end, (exp_end_x, exp_end_y), err_msg="Connection end point")


def test_connection_with_anchors():
    """
    Test connecting specific anchor points on two components.
    Expected Visual: Two squares (red, blue). A black orthogonal line starts
                     from the top-center anchor of the red square (going up initially),
                     turns, and goes to the bottom-center anchor of the blue square.
    """

    PAD = 10.0
    ROOT_SIZE = 300.0
    BOX_SIZE = 50.0
    GAP = 30.0

    # Define anchors using parent_relative offsets
    anchor_a_top = AnchorComponent(
        id="anchor_a",  # give anchors ids for clarity
        offset=Offset(parent_relative=(0.5, 1.0)),  # top-center
        direction=(0, -1),  # point down
        min_segment=25,
    )
    anchor_b_bottom = AnchorComponent(
        id="anchor_b",
        offset=Offset(parent_relative=(0.5, 0.0)),  # bottom-center
        # no direction specified for anchor_b, should resolve automatically or use curve default
    )

    box_a = Container(
        id="box_a",  # give boxes ids
        min_dimensions=Size(width=BOX_SIZE, height=BOX_SIZE),
        style=BoxStyle(background_color="red"),
        anchor_points=[anchor_a_top],
    )
    box_b = Container(
        id="box_b",
        min_dimensions=Size(width=BOX_SIZE, height=BOX_SIZE),
        style=BoxStyle(background_color="blue"),
        anchor_points=[anchor_b_bottom],
    )

    # --- Explicitly connect the ANCHORS using their IDs ---
    # This ensures we test the string resolution and anchor logic specifically
    conn = Connection(
        id="conn_anchors",
        start_component="box_a/anchor_a",
        end_component="box_b/anchor_b",
        curve_type=OrthogonalCurve(),  # Default is 'auto' directions
        color="black",
        line_width=2,
        auto_route=False,  # disable auto-routing to force connection to specified anchors
        start_offset=Offset(),  # offset from anchor itself (usually 0)
        end_offset=Offset(),  # offset from anchor itself (usually 0)
    )

    root = Container(
        id="conn-test-2",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        layout=LayoutConstraints(
            direction="row", align_items="center", justify_content="start", gap=GAP
        ),
        children=[box_a, box_b, conn],  # Add connection last
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)  # This resolves paths and calculates layout/connections
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)  # Lower dpi for faster testing runs
    renderer.render_component(ax, root)

    # --- Assertions ---
    # Check component positions (as before)
    pos_a = get_world_origin(box_a)
    exp_a_x = PAD
    exp_a_y = PAD + (ROOT_SIZE - 2 * PAD - BOX_SIZE) / 2  # 10 + 115 = 125
    assert_allclose(pos_a, (exp_a_x, exp_a_y), err_msg="Box A position")

    pos_b = get_world_origin(box_b)
    exp_b_x = exp_a_x + BOX_SIZE + GAP  # 10 + 50 + 30 = 90
    exp_b_y = exp_a_y  # 125
    assert_allclose(pos_b, (exp_b_x, exp_b_y), err_msg="Box B position")

    # Check anchor positions (as before)
    pos_anchor_a = get_world_origin(anchor_a_top)
    anchor_a_local_offset = anchor_a_top.offset.compute(Size(), box_a._dimensions)  # (25, 50)
    exp_anchor_a_x = exp_a_x + anchor_a_local_offset[0]  # 10 + 25 = 35
    exp_anchor_a_y = exp_a_y + anchor_a_local_offset[1]  # 125 + 50 = 175
    assert_allclose(pos_anchor_a, (exp_anchor_a_x, exp_anchor_a_y), err_msg="Anchor A position")

    pos_anchor_b = get_world_origin(anchor_b_bottom)
    anchor_b_local_offset = anchor_b_bottom.offset.compute(Size(), box_b._dimensions)  # (25, 0)
    exp_anchor_b_x = exp_b_x + anchor_b_local_offset[0]  # 90 + 25 = 115
    exp_anchor_b_y = exp_b_y + anchor_b_local_offset[1]  # 125 + 0 = 125
    assert_allclose(pos_anchor_b, (exp_anchor_b_x, exp_anchor_b_y), err_msg="Anchor B position")

    calculated_points = conn._get_world_connection_points()
    assert calculated_points is not None, "Failed to calculate connection points"
    conn_world_start, conn_world_end = calculated_points

    # Check connection points (as before)
    assert_allclose(
        conn_world_start,
        (exp_anchor_a_x, exp_anchor_a_y),
        err_msg="Connection start point (anchor)",
    )
    assert_allclose(
        conn_world_end, (exp_anchor_b_x, exp_anchor_b_y), err_msg="Connection end point (anchor)"
    )

    # --- New Assertions: Check Resolved Curve Parameters ---
    active_curve = conn._get_active_curve()
    assert isinstance(active_curve, OrthogonalCurve), "Curve type should be OrthogonalCurve"

    # Check start direction/length based on anchor_a_top
    assert (
        active_curve.start_direction == "down"
    ), f"Expected start_direction 'up' (from anchor (0,-1)), got '{active_curve.start_direction}'"
    assert_allclose(
        active_curve.start_length, 25.0, err_msg="Start length mismatch (from anchor min_segment)"
    )

    # Check end direction/length - anchor_b had no direction, so curve should use 'auto' or default resolve
    # The actual *resolved* direction might be 'down' based on points, but the parameter set by
    # _update_curve_params_from_target should remain 'auto' because the anchor didn't specify one.
    # We check that it wasn't incorrectly overridden.
    # Note: OrthogonalCurve._resolve_direction handles 'auto' internally during get_path/get_directions
    assert (
        active_curve.end_direction == "auto"
    ), f"Expected end_direction 'auto' (anchor had no direction), got '{active_curve.end_direction}'"
    # Default length should be used if anchor doesn't specify min_segment
    assert_allclose(active_curve.end_length, 5.0, err_msg="End length mismatch (default expected)")


def test_overlay_container_internal_layout():
    """
    tests that an overlay container correctly applies its internal layout
    properties (e.g., align_items, justify_content) to its own children.
    this would fail before the fix in container.apply_layout.

    expected visual: a yellow overlay box positioned absolutely. inside it,
                     a small cyan text box should be perfectly centered.
    """

    ROOT_SIZE = 200.0
    OVERLAY_SIZE = Size(width=100, height=80)
    OVERLAY_OFFSET = Offset(absolute=(50, 60))  # absolute offset for predictability
    TEXT_CONTENT = "Centered"

    # create the child text component first to measure it
    inner_text = Text(
        id="inner_text",
        text=TEXT_CONTENT,
        style=BoxStyle(background_color="cyan", border_width=0.5, border_color="blue"),
        # these internal alignments are less critical now, but good practice
        align="center",
        vertical_align="middle",
    )

    # create the overlay container that holds the text
    overlay_child = Container(
        id="overlay_container",
        min_dimensions=OVERLAY_SIZE,
        max_dimensions=OVERLAY_SIZE,  # fix size for predictability
        is_overlay=True,
        offset=OVERLAY_OFFSET,
        # key layout properties for centering the inner_text
        layout=LayoutConstraints(align_items="center", justify_content="center"),
        style=BoxStyle(background_color="lightyellow", border_width=1, border_color="orange"),
        children=[inner_text],
    )

    # create the root container
    root = Container(
        id="overlay-layout-test-root",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        children=[overlay_child],  # only contains the overlay container
    )

    # --- execution ---
    renderer = MatplotlibRenderer()

    # measure text standalone first to get its natural size for calculations
    text_natural_size = renderer.measure_text(inner_text)
    assert text_natural_size.width > 0 and text_natural_size.height > 0, "Text measurement failed"
    print(f"Measured text size: {text_natural_size}")

    # measure and layout the entire structure
    root.measure_and_layout(renderer)

    # (optional) render for visual verification during development
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    ax.set_aspect("equal")
    renderer.render_component(ax, root, adjust_lims=True)
    plt.title("Overlay Internal Layout Test")
    # plt.show() # uncomment to show plot

    # --- calculations ---
    # overlay's world origin = root origin (0,0) + overlay offset
    exp_overlay_origin_x = OVERLAY_OFFSET.absolute[0]  # 50
    exp_overlay_origin_y = OVERLAY_OFFSET.absolute[1]  # 60

    # expected text origin *within* the overlay container, based on centering
    exp_text_local_x = (OVERLAY_SIZE.width - text_natural_size.width) / 2.0
    exp_text_local_y = (OVERLAY_SIZE.height - text_natural_size.height) / 2.0

    # expected text origin in world coordinates
    exp_text_world_x = exp_overlay_origin_x + exp_text_local_x
    exp_text_world_y = exp_overlay_origin_y + exp_text_local_y

    # --- assertions ---
    # verify overlay container position
    overlay_actual_origin = get_world_origin(overlay_child)
    assert_allclose(
        overlay_actual_origin,
        (exp_overlay_origin_x, exp_overlay_origin_y),
        err_msg="Overlay container position is incorrect",
        atol=1e-6,  # Use tolerance for float comparisons
    )
    assert overlay_child._dimensions == OVERLAY_SIZE, "Overlay container dimensions mismatch"

    # verify inner text position (relative to world)
    inner_actual_origin = get_world_origin(inner_text)
    assert_allclose(
        inner_actual_origin,
        (exp_text_world_x, exp_text_world_y),
        err_msg="Inner text is not centered within the overlay container",
        atol=1e-6,  # Use tolerance for float comparisons
    )
    # verify text component retains its natural measured size
    assert inner_text._dimensions.width == pytest.approx(text_natural_size.width)
    assert inner_text._dimensions.height == pytest.approx(text_natural_size.height)

    assert hasattr(
        inner_text,
        "_layout_origin_in_parent",
    ), "Inner text should have layout origin stored"
    assert_allclose(
        inner_text._layout_origin_in_parent,
        (exp_text_local_x, exp_text_local_y),
        err_msg="Inner text stored layout origin is incorrect",
        atol=1e-6,
    )


def test_row_layout_heterogeneous_padding_margin_start():
    """test row layout, start justification, start alignment with non-uniform padding and margin"""
    PAD = (10, 5, 20, 15)  # t, r, b, l
    MAR = [(5, 2, 10, 3), (8, 4, 6, 1), (12, 7, 9, 5)]  # t, r, b, l for each child
    GAP = 5.0
    BOXW = 40.0
    BOXH = 30.0
    PWIDTH = 300.0
    PHEIGHT = 150.0

    parent = Container(
        id="hetero-pad-mar-row-start",
        min_dimensions=Size(width=PWIDTH, height=PHEIGHT),
        style=BoxStyle(padding=PAD),
        layout=LayoutConstraints(
            direction="row",
            align_items="start",  # align top edges (considering top margins)
            justify_content="start",  # place children from left (considering left margins)
            gap=GAP,
        ),
    )
    box1 = Container(
        id="b1",
        min_dimensions=Size(width=BOXW, height=BOXH),
        style=BoxStyle(background_color="red", margin=MAR[0]),
    )
    box2 = Container(
        id="b2",
        min_dimensions=Size(width=BOXW, height=BOXH + 10),  # taller
        style=BoxStyle(background_color="green", margin=MAR[1]),
    )
    box3 = Container(
        id="b3",
        min_dimensions=Size(width=BOXW, height=BOXH - 10),  # shorter
        style=BoxStyle(background_color="blue", margin=MAR[2]),
    )

    parent.add_children([box1, box2, box3])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    ax = renderer.create_context(width=400, height=200)
    renderer.render_component(ax, parent)

    # calculations
    content_x = PAD[3]  # parent left padding
    content_y = PAD[0]  # parent top padding

    # box1 pos
    exp_b1_x = content_x + MAR[0][3]  # content start + own left margin
    exp_b1_y = content_y + MAR[0][0]  # content start + own top margin
    assert_allclose(get_world_origin(box1), (exp_b1_x, exp_b1_y), err_msg="Box1 pos")

    # box2 pos
    exp_b2_x = (
        exp_b1_x + BOXW + MAR[0][1] + GAP + MAR[1][3]
    )  # prev X + prev W + prev R mar + gap + own L mar
    exp_b2_y = content_y + MAR[1][0]  # content start + own top margin (align_items='start')
    assert_allclose(get_world_origin(box2), (exp_b2_x, exp_b2_y), err_msg="Box2 pos")

    # box3 pos
    exp_b3_x = (
        exp_b2_x + BOXW + MAR[1][1] + GAP + MAR[2][3]
    )  # prev X + prev W + prev R mar + gap + own L mar
    exp_b3_y = content_y + MAR[2][0]  # content start + own top margin (align_items='start')
    assert_allclose(get_world_origin(box3), (exp_b3_x, exp_b3_y), err_msg="Box3 pos")


def test_column_layout_heterogeneous_padding_margin_center():
    """test column layout, center justification, center alignment with non-uniform padding and margin"""
    PAD = (10, 15, 20, 5)  # t, r, b, l
    MAR = [(5, 2, 10, 3), (8, 4, 6, 1), (12, 7, 9, 5)]  # t, r, b, l for each child
    GAP = 8.0
    BOXW = 40.0
    BOXH = 30.0
    PWIDTH = 150.0
    PHEIGHT = 400.0

    parent = Container(
        id="hetero-pad-mar-col-center",
        min_dimensions=Size(width=PWIDTH, height=PHEIGHT),
        style=BoxStyle(padding=PAD),
        layout=LayoutConstraints(
            direction="column",
            align_items="center",  # center horizontally (considering L/R margins)
            justify_content="center",  # center vertically (considering T/B margins and gap)
            gap=GAP,
        ),
    )
    box1 = Container(
        id="b1",
        min_dimensions=Size(width=BOXW, height=BOXH),
        style=BoxStyle(background_color="red", margin=MAR[0]),
    )
    box2 = Container(
        id="b2",
        min_dimensions=Size(width=BOXW + 10, height=BOXH),  # wider
        style=BoxStyle(background_color="green", margin=MAR[1]),
    )
    box3 = Container(
        id="b3",
        min_dimensions=Size(width=BOXW - 10, height=BOXH),  # narrower
        style=BoxStyle(background_color="blue", margin=MAR[2]),
    )

    parent.add_children([box1, box2, box3])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    ax = renderer.create_context(width=200, height=500)
    renderer.render_component(ax, parent)

    # calculations
    content_x = PAD[3]
    content_y = PAD[0]
    content_w = PWIDTH - PAD[3] - PAD[1]
    content_h = PHEIGHT - PAD[0] - PAD[2]

    # total height required by children + margins + gaps
    total_child_h_mar = (
        (BOXH + MAR[0][0] + MAR[0][2])
        + (BOXH + MAR[1][0] + MAR[1][2])
        + (BOXH + MAR[2][0] + MAR[2][2])
    )
    total_gap_h = (len(parent.children) - 1) * GAP
    required_h = total_child_h_mar + total_gap_h
    extra_h = content_h - required_h
    start_y_offset = extra_h / 2.0  # offset for justify_content='center'

    # box1 pos
    exp_b1_x = (
        content_x + MAR[0][3] + (content_w - MAR[0][3] - MAR[0][1] - BOXW) / 2.0
    )  # center align
    exp_b1_y = content_y + start_y_offset + MAR[0][0]  # justify center + own top margin
    assert_allclose(get_world_origin(box1), (exp_b1_x, exp_b1_y), err_msg="Box1 pos")

    # box2 pos
    exp_b2_x = (
        content_x + MAR[1][3] + (content_w - MAR[1][3] - MAR[1][1] - (BOXW + 10)) / 2.0
    )  # center align wider box
    exp_b2_y = (
        exp_b1_y + BOXH + MAR[0][2] + GAP + MAR[1][0]
    )  # prev Y + prev H + prev B mar + gap + own T mar
    assert_allclose(get_world_origin(box2), (exp_b2_x, exp_b2_y), err_msg="Box2 pos")

    # box3 pos
    exp_b3_x = (
        content_x + MAR[2][3] + (content_w - MAR[2][3] - MAR[2][1] - (BOXW - 10)) / 2.0
    )  # center align narrower box
    exp_b3_y = (
        exp_b2_y + BOXH + MAR[1][2] + GAP + MAR[2][0]
    )  # prev Y + prev H + prev B mar + gap + own T mar
    assert_allclose(get_world_origin(box3), (exp_b3_x, exp_b3_y), err_msg="Box3 pos")


def test_row_layout_space_between_stretch_margins():
    """test row layout, space-between, stretch alignment with non-uniform padding and margin"""
    PAD = (5, 10, 15, 20)  # t, r, b, l
    MAR = [(2, 8, 4, 6), (10, 3, 5, 7), (6, 9, 11, 2)]  # t, r, b, l for each child
    GAP = 12.0
    BOXW = 50.0
    PWIDTH = 400.0
    PHEIGHT = 100.0  # relatively short parent height

    parent = Container(
        id="hetero-stretch-between-row",
        min_dimensions=Size(width=PWIDTH, height=PHEIGHT),
        style=BoxStyle(padding=PAD, background_color="#eee"),
        layout=LayoutConstraints(
            direction="row",
            align_items="stretch",  # stretch vertically (respecting T/B margins)
            justify_content="space-between",  # distribute horizontally (respecting L/R margins and gap)
            gap=GAP,
        ),
    )
    box1 = Container(
        id="b1",
        min_dimensions=Size(width=BOXW, height=20),  # natural height 20
        style=BoxStyle(background_color="red", margin=MAR[0]),
    )
    box2 = Container(
        id="b2",
        min_dimensions=Size(width=BOXW, height=40),  # natural height 40
        style=BoxStyle(background_color="green", margin=MAR[1]),
    )
    box3 = Container(
        id="b3",
        min_dimensions=Size(width=BOXW, height=10),  # natural height 10
        style=BoxStyle(background_color="blue", margin=MAR[2]),
    )

    parent.add_children([box1, box2, box3])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    fig, ax = plt.subplots()  # only for visual debug
    renderer.render_component(ax, parent)  # only for visual debug

    # calculations
    content_x = PAD[3]  # 20
    content_y = PAD[0]  # 5
    content_w = PWIDTH - PAD[3] - PAD[1]  # 400 - 20 - 10 = 370
    content_h = PHEIGHT - PAD[0] - PAD[2]  # 100 - 5 - 15 = 80 (available height for stretching)

    # expected stretched heights
    exp_h1 = max(0, content_h - MAR[0][0] - MAR[0][2])  # 80 - 2 - 4 = 74
    exp_h2 = max(0, content_h - MAR[1][0] - MAR[1][2])  # 80 - 10 - 5 = 65
    exp_h3 = max(0, content_h - MAR[2][0] - MAR[2][2])  # 80 - 6 - 11 = 63

    # check dimensions first (convert Size object to tuple for comparison)
    assert_allclose(
        (box1._dimensions.width, box1._dimensions.height),
        (BOXW, exp_h1),
        err_msg="Box1 dimensions (stretch)",
    )
    assert_allclose(
        (box2._dimensions.width, box2._dimensions.height),
        (BOXW, exp_h2),
        err_msg="Box2 dimensions (stretch)",
    )
    assert_allclose(
        (box3._dimensions.width, box3._dimensions.height),
        (BOXW, exp_h3),
        err_msg="Box3 dimensions (stretch)",
    )

    # total width required by children + margins + gaps
    total_child_w_mar = (
        (BOXW + MAR[0][3] + MAR[0][1])  # 50 + 6 + 8 = 64
        + (BOXW + MAR[1][3] + MAR[1][1])  # 50 + 7 + 3 = 60
        + (BOXW + MAR[2][3] + MAR[2][1])  # 50 + 2 + 9 = 61
    )  # total = 185
    total_gap_w = (len(parent.children) - 1) * GAP  # 2 * 12 = 24
    required_w = total_child_w_mar + total_gap_w  # 185 + 24 = 209
    extra_w = content_w - required_w  # 370 - 209 = 161
    spacing = (
        extra_w / (len(parent.children) - 1) if len(parent.children) > 1 else 0
    )  # 161 / 2 = 80.5

    # box1 pos (stretch aligns top margin with content_y)
    exp_b1_x = content_x + MAR[0][3]  # 20 + 6 = 26
    exp_b1_y = content_y + MAR[0][0]  # 5 + 2 = 7
    assert_allclose(get_world_origin(box1), (exp_b1_x, exp_b1_y), err_msg="Box1 pos")

    # box2 pos
    exp_b2_x = (
        exp_b1_x + BOXW + MAR[0][1] + GAP + MAR[1][3] + spacing
    )  # 26 + 50 + 8 + 12 + 7 + 80.5 = 183.5
    exp_b2_y = content_y + MAR[1][0]  # 5 + 10 = 15
    assert_allclose(get_world_origin(box2), (exp_b2_x, exp_b2_y), err_msg="Box2 pos")

    # box3 pos
    exp_b3_x = (
        exp_b2_x + BOXW + MAR[1][1] + GAP + MAR[2][3] + spacing
    )  # 183.5 + 50 + 3 + 12 + 2 + 80.5 = 331
    exp_b3_y = content_y + MAR[2][0]  # 5 + 6 = 11
    assert_allclose(get_world_origin(box3), (exp_b3_x, exp_b3_y), err_msg="Box3 pos")


def test_nested_heterogeneous_padding_margin():
    """test nested containers with different paddings and margins interacting"""
    OPAD = (10, 10, 10, 10)  # outer padding: t, r, b, l
    IPAD = (5, 15, 10, 20)  # inner padding: t, r, b, l
    IMAR = (8, 2, 12, 4)  # inner container's margin: t, r, b, l
    GMAR = (3, 6, 9, 1)  # grandchild's margin: t, r, b, l
    BOXW = 30.0
    BOXH = 20.0
    IWIDTH = 150.0
    IHEIGHT = 100.0
    OWIDTH = 300.0
    OHEIGHT = 200.0

    grandchild = Container(
        id="gc",
        min_dimensions=Size(width=BOXW, height=BOXH),
        style=BoxStyle(background_color="cyan", margin=GMAR),
    )
    inner_child = Container(
        id="ic",
        min_dimensions=Size(width=IWIDTH, height=IHEIGHT),
        style=BoxStyle(background_color="lightgreen", padding=IPAD, margin=IMAR),
        layout=LayoutConstraints(direction="column", align_items="start", justify_content="end"),
        children=[grandchild],
    )
    outer_parent = Container(
        id="op",
        min_dimensions=Size(width=OWIDTH, height=OHEIGHT),
        style=BoxStyle(background_color="#eee", padding=OPAD),
        layout=LayoutConstraints(direction="row", align_items="center", justify_content="center"),
        children=[inner_child],
    )

    renderer = MatplotlibRenderer()
    outer_parent.measure_and_layout(renderer)
    fig, ax = plt.subplots()  # only for visual debug
    renderer.render_component(ax, outer_parent)  # only for visual debug

    # calculations
    op_content_x = OPAD[3]  # 10
    op_content_y = OPAD[0]  # 10
    op_content_w = OWIDTH - OPAD[3] - OPAD[1]  # 300 - 10 - 10 = 280
    op_content_h = OHEIGHT - OPAD[0] - OPAD[2]  # 200 - 10 - 10 = 180

    # inner_child position within outer_parent (centered)
    avail_op_w = op_content_w - IMAR[3] - IMAR[1]  # 280 - 4 - 2 = 274
    avail_op_h = op_content_h - IMAR[0] - IMAR[2]  # 180 - 8 - 12 = 160
    exp_ic_x = (
        op_content_x + IMAR[3] + (avail_op_w - IWIDTH) / 2.0
    )  # 10 + 4 + (274 - 150)/2 = 14 + 62 = 76
    exp_ic_y = (
        op_content_y + IMAR[0] + (avail_op_h - IHEIGHT) / 2.0
    )  # 10 + 8 + (160 - 100)/2 = 18 + 30 = 48
    assert_allclose(get_world_origin(inner_child), (exp_ic_x, exp_ic_y), err_msg="Inner Child pos")

    # grandchild position within inner_child
    ic_content_x = IPAD[3]  # 20
    ic_content_y = IPAD[0]  # 5
    ic_content_w = IWIDTH - IPAD[3] - IPAD[1]  # 150 - 20 - 15 = 115
    ic_content_h = IHEIGHT - IPAD[0] - IPAD[2]  # 100 - 5 - 10 = 85

    # align='start' -> use left padding + own left margin
    exp_gc_local_x = ic_content_x + GMAR[3]  # 20 + 1 = 21
    # justify='end' -> align bottom edge (plus bottom margin) with bottom of content box
    exp_gc_local_y = ic_content_y + ic_content_h - GMAR[2] - BOXH  # 5 + 85 - 9 - 20 = 61
    # world position
    exp_gc_world_x = exp_ic_x + exp_gc_local_x  # 76 + 21 = 97
    exp_gc_world_y = exp_ic_y + exp_gc_local_y  # 48 + 61 = 109
    assert_allclose(
        get_world_origin(grandchild), (exp_gc_world_x, exp_gc_world_y), err_msg="Grandchild pos"
    )


def test_row_layout_margins_with_offsets():
    """test combining layout positioning (with margins) and explicit child offsets"""
    PAD = (10, 5, 20, 15)  # t, r, b, l
    MAR = (8, 4, 6, 2)  # t, r, b, l for child
    OFF = Offset(relative=(0.1, -0.2), absolute=(5, -3))
    GAP = 10.0
    BOXW = 60.0
    BOXH = 40.0
    PWIDTH = 300.0
    PHEIGHT = 150.0

    parent = Container(
        id="margins-offsets-row",
        min_dimensions=Size(width=PWIDTH, height=PHEIGHT),
        style=BoxStyle(padding=PAD, background_color="#eee"),
        layout=LayoutConstraints(
            direction="row",
            align_items="end",  # align bottom edges
            justify_content="center",  # center horizontally
            gap=GAP,
        ),
    )
    box1 = Container(
        id="b1",  # reference box, no offset
        min_dimensions=Size(width=BOXW, height=BOXH),
        style=BoxStyle(background_color="red", margin=MAR),
    )
    box2 = Container(
        id="b2",  # box with offset
        min_dimensions=Size(width=BOXW, height=BOXH),
        style=BoxStyle(background_color="green", margin=MAR),
        offset=OFF,
    )

    parent.add_children([box1, box2])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    fig, ax = plt.subplots()  # only for visual debug
    renderer.render_component(ax, parent)  # only for visual debug

    # calculations
    content_x = PAD[3]  # 15
    content_y = PAD[0]  # 10
    content_w = PWIDTH - PAD[3] - PAD[1]  # 300 - 15 - 5 = 280
    content_h = PHEIGHT - PAD[0] - PAD[2]  # 150 - 10 - 20 = 120

    # total width required by children + margins + gaps
    total_child_w_mar = 2 * (BOXW + MAR[3] + MAR[1])  # 2 * (60 + 2 + 4) = 132
    total_gap_w = GAP  # 10
    required_w = total_child_w_mar + total_gap_w  # 132 + 10 = 142
    extra_w = content_w - required_w  # 280 - 142 = 138
    start_x_offset = extra_w / 2.0  # 69

    # --- box1 (layout only) ---
    # layout position calculation
    layout_b1_x = content_x + start_x_offset + MAR[3]  # 15 + 69 + 2 = 86
    # align_items='end' -> bottom edge (+ margin) aligns with bottom of content box
    # y = content_y + content_h - bottom_margin - box_height
    layout_b1_y = content_y + content_h - MAR[2] - BOXH  # 10 + 120 - 6 - 40 = 84
    # final position = layout position (no offset)
    exp_b1_x = layout_b1_x
    exp_b1_y = layout_b1_y
    assert_allclose(get_world_origin(box1), (exp_b1_x, exp_b1_y), err_msg="Box1 pos (layout)")

    # --- box2 (layout + offset) ---
    # layout position calculation (same logic as box1, just shifted by box1's presence)
    # x = prev_x + prev_w + prev_r_mar + gap + own_l_mar
    layout_b2_x = exp_b1_x + BOXW + MAR[1] + GAP + MAR[3]  # 86 + 60 + 4 + 10 + 2 = 162
    layout_b2_y = layout_b1_y  # same vertical alignment due to align_items='end' -> 84
    # offset calculation (delta added to layout position)
    # delta_x = rel_x * W_self + parent_rel_x * W_parent + abs_x
    # delta_y = rel_y * H_self + parent_rel_y * H_parent + abs_y
    delta_b2_x, delta_b2_y = OFF.compute(box2._dimensions, parent._dimensions)
    # delta_b2_x = 0.1 * 60 + 0 * 300 + 5 = 6 + 5 = 11
    # delta_b2_y = -0.2 * 40 + 0 * 150 + (-3) = -8 - 3 = -11
    assert_allclose((delta_b2_x, delta_b2_y), (11, -11), err_msg="Box2 offset delta")
    # final position = layout position + offset delta
    exp_b2_x = layout_b2_x + delta_b2_x  # 162 + 11 = 173
    exp_b2_y = layout_b2_y + delta_b2_y  # 84 + (-11) = 73
    assert_allclose(
        get_world_origin(box2), (exp_b2_x, exp_b2_y), err_msg="Box2 pos (layout+offset)"
    )


if __name__ == "__main__":
    test_row_layout()
    plt.show()
    test_basic_row_space_between_center()
    plt.show()
    test_layout_children_offsets()
    plt.show()
    test_overlay_positioning()
    plt.show()
    test_nested_layout_and_overlay()
    plt.show()
    test_complex_layout()

    plt.show()
    test_simple_attachment_no_offset()
    plt.show()
    test_attachment_with_offset()
    plt.show()
    test_attachment_string_path_nested()
    plt.show()
    test_simple_connection_centers()
    plt.show()
    test_connection_with_anchors()
    plt.show()
    test_overlay_container_internal_layout()
    plt.show()

    test_row_layout_heterogeneous_padding_margin_start()
    plt.show()
    test_column_layout_heterogeneous_padding_margin_center()
    plt.show()
    test_row_layout_heterogeneous_padding_margin_start()
    plt.show()
    test_row_layout_space_between_stretch_margins()
    plt.show()
    test_nested_heterogeneous_padding_margin()
    plt.show()
    test_row_layout_margins_with_offsets()
    plt.show()
