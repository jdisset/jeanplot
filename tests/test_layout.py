# File: tests/test_layout.py
import pytest
from jeanplot.component import Component
import numpy as np
import matplotlib.pyplot as plt
from jeanplot.models import Transform, Size, BoxStyle, LayoutConstraints, Offset
from jeanplot.style import jstyle
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.container import Container
from jeanplot.text import Text
from jeanplot.component import Overlay, AnchorComponent
from numpy.testing import assert_allclose
from jeanplot.debug import set_debug
import math
from jeanplot.connector import (
    Connection,
    StraightCurve,
    SimpleBezierCurve,
    OrthogonalCurve,
)
from jeanplot.path_utils import find_component_by_path
from pydantic import BaseModel, Field

set_debug(False)


def get_world_origin(component: Component):
    return component.get_world_origin()


@pytest.fixture(autouse=True)
def setup_test():
    set_debug(False)  # disable debug by default for tests unless explicitly enabled
    jstyle.clear()  # ensure styles are cleared before each test
    yield
    set_debug(False)
    jstyle.clear()
    # close all figures to avoid display issues
    plt.close("all")


def test_row_layout():
    """test row layout with different alignment options"""
    set_debug(True)  # enable for this specific test if needed

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
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    container.measure_and_layout(renderer)
    renderer.render_component(ax, container)  # keep commented unless debugging interactively

    cont_pos = container.get_world_origin()
    assert_allclose(cont_pos, (0, 0), err_msg="Container origin")
    assert container._dimensions == Size(width=CWIDTH, height=CHEIGHT), "Container dims"
    assert len(container.children) == 3, "Child count"

    # box1:
    box1_pos = box1.get_world_origin()
    exp_b1_x = PAD  # parent left padding
    exp_b1_y = PAD  # parent top padding
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
    assert_allclose(
        box2_pos,
        (exp_b2_x, exp_b2_y),
        err_msg=f"Box2 position exp=({exp_b2_x:.1f},{exp_b2_y:.1f}) got={box2_pos}",
    )

    # check box3 position relative to box2 + spacing
    exp_b3_x = exp_b2_x + B2W + GAP + spacing
    exp_b3_y = PAD  # align_items=start
    assert_allclose(
        box3_pos,
        (exp_b3_x, exp_b3_y),
        err_msg=f"Box3 position exp=({exp_b3_x:.1f},{exp_b3_y:.1f}) got={box3_pos}",
    )


def test_basic_row_space_between_center():
    """test row layout: space-between justification, center alignment, no child offsets"""
    set_debug(True)  # enable for this specific test if needed

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
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, parent)  # keep commented unless debugging interactively

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
    set_debug(True)  # enable for this specific test if needed

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
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, parent)  # keep commented unless debugging interactively

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
    user_offset_b2 = Offset(relative=(B2_R_OFF_X, B2_R_OFF_Y))
    delta_b2_x, delta_b2_y = user_offset_b2.compute(box2._dimensions, parent._dimensions)
    exp_b2_x = exp_base_b2_x + delta_b2_x
    exp_b2_y = exp_base_b2_y + delta_b2_y
    assert_allclose(pos_b2, (exp_b2_x, exp_b2_y), err_msg="Box2 final pos")

    # box3 (absolute offset)
    pos_b3 = get_world_origin(box3)
    user_offset_b3 = Offset(absolute=(B3_ABS_OFF_X, B3_ABS_OFF_Y))
    delta_b3_x, delta_b3_y = user_offset_b3.compute(box3._dimensions, parent._dimensions)
    exp_b3_x = exp_base_b3_x + delta_b3_x
    exp_b3_y = exp_base_b3_y + delta_b3_y
    assert_allclose(pos_b3, (exp_b3_x, exp_b3_y), err_msg="Box3 final pos")

    # box4 (parent relative offset)
    pos_b4 = get_world_origin(box4)
    user_offset_b4 = Offset(parent_relative=(B4_PR_OFF_X, B4_PR_OFF_Y))
    delta_b4_x, delta_b4_y = user_offset_b4.compute(box4._dimensions, parent._dimensions)
    exp_b4_x = exp_base_b4_x + delta_b4_x
    exp_b4_y = exp_base_b4_y + delta_b4_y
    assert_allclose(pos_b4, (exp_b4_x, exp_b4_y), err_msg="Box4 final pos")

    # box5 (mixed offset)
    pos_b5 = get_world_origin(box5)
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
    set_debug(True)  # enable for this specific test if needed

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

    box6 = Container(
        id="overlay-no-offset",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="lightyellow"),
        is_overlay=True,
    )

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
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    ax.set_aspect("equal")
    renderer.render_component(
        ax, parent, adjust_lims=True
    )  # keep commented unless debugging interactively

    # --- assertions ---
    pos_p = get_world_origin(parent)
    assert_allclose(pos_p, (0, 0))

    # box6 (overlay, no offset)
    pos_b6 = get_world_origin(box6)
    exp_b6_x, exp_b6_y = (0, 0)
    assert_allclose(pos_b6, (exp_b6_x, exp_b6_y), err_msg="Box6 overlay pos")
    assert box6._dimensions == Size(width=BOXW, height=BOXW), "Box6 dims"

    # box7 (overlay, with offset)
    pos_b7 = get_world_origin(box7)
    offset_b7_x, offset_b7_y = box7.offset.compute(box7._dimensions, parent._dimensions)
    exp_b7_x = offset_b7_x
    exp_b7_y = offset_b7_y
    assert_allclose(pos_b7, (exp_b7_x, exp_b7_y), err_msg="Box7 overlay w/ offset pos")
    assert box7._dimensions == Size(width=BOXW, height=BOXW), "Box7 dims"


def test_nested_layout_and_overlay():
    """test internal layout and overlay positioning within a nested container"""
    set_debug(True)  # enable for this specific test if needed

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
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, parent)  # keep commented unless debugging interactively

    # Calculations for Box8 internal layout
    b8_content_w = box8._dimensions.width
    b8_content_h = box8._dimensions.height
    b8_num_layout_children = 2
    b8_total_children_height = b8_num_layout_children * IBOXH
    b8_total_gap_height = (b8_num_layout_children - 1) * GAP
    b8_total_required_height = b8_total_children_height + b8_total_gap_height
    b8_extra_space = b8_content_h - b8_total_required_height
    b8_spacing = b8_extra_space / (b8_num_layout_children - 1) if b8_num_layout_children > 1 else 0
    b8_x_align_offset = (b8_content_w - IBOXW) / 2.0

    # Assertions
    pos_p = get_world_origin(parent)
    assert_allclose(pos_p, (0, 0))

    pos_b8 = get_world_origin(box8)
    exp_b8_x = PAD + box8.offset.absolute[0]
    exp_b8_y = PAD + box8.offset.absolute[1]
    assert_allclose(pos_b8, (exp_b8_x, exp_b8_y), err_msg="Box8 position")
    assert box8._dimensions == Size(width=BOXW, height=BOXW), "Box8 dims"

    pos_ib1 = get_world_origin(inner_box1)
    offset_ib1_x, offset_ib1_y = inner_box1.offset.compute(inner_box1._dimensions, box8._dimensions)
    exp_ib1_x = pos_b8[0] + offset_ib1_x
    exp_ib1_y = pos_b8[1] + offset_ib1_y
    assert_allclose(pos_ib1, (exp_ib1_x, exp_ib1_y), err_msg="Inner Box 1 overlay pos")
    assert inner_box1._dimensions == Size(width=IBOXW, height=IBOXH), "Inner Box 1 dims"

    pos_ib2 = get_world_origin(inner_box2)
    local_ib2_x = b8_x_align_offset
    local_ib2_y = 0
    exp_ib2_x = pos_b8[0] + local_ib2_x
    exp_ib2_y = pos_b8[1] + local_ib2_y
    assert_allclose(pos_ib2, (exp_ib2_x, exp_ib2_y), err_msg="Inner Box 2 layout pos")
    assert inner_box2._dimensions == Size(width=IBOXW, height=IBOXH), "Inner Box 2 dims"

    pos_ib3 = get_world_origin(inner_box3)
    local_ib3_x = b8_x_align_offset
    local_ib3_y = local_ib2_y + IBOXH + GAP + b8_spacing
    exp_ib3_x = pos_b8[0] + local_ib3_x
    exp_ib3_y = pos_b8[1] + local_ib3_y
    assert_allclose(pos_ib3, (exp_ib3_x, exp_ib3_y), err_msg="Inner Box 3 layout pos")
    assert inner_box3._dimensions == Size(width=IBOXW, height=IBOXH), "Inner Box 3 dims"


def test_complex_layout():
    """test row layout with complex offsets and overlays"""
    set_debug(True)  # enable for this specific test if needed

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

    box6 = Overlay(
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
    box7 = Container(
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
    inner_box1 = Container(
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

    inner_box2 = Container(
        id="inner-box2",
        min_dimensions=Size(width=IBOXW, height=IBOXH),
        style=BoxStyle(background_color="darkgreen", border_color="green", border_width=1),
    )

    inner_box3 = Container(
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
    box8 = Container(
        id="box8-nested",
        min_dimensions=Size(width=BOXW, height=BOXW),
        style=BoxStyle(background_color="#dddddd", border_color="black", border_width=1),
        layout=LayoutConstraints(
            direction="column",
            align_items="center",
            justify_content="space-between",
            gap=GAP,
        ),
        children=[inner_box1, inner_box2, inner_box3],
        offset=Offset(
            relative=(B8_R_OFF_X, B8_R_OFF_Y),
            parent_relative=(B8_PR_OFF_X, B8_PR_OFF_Y),
            absolute=(B8_ABS_OFF_X, B8_ABS_OFF_Y),
        ),
    )

    parent.children = [box1, box2, box3, box4, box5, box8, box6, box7]

    renderer = MatplotlibRenderer()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    parent.measure_and_layout(renderer)
    renderer.render_component(ax, parent)  # keep commented unless debugging interactively

    # Calculations
    content_w = PWIDTH - 2 * PAD
    content_h = PHEIGHT - 2 * PAD
    content_x = PAD
    content_y = PAD
    num_layout_children = 6
    total_layout_children_width = num_layout_children * BOXW
    total_gap_width = (num_layout_children - 1) * GAP
    total_required_width = total_layout_children_width + total_gap_width
    extra_space = content_w - total_required_width
    spacing = extra_space / (num_layout_children - 1) if num_layout_children > 1 else 0
    y_align_offset = (content_h - BOXW) / 2.0

    # Expected base positions
    exp_base_b1_x = content_x
    exp_base_b1_y = content_y + y_align_offset
    exp_base_b2_x = exp_base_b1_x + BOXW + GAP + spacing
    exp_base_b2_y = exp_base_b1_y
    exp_base_b3_x = exp_base_b2_x + BOXW + GAP + spacing
    exp_base_b3_y = exp_base_b2_y
    exp_base_b4_x = exp_base_b3_x + BOXW + GAP + spacing
    exp_base_b4_y = exp_base_b3_y
    exp_base_b5_x = exp_base_b4_x + BOXW + GAP + spacing
    exp_base_b5_y = exp_base_b4_y
    exp_base_b8_x = exp_base_b5_x + BOXW + GAP + spacing
    exp_base_b8_y = exp_base_b5_y

    # Parent Assertions
    pos_parent = get_world_origin(parent)
    assert_allclose(pos_parent, (0, 0), err_msg="Parent should be at origin")
    assert parent._dimensions == Size(width=PWIDTH, height=PHEIGHT), "Parent dimensions mismatch"

    # Box1 Assertions (No Offset)
    pos_b1 = get_world_origin(box1)
    assert_allclose(pos_b1, (exp_base_b1_x, exp_base_b1_y), err_msg="Box1 position mismatch")
    assert box1._dimensions == Size(width=BOXW, height=BOXW), "Box1 dimensions mismatch"

    # Box2 Assertions (Relative Offset)
    pos_b2 = get_world_origin(box2)
    user_offset_b2 = Offset(relative=(B2_R_OFF_X, B2_R_OFF_Y))
    delta_b2_x, delta_b2_y = user_offset_b2.compute(box2._dimensions, parent._dimensions)
    exp_b2_x_final = exp_base_b2_x + delta_b2_x
    exp_b2_y_final = exp_base_b2_y + delta_b2_y
    assert_allclose(pos_b2, (exp_b2_x_final, exp_b2_y_final), err_msg="Box2 position mismatch")
    assert box2._dimensions == Size(width=BOXW, height=BOXW), "Box2 dimensions mismatch"

    # Box3 Assertions (Absolute Offset)
    pos_b3 = get_world_origin(box3)
    user_offset_b3 = Offset(absolute=(B3_ABS_OFF_X, B3_ABS_OFF_Y))
    delta_b3_x, delta_b3_y = user_offset_b3.compute(box3._dimensions, parent._dimensions)
    exp_b3_x_final = exp_base_b3_x + delta_b3_x
    exp_b3_y_final = exp_base_b3_y + delta_b3_y
    assert_allclose(pos_b3, (exp_b3_x_final, exp_b3_y_final), err_msg="Box3 position mismatch")
    assert box3._dimensions == Size(width=BOXW, height=BOXW), "Box3 dimensions mismatch"

    # Box4 Assertions (Parent Relative Offset)
    pos_b4 = get_world_origin(box4)
    user_offset_b4 = Offset(parent_relative=(B4_PR_OFF_X, B4_PR_OFF_Y))
    delta_b4_x, delta_b4_y = user_offset_b4.compute(box4._dimensions, parent._dimensions)
    exp_b4_x_final = exp_base_b4_x + delta_b4_x
    exp_b4_y_final = exp_base_b4_y + delta_b4_y
    assert_allclose(pos_b4, (exp_b4_x_final, exp_b4_y_final), err_msg="Box4 position mismatch")
    assert box4._dimensions == Size(width=BOXW, height=BOXW), "Box4 dimensions mismatch"

    # Box5 Assertions (Mixed Offset)
    pos_b5 = get_world_origin(box5)
    user_offset_b5 = Offset(
        relative=(B5_R_OFF_X, B5_R_OFF_Y),
        parent_relative=(B5_PR_OFF_X, B5_PR_OFF_Y),
        absolute=(B5_ABS_OFF_X, B5_ABS_OFF_Y),
    )
    delta_b5_x, delta_b5_y = user_offset_b5.compute(box5._dimensions, parent._dimensions)
    exp_b5_x_final = exp_base_b5_x + delta_b5_x
    exp_b5_y_final = exp_base_b5_y + delta_b5_y
    assert_allclose(pos_b5, (exp_b5_x_final, exp_b5_y_final), err_msg="Box5 position mismatch")
    assert box5._dimensions == Size(width=BOXW, height=BOXW), "Box5 dimensions mismatch"

    # Box8 Assertions (Container with Offset)
    pos_b8 = get_world_origin(box8)
    user_offset_b8 = Offset(
        relative=(B8_R_OFF_X, B8_R_OFF_Y),
        parent_relative=(B8_PR_OFF_X, B8_PR_OFF_Y),
        absolute=(B8_ABS_OFF_X, B8_ABS_OFF_Y),
    )
    delta_b8_x, delta_b8_y = user_offset_b8.compute(box8._dimensions, parent._dimensions)
    exp_b8_x_final = exp_base_b8_x + delta_b8_x
    exp_b8_y_final = exp_base_b8_y + delta_b8_y
    assert_allclose(pos_b8, (exp_b8_x_final, exp_b8_y_final), err_msg="Box8 position mismatch")
    assert box8._dimensions == Size(width=BOXW, height=BOXW), "Box8 dimensions mismatch"

    # Box6 Assertions (Overlay, No Offset)
    pos_b6 = get_world_origin(box6)
    exp_b6_x, exp_b6_y = (0, 0)
    assert_allclose(pos_b6, (exp_b6_x, exp_b6_y), err_msg="Box6 (Overlay) position mismatch")
    assert box6._dimensions == Size(width=BOXW, height=BOXW), "Box6 dimensions mismatch"

    # Box7 Assertions (Overlay, Mixed Offset)
    pos_b7 = get_world_origin(box7)
    offset_b7_x, offset_b7_y = box7.offset.compute(box7._dimensions, parent._dimensions)
    exp_b7_x = 0 + offset_b7_x
    exp_b7_y = 0 + offset_b7_y
    assert_allclose(
        pos_b7, (exp_b7_x, exp_b7_y), err_msg="Box7 (Overlay w/ Offset) position mismatch"
    )
    assert box7._dimensions == Size(width=BOXW, height=BOXW), "Box7 dimensions mismatch"

    # Box8 Internal Layout Calculations
    b8_content_w = box8._dimensions.width
    b8_content_h = box8._dimensions.height
    b8_num_layout_children = 2
    b8_total_children_height = b8_num_layout_children * IBOXH
    b8_total_gap_height = (b8_num_layout_children - 1) * GAP
    b8_total_required_height = b8_total_children_height + b8_total_gap_height
    b8_extra_space = b8_content_h - b8_total_required_height
    b8_spacing = b8_extra_space / (b8_num_layout_children - 1) if b8_num_layout_children > 1 else 0
    b8_x_align_offset = (b8_content_w - IBOXW) / 2.0

    # Inner_box1 Assertions (Overlay inside Box8)
    pos_ib1 = get_world_origin(inner_box1)
    offset_ib1_x, offset_ib1_y = inner_box1.offset.compute(inner_box1._dimensions, box8._dimensions)
    exp_ib1_x = pos_b8[0] + offset_ib1_x
    exp_ib1_y = pos_b8[1] + offset_ib1_y
    assert_allclose(
        pos_ib1, (exp_ib1_x, exp_ib1_y), err_msg="Inner Box 1 (Overlay) position mismatch"
    )
    assert inner_box1._dimensions == Size(
        width=IBOXW, height=IBOXH
    ), "Inner Box 1 dimensions mismatch"

    # Inner_box2 Assertions (Layout Child inside Box8)
    pos_ib2 = get_world_origin(inner_box2)
    local_ib2_x = b8_x_align_offset
    local_ib2_y = 0
    exp_ib2_x = pos_b8[0] + local_ib2_x
    exp_ib2_y = pos_b8[1] + local_ib2_y
    assert_allclose(pos_ib2, (exp_ib2_x, exp_ib2_y), err_msg="Inner Box 2 position mismatch")
    assert inner_box2._dimensions == Size(
        width=IBOXW, height=IBOXH
    ), "Inner Box 2 dimensions mismatch"

    # Inner_box3 Assertions (Layout Child inside Box8)
    pos_ib3 = get_world_origin(inner_box3)
    local_ib3_x = b8_x_align_offset
    local_ib3_y = local_ib2_y + IBOXH + GAP + b8_spacing
    exp_ib3_x = pos_b8[0] + local_ib3_x
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
    set_debug(True)  # enable for this specific test if needed

    PAD = 10.0
    ROOT_SIZE = 200.0
    TARGET_SIZE = 100.0
    ATTACHED_SIZE = 20.0

    target = Container(
        id="target",
        min_dimensions=Size(width=TARGET_SIZE, height=TARGET_SIZE),
        style=BoxStyle(border_color="red", border_width=2),
        offset=Offset(absolute=(50, 50)),
    )
    attached = Container(
        id="attached",
        min_dimensions=Size(width=ATTACHED_SIZE, height=ATTACHED_SIZE),
        style=BoxStyle(background_color="lightgreen"),
        attached_to=target,
    )

    root = Container(
        id="attach-test-1",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        children=[target, attached],
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, root)  # keep commented unless debugging interactively

    # Assertions
    pos_target = get_world_origin(target)
    exp_target_x = PAD + 50
    exp_target_y = PAD + 50
    assert_allclose(pos_target, (exp_target_x, exp_target_y), err_msg="Target position")

    pos_attached = get_world_origin(attached)
    exp_attached_x = exp_target_x
    exp_attached_y = exp_target_y
    assert_allclose(
        pos_attached, (exp_attached_x, exp_attached_y), err_msg="Attached position (no offset)"
    )
    assert attached._resolved_attach_target is target, "Attachment target resolution failed"


def test_attachment_with_offset():
    """
    Test attaching one component relative to another using attachment_offset.
    Expected Visual: A small green square positioned relative to the top-right corner of a larger red-border square.
    """
    set_debug(True)  # enable for this specific test if needed

    PAD = 10.0
    ROOT_SIZE = 200.0
    TARGET_SIZE = 100.0
    ATTACHED_SIZE = 20.0

    target = Container(
        id="target",
        min_dimensions=Size(width=TARGET_SIZE, height=TARGET_SIZE),
        style=BoxStyle(border_color="red", border_width=2),
        offset=Offset(absolute=(50, 50)),
    )
    attach_offset = Offset(reference_relative=(1.0, 1.0), absolute=(5, -10), relative=(-0.5, -0.5))
    attached = Container(
        id="attached",
        min_dimensions=Size(width=ATTACHED_SIZE, height=ATTACHED_SIZE),
        style=BoxStyle(background_color="lightgreen"),
        attached_to=target,
        attachment_offset=attach_offset,
    )

    root = Container(
        id="attach-test-2",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        children=[target, attached],
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, root)  # keep commented unless debugging interactively

    # Assertions
    pos_target = get_world_origin(target)
    exp_target_x = PAD + 50
    exp_target_y = PAD + 50
    assert_allclose(pos_target, (exp_target_x, exp_target_y), err_msg="Target position")

    pos_attached = get_world_origin(attached)
    # calculate expected offset from target origin:
    # target_W * parent_rel_x + self_W * rel_x + abs_x
    # target_H * parent_rel_y + self_H * rel_y + abs_y
    exp_offset_x_calc = (
        TARGET_SIZE * attach_offset.reference_relative[0]
        + ATTACHED_SIZE * attach_offset.relative[0]
        + attach_offset.absolute[0]
    )  # 100*1.0 + 20*(-0.5) + 5 = 100 - 10 + 5 = 95
    exp_offset_y_calc = (
        TARGET_SIZE * attach_offset.reference_relative[1]
        + ATTACHED_SIZE * attach_offset.relative[1]
        + attach_offset.absolute[1]
    )  # 100*1.0 + 20*(-0.5) + (-10) = 100 - 10 - 10 = 80

    exp_attached_x = exp_target_x + exp_offset_x_calc
    exp_attached_y = exp_target_y + exp_offset_y_calc
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
    set_debug(True)  # enable for this specific test if needed

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

    attached = Container(
        id="attached",
        min_dimensions=Size(width=ATTACHED_SIZE, height=ATTACHED_SIZE),
        style=BoxStyle(background_color="lightgreen"),
        attached_to="parent/child-target",
        attachment_offset=Offset(relative=(0.5, 0.5), reference_relative=(0.5, 0.5)),
    )

    root = Container(
        id="attach-test-3",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        children=[parent, attached],
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, root)  # keep commented unless debugging interactively

    # Assertions
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
    exp_offset_x_calc = 0.5 * ATTACHED_SIZE + 0.5 * CHILD_SIZE  # = 10 + 25 = 35
    exp_offset_y_calc = 0.5 * ATTACHED_SIZE + 0.5 * CHILD_SIZE  # = 10 + 25 = 35
    assert_allclose(
        (offset_x, offset_y),
        (exp_offset_x_calc, exp_offset_y_calc),
        err_msg="Attachment offset calculation",
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
    set_debug(True)  # enable for this specific test if needed

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
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
    renderer.render_component(ax, root)  # keep commented unless debugging interactively

    # Assertions
    pos_a = box_a.get_world_origin()
    exp_a_x = PAD
    exp_a_y = PAD + (ROOT_SIZE - 2 * PAD - BOX_SIZE) / 2
    assert_allclose(pos_a, (exp_a_x, exp_a_y), err_msg="Box A position")

    pos_b = box_b.get_world_origin()
    exp_b_x = exp_a_x + BOX_SIZE + GAP
    exp_b_y = exp_a_y
    assert_allclose(pos_b, (exp_b_x, exp_b_y), err_msg="Box B position")

    calculated_points = conn._get_world_connection_points()
    assert calculated_points is not None, "Failed to calculate connection points"
    conn_world_start, conn_world_end, _ = calculated_points

    exp_start_x = exp_a_x + BOX_SIZE / 2
    exp_start_y = exp_a_y + BOX_SIZE / 2
    assert_allclose(conn_world_start, (exp_start_x, exp_start_y), err_msg="Connection start point")

    exp_end_x = exp_b_x + BOX_SIZE / 2
    exp_end_y = exp_b_y + BOX_SIZE / 2
    assert_allclose(conn_world_end, (exp_end_x, exp_end_y), err_msg="Connection end point")


def test_connection_with_anchors():
    """
    Test connecting specific anchor points on two components.
    Expected Visual: Two squares (red, blue). A black orthogonal line starts
                     from the top-center anchor of the red square (going down initially),
                     turns, and goes to the bottom-center anchor of the blue square.
    """
    set_debug(True)  # enable for this specific test if needed

    PAD = 10.0
    ROOT_SIZE = 300.0
    BOX_SIZE = 50.0
    GAP = 30.0

    anchor_a_top = AnchorComponent(
        id="anchor_a",
        offset=Offset(reference_relative=(0.5, 1.0)),  # top-center of parent
        direction=(0, -1),  # pointing DOWN (corrected vector)
        min_segment=25,
    )
    anchor_b_bottom = AnchorComponent(
        id="anchor_b",
        offset=Offset(reference_relative=(0.5, 0.0)),  # bottom-center of parent
    )

    box_a = Container(
        id="box_a",
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

    conn = Connection(
        id="conn_anchors",
        start_component="box_a/anchor_a",
        end_component="box_b/anchor_b",
        curve_type=OrthogonalCurve(start_length=5.0, end_length=5.0),  # Explicit default length
        color="black",
        line_width=2,
        auto_route=False,
        start_offset=Offset(),
        end_offset=Offset(),
    )

    root = Container(
        id="conn-test-2",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        layout=LayoutConstraints(
            direction="row", align_items="center", justify_content="start", gap=GAP
        ),
        children=[box_a, box_b, conn],
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
    renderer.render_component(ax, root)  # keep commented unless debugging interactively

    # Assertions
    pos_a = get_world_origin(box_a)
    exp_a_x = PAD
    exp_a_y = PAD + (ROOT_SIZE - 2 * PAD - BOX_SIZE) / 2
    assert_allclose(pos_a, (exp_a_x, exp_a_y), err_msg="Box A position")

    pos_b = get_world_origin(box_b)
    exp_b_x = exp_a_x + BOX_SIZE + GAP
    exp_b_y = exp_a_y
    assert_allclose(pos_b, (exp_b_x, exp_b_y), err_msg="Box B position")

    pos_anchor_a = get_world_origin(anchor_a_top)
    anchor_a_local_offset = anchor_a_top.offset.compute(Size(), box_a._dimensions)
    exp_anchor_a_x = exp_a_x + anchor_a_local_offset[0]  # parent x + 0.5 * parent w = 10 + 25 = 35
    exp_anchor_a_y = (
        exp_a_y + anchor_a_local_offset[1]
    )  # parent y + 1.0 * parent h = 125 + 50 = 175
    assert_allclose(pos_anchor_a, (exp_anchor_a_x, exp_anchor_a_y), err_msg="Anchor A position")

    pos_anchor_b = get_world_origin(anchor_b_bottom)
    anchor_b_local_offset = anchor_b_bottom.offset.compute(Size(), box_b._dimensions)
    exp_anchor_b_x = exp_b_x + anchor_b_local_offset[0]  # parent x + 0.5 * parent w = 90 + 25 = 115
    exp_anchor_b_y = exp_b_y + anchor_b_local_offset[1]  # parent y + 0.0 * parent h = 125 + 0 = 125
    assert_allclose(pos_anchor_b, (exp_anchor_b_x, exp_anchor_b_y), err_msg="Anchor B position")

    calculated_points = conn._get_world_connection_points()
    assert calculated_points is not None, "Failed to calculate connection points"
    conn_world_start, conn_world_end, active_curve = calculated_points

    assert_allclose(
        conn_world_start,
        (exp_anchor_a_x, exp_anchor_a_y),
        err_msg="Connection start point (anchor)",
    )
    assert_allclose(
        conn_world_end, (exp_anchor_b_x, exp_anchor_b_y), err_msg="Connection end point (anchor)"
    )

    assert isinstance(active_curve, OrthogonalCurve), "Curve type should be OrthogonalCurve"

    # --- Corrected Assertions ---
    assert (
        active_curve.start_direction
        == "down"  # Corrected expected direction based on vector (0, -1)
    ), f"Expected start_direction 'down' (from anchor (0,-1)), got '{active_curve.start_direction}'"
    assert_allclose(
        active_curve.start_length, 25.0, err_msg="Start length mismatch (from anchor min_segment)"
    )

    assert (
        active_curve.end_direction == "auto"
    ), f"Expected end_direction 'auto' (anchor had no direction), got '{active_curve.end_direction}'"
    assert_allclose(active_curve.end_length, 5.0, err_msg="End length mismatch (default expected)")


def test_overlay_container_internal_layout():
    """
    tests that an overlay container correctly applies its internal layout
    properties (e.g., align_items, justify_content) to its own children.
    this would fail before the fix in container.apply_layout.

    expected visual: a yellow overlay box positioned absolutely. inside it,
                     a small cyan text box should be perfectly centered.
    """
    set_debug(True)  # enable for this specific test if needed

    ROOT_SIZE = 200.0
    OVERLAY_SIZE = Size(width=100, height=80)
    OVERLAY_OFFSET = Offset(absolute=(50, 60))  # absolute offset for predictability
    TEXT_CONTENT = "Centered"

    inner_text = Text(
        id="inner_text",
        text=TEXT_CONTENT,
        style=BoxStyle(background_color="cyan", border_width=0.5, border_color="blue"),
        align="center",
        vertical_align="middle",
    )

    overlay_child = Container(
        id="overlay_container",
        min_dimensions=OVERLAY_SIZE,
        max_dimensions=OVERLAY_SIZE,
        is_overlay=True,
        offset=OVERLAY_OFFSET,
        layout=LayoutConstraints(align_items="center", justify_content="center"),
        style=BoxStyle(background_color="lightyellow", border_width=1, border_color="orange"),
        children=[inner_text],
    )

    root = Container(
        id="overlay-layout-test-root",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        children=[overlay_child],
    )

    # --- execution ---
    renderer = MatplotlibRenderer()
    text_natural_size = renderer.measure_text(inner_text)
    assert text_natural_size.width > 0 and text_natural_size.height > 0, "Text measurement failed"
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    ax.set_aspect("equal")
    renderer.render_component(
        ax, root, adjust_lims=True
    )  # keep commented unless debugging interactively
    # plt.title("Overlay Internal Layout Test") # keep commented unless debugging interactively

    # --- calculations ---
    exp_overlay_origin_x = OVERLAY_OFFSET.absolute[0]
    exp_overlay_origin_y = OVERLAY_OFFSET.absolute[1]
    exp_text_local_x = (OVERLAY_SIZE.width - text_natural_size.width) / 2.0
    exp_text_local_y = (OVERLAY_SIZE.height - text_natural_size.height) / 2.0
    exp_text_world_x = exp_overlay_origin_x + exp_text_local_x
    exp_text_world_y = exp_overlay_origin_y + exp_text_local_y

    # --- assertions ---
    overlay_actual_origin = get_world_origin(overlay_child)
    assert_allclose(
        overlay_actual_origin,
        (exp_overlay_origin_x, exp_overlay_origin_y),
        err_msg="Overlay container position is incorrect",
        atol=1e-6,
    )
    assert overlay_child._dimensions == OVERLAY_SIZE, "Overlay container dimensions mismatch"

    inner_actual_origin = get_world_origin(inner_text)
    assert_allclose(
        inner_actual_origin,
        (exp_text_world_x, exp_text_world_y),
        err_msg="Inner text is not centered within the overlay container",
        atol=1e-6,
    )
    assert inner_text._dimensions.width == pytest.approx(text_natural_size.width)
    assert inner_text._dimensions.height == pytest.approx(text_natural_size.height)

    assert hasattr(
        inner_text, "_layout_origin_in_parent"
    ), "Inner text should have layout origin stored"
    assert_allclose(
        inner_text._layout_origin_in_parent,
        (exp_text_local_x, exp_text_local_y),
        err_msg="Inner text stored layout origin is incorrect",
        atol=1e-6,
    )


def test_row_layout_heterogeneous_padding_margin_start():
    """test row layout, start justification, start alignment with non-uniform padding and margin"""
    set_debug(True)  # enable for this specific test if needed

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
            align_items="start",
            justify_content="start",
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
        min_dimensions=Size(width=BOXW, height=BOXH + 10),
        style=BoxStyle(background_color="green", margin=MAR[1]),
    )
    box3 = Container(
        id="b3",
        min_dimensions=Size(width=BOXW, height=BOXH - 10),
        style=BoxStyle(background_color="blue", margin=MAR[2]),
    )

    parent.add_children([box1, box2, box3])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
    renderer.render_component(ax, parent)  # keep commented unless debugging interactively

    # calculations
    content_x = PAD[3]
    content_y = PAD[0]

    # box1 pos
    exp_b1_x = content_x + MAR[0][3]
    exp_b1_y = content_y + MAR[0][0]
    assert_allclose(get_world_origin(box1), (exp_b1_x, exp_b1_y), err_msg="Box1 pos")

    # box2 pos
    exp_b2_x = exp_b1_x + BOXW + MAR[0][1] + GAP + MAR[1][3]
    exp_b2_y = content_y + MAR[1][0]
    assert_allclose(get_world_origin(box2), (exp_b2_x, exp_b2_y), err_msg="Box2 pos")

    # box3 pos
    exp_b3_x = exp_b2_x + BOXW + MAR[1][1] + GAP + MAR[2][3]
    exp_b3_y = content_y + MAR[2][0]
    assert_allclose(get_world_origin(box3), (exp_b3_x, exp_b3_y), err_msg="Box3 pos")


def test_column_layout_heterogeneous_padding_margin_center():
    """test column layout, center justification, center alignment with non-uniform padding and margin"""
    set_debug(True)  # enable for this specific test if needed

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
            align_items="center",
            justify_content="center",
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
        min_dimensions=Size(width=BOXW + 10, height=BOXH),
        style=BoxStyle(background_color="green", margin=MAR[1]),
    )
    box3 = Container(
        id="b3",
        min_dimensions=Size(width=BOXW - 10, height=BOXH),
        style=BoxStyle(background_color="blue", margin=MAR[2]),
    )

    parent.add_children([box1, box2, box3])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, parent)  # keep commented unless debugging interactively

    # calculations
    content_x = PAD[3]
    content_y = PAD[0]
    content_w = PWIDTH - PAD[3] - PAD[1]
    content_h = PHEIGHT - PAD[0] - PAD[2]
    total_child_h_mar = (
        (BOXH + MAR[0][0] + MAR[0][2])
        + (BOXH + MAR[1][0] + MAR[1][2])
        + (BOXH + MAR[2][0] + MAR[2][2])
    )
    total_gap_h = (len(parent.children) - 1) * GAP
    required_h = total_child_h_mar + total_gap_h
    extra_h = content_h - required_h
    start_y_offset = extra_h / 2.0

    # box1 pos
    exp_b1_x = content_x + MAR[0][3] + (content_w - MAR[0][3] - MAR[0][1] - BOXW) / 2.0
    exp_b1_y = content_y + start_y_offset + MAR[0][0]
    assert_allclose(get_world_origin(box1), (exp_b1_x, exp_b1_y), err_msg="Box1 pos")

    # box2 pos
    exp_b2_x = content_x + MAR[1][3] + (content_w - MAR[1][3] - MAR[1][1] - (BOXW + 10)) / 2.0
    exp_b2_y = exp_b1_y + BOXH + MAR[0][2] + GAP + MAR[1][0]
    assert_allclose(get_world_origin(box2), (exp_b2_x, exp_b2_y), err_msg="Box2 pos")

    # box3 pos
    exp_b3_x = content_x + MAR[2][3] + (content_w - MAR[2][3] - MAR[2][1] - (BOXW - 10)) / 2.0
    exp_b3_y = exp_b2_y + BOXH + MAR[1][2] + GAP + MAR[2][0]
    assert_allclose(get_world_origin(box3), (exp_b3_x, exp_b3_y), err_msg="Box3 pos")


def test_row_layout_space_between_stretch_margins():
    """test row layout, space-between, stretch alignment with non-uniform padding and margin"""
    set_debug(True)  # enable for this specific test if needed

    PAD = (5, 10, 15, 20)  # t, r, b, l
    MAR = [(2, 8, 4, 6), (10, 3, 5, 7), (6, 9, 11, 2)]  # t, r, b, l for each child
    GAP = 12.0
    BOXW = 50.0
    PWIDTH = 400.0
    PHEIGHT = 100.0

    parent = Container(
        id="hetero-stretch-between-row",
        min_dimensions=Size(width=PWIDTH, height=PHEIGHT),
        style=BoxStyle(padding=PAD, background_color="#eee"),
        layout=LayoutConstraints(
            direction="row",
            align_items="stretch",
            justify_content="space-between",
            gap=GAP,
        ),
    )
    box1 = Container(
        id="b1",
        min_dimensions=Size(width=BOXW, height=20),
        style=BoxStyle(background_color="red", margin=MAR[0]),
    )
    box2 = Container(
        id="b2",
        min_dimensions=Size(width=BOXW, height=40),
        style=BoxStyle(background_color="green", margin=MAR[1]),
    )
    box3 = Container(
        id="b3",
        min_dimensions=Size(width=BOXW, height=10),
        style=BoxStyle(background_color="blue", margin=MAR[2]),
    )

    parent.add_children([box1, box2, box3])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, parent)  # keep commented unless debugging interactively

    # calculations
    content_x = PAD[3]
    content_y = PAD[0]
    content_w = PWIDTH - PAD[3] - PAD[1]
    content_h = PHEIGHT - PAD[0] - PAD[2]
    exp_h1 = max(0, content_h - MAR[0][0] - MAR[0][2])
    exp_h2 = max(0, content_h - MAR[1][0] - MAR[1][2])
    exp_h3 = max(0, content_h - MAR[2][0] - MAR[2][2])

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

    total_child_w_mar = (
        (BOXW + MAR[0][3] + MAR[0][1])
        + (BOXW + MAR[1][3] + MAR[1][1])
        + (BOXW + MAR[2][3] + MAR[2][1])
    )
    total_gap_w = (len(parent.children) - 1) * GAP
    required_w = total_child_w_mar + total_gap_w
    extra_w = content_w - required_w
    spacing = extra_w / (len(parent.children) - 1) if len(parent.children) > 1 else 0

    exp_b1_x = content_x + MAR[0][3]
    exp_b1_y = content_y + MAR[0][0]
    assert_allclose(get_world_origin(box1), (exp_b1_x, exp_b1_y), err_msg="Box1 pos")

    exp_b2_x = exp_b1_x + BOXW + MAR[0][1] + GAP + MAR[1][3] + spacing
    exp_b2_y = content_y + MAR[1][0]
    assert_allclose(get_world_origin(box2), (exp_b2_x, exp_b2_y), err_msg="Box2 pos")

    exp_b3_x = exp_b2_x + BOXW + MAR[1][1] + GAP + MAR[2][3] + spacing
    exp_b3_y = content_y + MAR[2][0]
    assert_allclose(get_world_origin(box3), (exp_b3_x, exp_b3_y), err_msg="Box3 pos")


def test_nested_heterogeneous_padding_margin():
    """test nested containers with different paddings and margins interacting"""
    set_debug(True)  # enable for this specific test if needed

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
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, outer_parent)  # keep commented unless debugging interactively

    # calculations
    op_content_x = OPAD[3]
    op_content_y = OPAD[0]
    op_content_w = OWIDTH - OPAD[3] - OPAD[1]
    op_content_h = OHEIGHT - OPAD[0] - OPAD[2]
    avail_op_w = op_content_w - IMAR[3] - IMAR[1]
    avail_op_h = op_content_h - IMAR[0] - IMAR[2]
    exp_ic_x = op_content_x + IMAR[3] + (avail_op_w - IWIDTH) / 2.0
    exp_ic_y = op_content_y + IMAR[0] + (avail_op_h - IHEIGHT) / 2.0
    assert_allclose(get_world_origin(inner_child), (exp_ic_x, exp_ic_y), err_msg="Inner Child pos")

    ic_content_x = IPAD[3]
    ic_content_y = IPAD[0]
    ic_content_w = IWIDTH - IPAD[3] - IPAD[1]
    ic_content_h = IHEIGHT - IPAD[0] - IPAD[2]
    exp_gc_local_x = ic_content_x + GMAR[3]
    exp_gc_local_y = ic_content_y + ic_content_h - GMAR[2] - BOXH
    exp_gc_world_x = exp_ic_x + exp_gc_local_x
    exp_gc_world_y = exp_ic_y + exp_gc_local_y
    assert_allclose(
        get_world_origin(grandchild), (exp_gc_world_x, exp_gc_world_y), err_msg="Grandchild pos"
    )


def test_row_layout_margins_with_offsets():
    """test combining layout positioning (with margins) and explicit child offsets"""
    set_debug(True)  # enable for this specific test if needed

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
            align_items="end",
            justify_content="center",
            gap=GAP,
        ),
    )
    box1 = Container(
        id="b1",
        min_dimensions=Size(width=BOXW, height=BOXH),
        style=BoxStyle(background_color="red", margin=MAR),
    )
    box2 = Container(
        id="b2",
        min_dimensions=Size(width=BOXW, height=BOXH),
        style=BoxStyle(background_color="green", margin=MAR),
        offset=OFF,
    )

    parent.add_children([box1, box2])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, parent)  # keep commented unless debugging interactively

    # calculations
    content_x = PAD[3]
    content_y = PAD[0]
    content_w = PWIDTH - PAD[3] - PAD[1]
    content_h = PHEIGHT - PAD[0] - PAD[2]
    total_child_w_mar = 2 * (BOXW + MAR[3] + MAR[1])
    total_gap_w = GAP
    required_w = total_child_w_mar + total_gap_w
    extra_w = content_w - required_w
    start_x_offset = extra_w / 2.0

    # box1 (layout only)
    layout_b1_x = content_x + start_x_offset + MAR[3]
    layout_b1_y = content_y + content_h - MAR[2] - BOXH
    exp_b1_x = layout_b1_x
    exp_b1_y = layout_b1_y
    assert_allclose(get_world_origin(box1), (exp_b1_x, exp_b1_y), err_msg="Box1 pos (layout)")

    # box2 (layout + offset)
    layout_b2_x = exp_b1_x + BOXW + MAR[1] + GAP + MAR[3]
    layout_b2_y = layout_b1_y
    delta_b2_x, delta_b2_y = OFF.compute(box2._dimensions, parent._dimensions)
    assert_allclose((delta_b2_x, delta_b2_y), (11, -11), err_msg="Box2 offset delta")
    exp_b2_x = layout_b2_x + delta_b2_x
    exp_b2_y = layout_b2_y + delta_b2_y
    assert_allclose(
        get_world_origin(box2), (exp_b2_x, exp_b2_y), err_msg="Box2 pos (layout+offset)"
    )


def test_container_natural_size_with_constrained_children():
    """
    tests that a container's natural size calculation correctly uses the
    *constrained* size of children when those children have no intrinsic size
    but are sized by constraints like min_dimensions (often set by styles).
    also tests justify_content='space-around' when available space exactly
    matches required space (should behave like 'start').
    """

    CHILD_W, CHILD_H = 15.0, 4.0
    PARENT_PAD_T, PARENT_PAD_R, PARENT_PAD_B, PARENT_PAD_L = 12.0, 4.0, 1.0, 4.0
    PARENT_GAP = 5.0
    PARENT_MIN_W, PARENT_MIN_H = 18.0, 18.0

    jstyle.update(
        {
            "ConstrainedChild": {
                "min_dimensions": Size(width=CHILD_W, height=CHILD_H),
                "style.margin": (0, 0, 0, 0),
            },
            "ParentSizedByChildren": {
                "min_dimensions": Size(width=PARENT_MIN_W, height=PARENT_MIN_H),
                "style.padding": (PARENT_PAD_T, PARENT_PAD_R, PARENT_PAD_B, PARENT_PAD_L),
                "layout": LayoutConstraints(
                    direction="column",
                    justify_content="space-around",
                    align_items="start",
                    gap=PARENT_GAP,
                ),
            },
        }
    )

    class ConstrainedChild(Component):
        style: BoxStyle = Field(default_factory=BoxStyle)

    class ParentSizedByChildren(Container):
        style: BoxStyle = Field(default_factory=BoxStyle)

    child1 = ConstrainedChild(id="c1")
    child2 = ConstrainedChild(id="c2")
    child3 = ConstrainedChild(id="c3")

    parent = ParentSizedByChildren(id="parent", children=[child1, child2, child3])

    renderer = MatplotlibRenderer()
    parent.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, parent)  # keep commented unless debugging interactively

    # calculations
    num_children = 3
    total_child_height = num_children * CHILD_H
    total_gap_height = (num_children - 1) * PARENT_GAP
    content_natural_h = total_child_height + total_gap_height
    expected_natural_height = content_natural_h + PARENT_PAD_T + PARENT_PAD_B
    content_natural_w = CHILD_W
    expected_natural_width = content_natural_w + PARENT_PAD_L + PARENT_PAD_R
    final_parent_w = max(PARENT_MIN_W, expected_natural_width)
    final_parent_h = max(PARENT_MIN_H, expected_natural_height)
    final_content_h = final_parent_h - PARENT_PAD_T - PARENT_PAD_B
    final_content_x = PARENT_PAD_L
    final_content_y = PARENT_PAD_T
    exp_c1_y = final_content_y + 0
    exp_c2_y = exp_c1_y + CHILD_H + 0 + PARENT_GAP + 0
    exp_c3_y = exp_c2_y + CHILD_H + 0 + PARENT_GAP + 0
    exp_c1_x = final_content_x + 0
    exp_c2_x = final_content_x + 0
    exp_c3_x = final_content_x + 0

    # assertions
    assert child1._dimensions == Size(width=CHILD_W, height=CHILD_H), "Child1 dimensions incorrect"
    assert child2._dimensions == Size(width=CHILD_W, height=CHILD_H), "Child2 dimensions incorrect"
    assert child3._dimensions == Size(width=CHILD_W, height=CHILD_H), "Child3 dimensions incorrect"
    assert parent._natural_dimensions.width == pytest.approx(expected_natural_width)
    assert parent._natural_dimensions.height == pytest.approx(expected_natural_height)
    assert parent._dimensions.width == pytest.approx(final_parent_w)
    assert parent._dimensions.height == pytest.approx(final_parent_h)
    assert_allclose(
        child1._layout_origin_in_parent, (exp_c1_x, exp_c1_y), err_msg="Child1 position", rtol=1e-6
    )
    assert_allclose(
        child2._layout_origin_in_parent, (exp_c2_x, exp_c2_y), err_msg="Child2 position", rtol=1e-6
    )
    assert_allclose(
        child3._layout_origin_in_parent, (exp_c3_x, exp_c3_y), err_msg="Child3 position", rtol=1e-6
    )


# Helper for rotation matrix
def rotation_matrix(angle_deg):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Helper for translation matrix
def translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


# Helper for scaling matrix
def scale_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])


def test_transform_on_root():
    """test applying a transform to the root container"""
    set_debug(True)

    ROOT_SIZE = 100.0
    ROTATE_DEG = 45.0
    SCALE_XY = (0.5, 1.5)
    TRANSLATE_XY = (10, 20)
    CHILD_OFFSET = (10, 10)
    CHILD_SIZE = (10, 10)

    root = Container(
        id="transformed-root",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(background_color="lightblue"),
        transform=Transform(rotate=ROTATE_DEG, scale=SCALE_XY, translate=TRANSLATE_XY),
    )
    child = Component(
        id="child",
        min_dimensions=Size(width=CHILD_SIZE[0], height=CHILD_SIZE[1]),
        offset=Offset(absolute=CHILD_OFFSET),
    )
    root.add_child(child)

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, root)

    # manually calculate expected root origin
    # transform order: scale -> skew (none) -> rotate (centered) -> translate
    # rotation center = (50, 50)
    mat_s = scale_matrix(SCALE_XY[0], SCALE_XY[1])
    mat_r = rotation_matrix(ROTATE_DEG)
    mat_t_center = translation_matrix(ROOT_SIZE / 2, ROOT_SIZE / 2)
    mat_t_uncenter = translation_matrix(-ROOT_SIZE / 2, -ROOT_SIZE / 2)
    mat_t = translation_matrix(TRANSLATE_XY[0], TRANSLATE_XY[1])
    # combined transform matrix applied to the root's coordinate system
    root_transform_matrix = mat_t @ mat_t_center @ mat_r @ mat_t_uncenter @ mat_s
    # world origin is the transform applied to local (0,0)
    exp_root_origin = (root_transform_matrix @ np.array([0, 0, 1]))[:2]
    # Calculation:
    # T(10,20)@T(50,50)@R(45)@T(-50,-50)@S(0.5,1.5) @ [0,0,1]
    # = T(10,20)@T(50,50)@R(45)@T(-50,-50) @ [0,0,1] (scale doesn't affect origin point)
    # = T(10,20)@T(50,50)@R(45) @ [-50,-50,1]
    # = T(10,20)@T(50,50) @ [-50c+50s, -50s-50c, 1] (c=s=0.707)
    # = T(10,20)@T(50,50) @ [0, -70.71, 1]
    # = T(10,20) @ [50, -20.71, 1]
    # = [60, -0.71, 1]
    # Expected: (60, -0.7106...)

    pos_root = get_world_origin(root)
    assert_allclose(pos_root, exp_root_origin, err_msg="Root origin with transform", atol=1e-6)

    # manually calculate expected child origin
    # child is positioned at (0,0) in root's layout, then offset by (10,10)
    child_pos_before_offset = np.array([0, 0, 1])
    child_pos_after_offset = (
        translation_matrix(CHILD_OFFSET[0], CHILD_OFFSET[1]) @ child_pos_before_offset
    )
    # apply root's world transform to the child's offset position
    exp_child_origin = (root_transform_matrix @ child_pos_after_offset)[:2]
    # Calculation: W_root @ [10, 10, 1]
    # c = 0.7071
    # W_root = [[0.5c, -1.5c, 60], [0.5c, 1.5c, 70-100c], [0, 0, 1]] (from previous derivation)
    # [0.5c, -1.5c, 60] @ [10,10,1] = 5c - 15c + 60 = 60 - 10c = 60 - 7.07 = 52.93
    # [0.5c, 1.5c, 70-100c] @ [10,10,1] = 5c + 15c + 70 - 100c = 70 - 80c = 70 - 56.57 = 13.43
    # Expected: (52.93, 13.43)

    pos_child = get_world_origin(child)
    assert_allclose(
        pos_child, exp_child_origin, err_msg="Child origin under transformed root", atol=1e-6
    )


def test_transform_on_child():
    """test applying a transform to a direct child in a layout"""
    set_debug(True)

    PAD = 10.0
    ROOT_SIZE = 200.0
    BOX_SIZE = 50.0
    GAP = 20.0
    ROTATE_DEG = -30.0
    SCALE_XY = (1.2, 0.8)
    TRANSLATE_XY = (5, -5)

    box_a = Container(
        id="box_a",
        min_dimensions=Size(width=BOX_SIZE, height=BOX_SIZE),
        style=BoxStyle(background_color="red"),
    )
    box_b = Container(
        id="box_b_transformed",
        min_dimensions=Size(width=BOX_SIZE, height=BOX_SIZE),
        style=BoxStyle(background_color="blue"),
        transform=Transform(rotate=ROTATE_DEG, scale=SCALE_XY, translate=TRANSLATE_XY),
    )

    root = Container(
        id="child-transform-test",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        layout=LayoutConstraints(
            direction="row", align_items="center", justify_content="start", gap=GAP
        ),
        children=[box_a, box_b],
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, root)

    # assertions
    content_h = ROOT_SIZE - 2 * PAD
    exp_a_x = PAD
    exp_a_y = PAD + (content_h - BOX_SIZE) / 2  # = 10 + (180 - 50) / 2 = 75
    pos_a = get_world_origin(box_a)
    assert_allclose(pos_a, (exp_a_x, exp_a_y), err_msg="Box A position")

    # calculate box b's layout position *relative to root origin*
    layout_b_x = exp_a_x + BOX_SIZE + GAP  # = 10 + 50 + 20 = 80
    layout_b_y = exp_a_y  # = 75
    mat_layout_b = translation_matrix(layout_b_x, layout_b_y)

    # calculate box b's local transform matrix
    rot_center = (BOX_SIZE / 2, BOX_SIZE / 2)
    mat_s = scale_matrix(SCALE_XY[0], SCALE_XY[1])
    mat_r = rotation_matrix(ROTATE_DEG)
    mat_t_center = translation_matrix(rot_center[0], rot_center[1])
    mat_t_uncenter = translation_matrix(-rot_center[0], -rot_center[1])
    mat_t = translation_matrix(TRANSLATE_XY[0], TRANSLATE_XY[1])
    # transform applied relative to layout position
    box_b_transform_matrix = mat_t @ mat_t_center @ mat_r @ mat_t_uncenter @ mat_s

    # expected world matrix = layout_translation * local_transform
    exp_b_world_matrix = mat_layout_b @ box_b_transform_matrix
    exp_b_origin = (exp_b_world_matrix @ np.array([0, 0, 1]))[:2]
    # Calculation (from previous derivation): (77.5, 80.85)

    pos_b = get_world_origin(box_b)
    assert_allclose(pos_b, exp_b_origin, err_msg="Box B origin with transform", atol=1e-6)


def test_transform_on_nested_child():
    """test applying a transform to a grandchild"""
    set_debug(True)

    PAD = 5.0
    ROOT_SIZE = 150.0
    CHILD_PADDING = 10.0
    PARENT_SIZE = 80.0  # child container size
    CHILD_SIZE = 30.0  # grandchild size
    ROTATE_DEG = 90.0
    SCALE_XY = (2.0, 1.0)
    TRANSLATE_XY = (3, 3)
    CHILD_OFFSET = (10, 10)  # child container offset relative to root
    GC_OFFSET = (5, 5)  # grandchild offset relative to child container

    grandchild = Container(
        id="gc_transformed",
        min_dimensions=Size(width=CHILD_SIZE, height=CHILD_SIZE),
        style=BoxStyle(background_color="cyan"),
        transform=Transform(rotate=ROTATE_DEG, scale=SCALE_XY, translate=TRANSLATE_XY),
        offset=Offset(absolute=GC_OFFSET),
    )
    child = Container(
        id="child",
        min_dimensions=Size(width=PARENT_SIZE, height=PARENT_SIZE),
        style=BoxStyle(
            background_color="lightgreen",
            padding=(CHILD_PADDING, CHILD_PADDING, CHILD_PADDING, CHILD_PADDING),
        ),
        layout=LayoutConstraints(align_items="start", justify_content="start"),
        children=[grandchild],
        offset=Offset(absolute=CHILD_OFFSET),
    )
    root = Container(
        id="nested-transform-test",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        children=[child],
        layout=LayoutConstraints(align_items="start", justify_content="start"),
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, root)

    # assertions
    # calculate child world matrix
    child_layout_x = PAD
    child_layout_y = PAD
    child_offset_x = CHILD_OFFSET[0]
    child_offset_y = CHILD_OFFSET[1]
    mat_child_pos = translation_matrix(
        child_layout_x + child_offset_x, child_layout_y + child_offset_y
    )  # (15, 15)

    # calculate grandchild local matrix relative to child
    gc_layout_x = CHILD_PADDING  # inside child's padding
    gc_layout_y = CHILD_PADDING
    gc_offset_x = GC_OFFSET[0]
    gc_offset_y = GC_OFFSET[1]
    mat_gc_layout_offset = translation_matrix(
        gc_layout_x + gc_offset_x, gc_layout_y + gc_offset_y
    )  # (15, 15)

    # grandchild transform matrix
    rot_center = (CHILD_SIZE / 2, CHILD_SIZE / 2)
    mat_s = scale_matrix(SCALE_XY[0], SCALE_XY[1])
    mat_r = rotation_matrix(ROTATE_DEG)
    mat_t_center = translation_matrix(rot_center[0], rot_center[1])
    mat_t_uncenter = translation_matrix(-rot_center[0], -rot_center[1])
    mat_t = translation_matrix(TRANSLATE_XY[0], TRANSLATE_XY[1])
    gc_transform_matrix = mat_t @ mat_t_center @ mat_r @ mat_t_uncenter @ mat_s

    # grandchild world matrix = child_world_pos * gc_layout_and_offset_in_child * gc_local_transform
    exp_gc_world_matrix = mat_child_pos @ mat_gc_layout_offset @ gc_transform_matrix
    exp_gc_origin = (exp_gc_world_matrix @ np.array([0, 0, 1]))[:2]
    # Calculation (from previous derivation): (63, 33)

    pos_gc = get_world_origin(grandchild)
    assert_allclose(pos_gc, exp_gc_origin, err_msg="Grandchild origin with transform", atol=1e-6)


def test_combined_offset_and_transform():
    """test applying both an offset and a transform to the same component"""
    set_debug(True)

    PAD = 10.0
    ROOT_SIZE = 200.0
    BOX_SIZE = 50.0
    ROTATE_DEG = 15.0
    TRANSLATE_XY = (2, 3)
    OFFSET_ABS = (20, 10)
    OFFSET_REL = (0.1, 0.2)

    box = Container(
        id="offset_and_transform",
        min_dimensions=Size(width=BOX_SIZE, height=BOX_SIZE),
        style=BoxStyle(background_color="orange"),
        offset=Offset(absolute=OFFSET_ABS, relative=OFFSET_REL),
        transform=Transform(rotate=ROTATE_DEG, translate=TRANSLATE_XY),
    )
    root = Container(
        id="combo-test",
        min_dimensions=Size(width=ROOT_SIZE, height=ROOT_SIZE),
        style=BoxStyle(padding=(PAD, PAD, PAD, PAD)),
        layout=LayoutConstraints(align_items="start", justify_content="start"),
        children=[box],
    )

    renderer = MatplotlibRenderer()
    root.measure_and_layout(renderer)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    renderer.render_component(ax, root)

    # 1. box layout origin relative to root origin
    layout_x = PAD
    layout_y = PAD
    # 2. box offset delta
    offset_dx = BOX_SIZE * OFFSET_REL[0] + OFFSET_ABS[0]  # 50*0.1 + 20 = 25
    offset_dy = BOX_SIZE * OFFSET_REL[1] + OFFSET_ABS[1]  # 50*0.2 + 10 = 20
    # 3. position matrix incorporating layout and offset
    mat_pos = translation_matrix(layout_x + offset_dx, layout_y + offset_dy)  # T(35, 30)
    # 4. box transform matrix
    rot_center = (BOX_SIZE / 2, BOX_SIZE / 2)
    mat_s = scale_matrix(1, 1)
    mat_r = rotation_matrix(ROTATE_DEG)
    mat_t_center = translation_matrix(rot_center[0], rot_center[1])
    mat_t_uncenter = translation_matrix(-rot_center[0], -rot_center[1])
    mat_t = translation_matrix(TRANSLATE_XY[0], TRANSLATE_XY[1])
    box_transform_matrix = mat_t @ mat_t_center @ mat_r @ mat_t_uncenter @ mat_s

    # 5. combined world matrix = position_matrix * transform_matrix
    exp_box_world_matrix = mat_pos @ box_transform_matrix
    exp_box_origin = (exp_box_world_matrix @ np.array([0, 0, 1]))[:2]
    # Calculation (from previous derivation): (44.325, 27.375)

    pos_box = get_world_origin(box)
    assert_allclose(
        pos_box, exp_box_origin, err_msg="Box origin with offset and transform", atol=1e-6
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
    test_container_natural_size_with_constrained_children()
    plt.show()

    test_transform_on_root()
    plt.show()
    test_transform_on_child()
    plt.show()
    test_transform_on_nested_child()
    plt.show()
    test_combined_offset_and_transform()
    plt.show()
