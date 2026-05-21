from jeanplot import Container, Size, LayoutConstraints


def test_main_axis_weights_distribute_width():
    parent = Container(
        id="p",
        min_dimensions=Size(100.0, 10.0),
        layout=LayoutConstraints(direction="row", main_axis_weights=[2.0, 1.0, 1.0]),
    )
    for i in range(3):
        parent.add_child(Container(id=f"c{i}", min_dimensions=Size(1.0, 10.0)))
    parent.measure_and_layout(None)
    widths = [c._dimensions.width for c in parent.children]
    assert abs(widths[0] - 50.0) < 1e-6
    assert abs(widths[1] - 25.0) < 1e-6
    assert abs(widths[2] - 25.0) < 1e-6


def test_main_axis_weights_distribute_height_in_column():
    parent = Container(
        id="p",
        min_dimensions=Size(10.0, 80.0),
        layout=LayoutConstraints(direction="column", main_axis_weights=[3.0, 1.0]),
    )
    for i in range(2):
        parent.add_child(Container(id=f"c{i}", min_dimensions=Size(10.0, 1.0)))
    parent.measure_and_layout(None)
    heights = [c._dimensions.height for c in parent.children]
    assert abs(heights[0] - 60.0) < 1e-6
    assert abs(heights[1] - 20.0) < 1e-6


def test_main_axis_weights_account_for_gap():
    parent = Container(
        id="p",
        min_dimensions=Size(110.0, 10.0),
        layout=LayoutConstraints(direction="row", gap=10.0, main_axis_weights=[1.0, 1.0]),
    )
    for i in range(2):
        parent.add_child(Container(id=f"c{i}", min_dimensions=Size(1.0, 10.0)))
    parent.measure_and_layout(None)
    widths = [c._dimensions.width for c in parent.children]
    assert abs(widths[0] - 50.0) < 1e-6
    assert abs(widths[1] - 50.0) < 1e-6


def test_no_weights_falls_back_to_default():
    parent = Container(
        id="p",
        min_dimensions=Size(100.0, 10.0),
        layout=LayoutConstraints(direction="row"),
    )
    for i in range(3):
        parent.add_child(Container(id=f"c{i}", min_dimensions=Size(20.0, 10.0)))
    parent.measure_and_layout(None)
    widths = [c._dimensions.width for c in parent.children]
    assert all(abs(w - 20.0) < 1e-6 for w in widths)
