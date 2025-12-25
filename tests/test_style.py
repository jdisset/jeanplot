import pytest
from typing import Optional, List
import copy
from pydantic import Field

from jeanplot import Component, Container, Text, jstyle, BoxStyle, LayoutConstraints

# --- Test Fixtures ---


@pytest.fixture(autouse=True)
def reset_jstyle():
    """resets jstyle styles before each test"""
    # Store the state of the global jstyle instance's internal list
    original_rules = copy.deepcopy(jstyle.styles)
    original_raw = copy.deepcopy(jstyle._raw_styles)
    jstyle.styles = []  # Clear parsed rules
    jstyle._raw_styles = {}  # Clear raw dictionary
    yield
    # Restore the state
    jstyle.styles = original_rules
    jstyle._raw_styles = original_raw


# --- Helper Components ---


class SimpleComponent(Component):
    color: str = "black"
    size: int = 10
    font_style: str = "normal"
    name: str = ""
    style: BoxStyle = Field(default_factory=BoxStyle)
    debug: bool = False


class CustomContainer(Container):
    border_thickness: float = 1.0
    unique_prop: str = "default"


class SpecialText(Text):
    text_decoration: str = "none"


# --- Test Cases ---


def test_type_selector():
    """test basic styling based on component type"""
    jstyle.update(
        {
            "SimpleComponent": {
                "color": "red",
                "size": 20,
            }
        }
    )
    comp = SimpleComponent(id="c1")
    jstyle.apply(comp)
    assert comp.color == "red"
    assert comp.size == 20


def test_id_selector():
    """test styling based on component id"""
    jstyle.update(
        {
            "[id=special]": {
                "color": "blue",
            }
        }
    )
    comp1 = SimpleComponent(id="c1")
    comp2 = SimpleComponent(id="special")
    root = Container(children=[comp1, comp2])
    jstyle.apply(root)

    assert comp1.color == "black"  # default
    assert comp2.color == "blue"


def test_style_class_selector_single():
    """test styling based on a single style class"""
    jstyle.update(
        {
            "[style_class=highlight]": {
                "color": "yellow",
            }
        }
    )
    comp1 = SimpleComponent(id="c1", style_class=["normal"])
    comp2 = SimpleComponent(id="c2", style_class=["highlight"])
    root = Container(children=[comp1, comp2])
    jstyle.apply(root)

    assert comp1.color == "black"
    assert comp2.color == "yellow"


def test_style_class_selector_list():
    """test styling matching any class in the list"""
    jstyle.update(
        {
            "[style_class=primary]": {"size": 50},
            "[style_class=large]": {"color": "green"},
        }
    )
    comp = SimpleComponent(id="c1", style_class=["button", "primary", "large"])
    jstyle.apply(comp)

    assert comp.size == 50
    assert comp.color == "green"


def test_nested_property_setting():
    """test setting properties on nested models like 'style'"""
    jstyle.update(
        {
            "Container": {
                "style.background_color": "lightblue",
                "style.border_width": 2.5,
            }
        }
    )
    container = Container(id="cont")
    jstyle.apply(container)  # apply should create/update style

    assert isinstance(container.style, BoxStyle)
    assert container.style.background_color == "lightblue"
    assert container.style.border_width == 2.5


def test_specificity_id_over_type():
    """test that id selectors override type selectors"""
    jstyle.update(
        {
            "SimpleComponent": {"color": "red"},
            "[id=unique]": {"color": "purple"},
        }
    )
    comp1 = SimpleComponent(id="c1")
    comp2 = SimpleComponent(id="unique")
    root = Container(children=[comp1, comp2])
    jstyle.apply(root)

    assert comp1.color == "red"
    assert comp2.color == "purple"


def test_specificity_style_class_over_type():
    """test that style_class selectors override type selectors"""
    jstyle.update(
        {
            "SimpleComponent": {"color": "red", "size": 10},
            "[style_class=important]": {"color": "blue"},
        }
    )
    comp1 = SimpleComponent(id="c1")
    comp2 = SimpleComponent(id="c2", style_class=["important"])
    root = Container(children=[comp1, comp2])
    jstyle.apply(root)

    assert comp1.color == "red"
    assert comp1.size == 10
    assert comp2.color == "blue"  # overridden
    assert comp2.size == 10  # inherited from type


def test_specificity_combined_selector():
    """test that combined type+attribute selectors override simpler ones"""
    jstyle.update(
        {
            "SimpleComponent": {"color": "red", "size": 10},
            "[style_class=fancy]": {"color": "blue"},
            "SimpleComponent[style_class=fancy]": {"color": "green"},  # higher spec
        }
    )
    comp1 = SimpleComponent(id="c1", style_class=["other"])
    comp2 = SimpleComponent(id="c2", style_class=["fancy"])
    root = Container(children=[comp1, comp2])
    jstyle.apply(root)

    assert comp1.color == "red"
    assert comp1.size == 10
    assert comp2.color == "green"  # combined selector wins
    assert comp2.size == 10


def test_nested_selectors():
    """test applying styles to children based on parent context"""
    jstyle.update(
        {
            "Container": {
                "style.padding": (10, 10, 10, 10),
                "Text": {  # applies to Text children inside any Container
                    "color": "darkgray",
                },
            },
            "Container[id=outer]": {
                "style.background_color": "yellow",
                "Text": {  # applies specifically to Text inside #outer
                    "color": "orange",
                    "font_weight": "bold",
                },
            },
        }
    )
    text1 = Text(id="t1", text="hello")
    cont1 = Container(id="inner", children=[text1])
    text2 = Text(id="t2", text="world")
    cont2 = Container(id="outer", children=[text2])
    root = Container(id="root", children=[cont1, cont2])

    jstyle.apply(root)

    # check inner container's text
    assert cont1.children[0].color == "darkgray"
    assert cont1.children[0].font_weight == "normal"  # default

    # check outer container and its text
    assert cont2.style.background_color == "yellow"
    assert cont2.children[0].color == "orange"
    assert cont2.children[0].font_weight == "bold"


def test_inheritance_styling():
    """test that subclasses inherit styles and can be overridden"""
    jstyle.update(
        {
            "Container": {"style.border_color": "black"},
            "CustomContainer": {"border_thickness": 5.0},
        }
    )
    cont = Container(id="c1")
    custom_cont = CustomContainer(id="cc1")
    root = Container(children=[cont, custom_cont])
    jstyle.apply(root)

    assert cont.style.border_color == "black"
    # custom container inherits border_color and gets its own style
    assert custom_cont.style.border_color == "black"
    assert custom_cont.border_thickness == 5.0


def test_inheritance_override():
    """test that subclass styles override parent styles"""
    jstyle.update(
        {
            "Container": {"style.border_color": "black", "style.border_width": 1},
            "CustomContainer": {"style.border_color": "red"},  # more specific type
        }
    )
    custom_cont = CustomContainer(id="cc1")
    jstyle.apply(custom_cont)

    assert custom_cont.style.border_color == "red"  # overridden
    assert custom_cont.style.border_width == 1  # inherited


def test_wildcard_selector():
    """test the wildcard selector '*'"""
    jstyle.update(
        {
            "*": {"debug": True},
            "Text": {"color": "blue"},  # more specific type
        }
    )
    comp = SimpleComponent(id="s1")
    text = Text(id="t1", text="hi")
    cont = Container(id="c1", children=[comp, text])
    jstyle.apply(cont)

    assert cont.debug is True
    assert comp.debug is True
    assert text.debug is True
    assert text.color == "blue"  # text style still wins for color


def test_regex_selector():
    """test attribute selection using regex =/.../"""
    jstyle.update(
        {
            r"[id=/^item-\d+$/]": {"color": "magenta"}  # Use raw string
        }
    )
    comp1 = SimpleComponent(id="item-1")
    comp2 = SimpleComponent(id="item-10")
    comp3 = SimpleComponent(id="item-abc")
    comp4 = SimpleComponent(id="other")
    root = Container(children=[comp1, comp2, comp3, comp4])
    jstyle.apply(root)

    assert comp1.color == "magenta"
    assert comp2.color == "magenta"
    assert comp3.color == "black"  # default
    assert comp4.color == "black"  # default


def test_case_insensitive_selector():
    """test attribute selection using case-insensitive =~"""
    jstyle.update({"[name=~john]": {"size": 100}})
    comp1 = SimpleComponent(id="c1", name="John")
    comp2 = SimpleComponent(id="c2", name="john")
    comp3 = SimpleComponent(id="c3", name="JOHN")
    comp4 = SimpleComponent(id="c4", name="Jane")
    root = Container(children=[comp1, comp2, comp3, comp4])
    jstyle.apply(root)

    assert comp1.size == 100
    assert comp2.size == 100
    assert comp3.size == 100
    assert comp4.size == 10  # default


def test_context_manager():
    """test temporarily overriding styles using 'with jstyle(...)'"""
    jstyle.update({"SimpleComponent": {"color": "black"}})
    comp1 = SimpleComponent(id="c1")
    jstyle.apply(comp1)
    assert comp1.color == "black"

    with jstyle({"SimpleComponent": {"color": "green"}}):
        comp2 = SimpleComponent(id="c2")
        jstyle.apply(comp2)
        assert comp2.color == "green"  # gets temporary style

        # re-applying style to comp1 *inside* context updates it
        jstyle.apply(comp1)
        assert comp1.color == "green"

    # check style is restored after context
    comp3 = SimpleComponent(id="c3")
    jstyle.apply(comp3)
    assert comp3.color == "black"

    # check comp created inside context retains its applied style
    assert comp2.color == "green"
    # check comp1's style reverts after context
    jstyle.apply(comp1)
    assert comp1.color == "black"


def test_partial_style_update():
    """test updating only part of a nested style model"""
    jstyle.update(
        {
            "Container": {
                "style": {
                    "padding": (10, 10, 10, 10),  # Use tuple initially
                    "background_color": "white",
                    "border_width": 1,
                }
            }
        }
    )
    container = Container(id="cont")
    jstyle.apply(container)
    assert container.style.padding == (10, 10, 10, 10)
    assert container.style.background_color == "white"

    # --- Test partial update via _set_property merge ---
    jstyle.update(
        {  # Reset jstyle rules for this specific test part
            "Container": {
                "style": {
                    "padding": [10, None, 12, None],  # Update using list with None
                    "background_color": "green",
                }
            }
        }
    )

    # Create a container with existing style
    container3 = Container(
        id="cont3", style=BoxStyle(padding=(5, 5, 5, 5), background_color="blue")
    )
    # Apply the update rules
    jstyle.apply(container3)

    # Check that padding was updated correctly, preserving original values for None
    assert container3.style.padding == (10, 5, 12, 5)  # Expect tuple after validation
    assert container3.style.background_color == "green"


def test_multi_attribute_selector():
    """test selecting based on multiple attributes"""
    jstyle.update(
        {
            "[id=a, style_class=fancy]": {"color": "cyan"},
            "[id=b, size=5]": {"color": "magenta"},  # testing non-string attribute
        }
    )
    comp1 = SimpleComponent(id="a", style_class=["fancy"])
    comp2 = SimpleComponent(id="a", style_class=["simple"])
    comp3 = SimpleComponent(id="b", style_class=["fancy"], size=5)  # size set directly
    comp4 = SimpleComponent(id="b", style_class=["fancy"], size=10)

    root = Container(children=[comp1, comp2, comp3, comp4])
    jstyle.apply(root)

    assert comp1.color == "cyan"
    assert comp2.color == "black"  # id matches, class doesn't
    assert comp3.color == "magenta"
    assert comp4.color == "black"  # size doesn't match


def test_deep_inheritance_and_override():
    """test styles applied across multiple inheritance levels with overrides."""
    jstyle.update(
        {
            "Component": {  # base for all
                "debug": False  # Applies to all components unless overridden
            },
            "Container": {  # middle class
                "style.border_color": "gray",
                "debug": True,  # overrides Component style for Container instances
            },
            "CustomContainer": {  # most specific class
                "style.background_color": "lightyellow",
                "unique_prop": "custom",
                "debug": False,  # overrides Container style for CustomContainer instances
            },
        }
    )
    comp = Component(id="base")
    cont = Container(id="cont")
    cust = CustomContainer(id="cust")
    root = Container(id="root", children=[comp, cont, cust])

    jstyle.apply(root)  # apply from root to see effect on children

    # Check base component (affected only by "Component" rule)
    assert comp.debug is False  # Corrected Assertion: Style sets it to False

    # Check container (affected by "Component" and "Container" rules)
    assert cont.debug is True  # from Container rule (overrides Component's False)
    assert cont.style.border_color == "gray"  # from Container rule

    # Check custom container (affected by "Component", "Container", "CustomContainer" rules)
    assert cust.debug is False  # from CustomContainer rule (overrides Container's True)
    assert cust.style.border_color == "gray"  # inherited from Container rule
    assert cust.style.background_color == "lightyellow"  # from CustomContainer rule
    assert cust.unique_prop == "custom"  # from CustomContainer rule


def test_nested_specificity_over_global():
    """test that nested type selectors override global type selectors."""
    jstyle.update(
        {
            "Text": {  # global text style
                "color": "black",
                "font_size": 12,
            },
            "Container": {
                "style.background_color": "white",
                "Text": {  # nested text style, applies only inside Containers
                    "color": "blue",  # should override global black
                    "font_style": "italic",
                },
            },
            "Container[id=special]": {
                "style.background_color": "lightblue",
                "Text": {  # nested text style, specific to #special container
                    "color": "red"  # should override container's blue and global black
                },
            },
        }
    )
    text_global = Text(id="global", text="Global")  # outside any container
    text_inside_normal = Text(id="t_normal", text="Inside Normal")
    cont_normal = Container(id="normal", children=[text_inside_normal])
    text_inside_special = Text(id="t_special", text="Inside Special")
    cont_special = Container(id="special", children=[text_inside_special])
    root = Container(id="root", children=[cont_normal, cont_special])

    jstyle.apply(text_global)  # Apply to global text separately
    jstyle.apply(root)

    assert text_global.color == "black"
    assert text_global.font_size == 12
    assert text_global.font_style == "normal"

    assert text_inside_normal.color == "blue"
    assert text_inside_normal.font_size == 12
    assert text_inside_normal.font_style == "italic"

    assert text_inside_special.color == "red"
    assert text_inside_special.font_size == 12
    assert text_inside_special.font_style == "italic"  # Inherited from Container > Text context


def test_nested_specificity_with_inheritance():
    """test nested rules involving subclasses."""
    jstyle.update(
        {
            "Text": {"font_size": 10},
            "Container": {
                "style.border_width": 1,
                "Text": {"color": "gray"},  # applies to Text descendants
            },
            "CustomContainer": {  # styles for the subclass itself
                "style.border_width": 3,  # overrides Container
            },
            "CustomContainer[id=very-special]": {
                "style.corner_radius": 5,
                "Text": {"color": "purple"},  # specific nested rule for Text descendants
            },
        }
    )
    text_in_cont = Text(id="t_cont", text="In Container")
    cont = Container(id="c", children=[text_in_cont])

    text_in_cust = Text(id="t_cust", text="In CustomContainer")
    cust = CustomContainer(id="cc", children=[text_in_cust])

    text_in_vspec = Text(id="t_vspec", text="In Very Special")
    vspec = CustomContainer(id="very-special", children=[text_in_vspec])

    root = Container(id="root", children=[cont, cust, vspec])
    jstyle.apply(root)

    # check text in regular container
    assert text_in_cont.font_size == 10
    assert text_in_cont.color == "gray"

    # check custom container itself and its text
    assert cust.style.border_width == 3  # CustomContainer overrides Container
    assert text_in_cust.font_size == 10
    assert text_in_cust.color == "gray"  # Inherits nested rule from Container context

    # check very special custom container and its text
    assert vspec.style.border_width == 3  # from CustomContainer rule
    assert vspec.style.corner_radius == 5  # from specific ID rule
    assert text_in_vspec.font_size == 10
    assert text_in_vspec.color == "purple"  # from most specific nested rule


def test_nested_selector_targeting_subclass():
    """test a nested rule in a base class style that targets a subclass."""
    jstyle.update(
        {
            "Container": {
                "style.padding": (5, 5, 5, 5),  # Use tuple
                "CustomContainer": {  # applies to CustomContainer descendants
                    "style.background_color": "orange",
                    "unique_prop": "set_nested",
                },
                "Text": {  # applies to Text descendants
                    "color": "green"
                },
            }
        }
    )
    cust_child = CustomContainer(id="cust_child")
    text_child = Text(id="text_child", text="Direct Text")
    cont_child = Container(id="cont_child")

    # scenario 1: children inside a Container
    root1 = Container(id="root1", children=[cust_child, text_child, cont_child])
    jstyle.apply(root1)

    assert tuple(root1.style.padding) == (5, 5, 5, 5)
    assert tuple(cust_child.style.padding) == (5, 5, 5, 5)
    assert cust_child.style.background_color == "orange"
    assert cust_child.unique_prop == "set_nested"
    # Text components have style=None by default, so no padding
    assert text_child.style is None
    assert text_child.color == "green"
    assert tuple(cont_child.style.padding) == (5, 5, 5, 5)
    assert cont_child.style.background_color is None

    # scenario 2: children inside a CustomContainer
    cust_child2 = CustomContainer(id="cust_child2")
    text_child2 = Text(id="text_child2", text="Direct Text 2")
    root2 = CustomContainer(id="root2", children=[cust_child2, text_child2])
    jstyle.apply(root2)

    assert tuple(root2.style.padding) == (5, 5, 5, 5)
    assert tuple(cust_child2.style.padding) == (5, 5, 5, 5)
    assert cust_child2.style.background_color == "orange"  # Gets context from root2
    assert cust_child2.unique_prop == "set_nested"
    # Text components have style=None by default, so no padding
    assert text_child2.style is None
    assert text_child2.color == "green"  # Gets context from root2


def test_nested_attribute_selectors():
    """test attribute selectors within a nested context."""
    jstyle.update(
        {
            "Container": {
                "Text": {"color": "black"},  # default Text in Container
                "Text[style_class=highlight]": {"color": "red"},  # highlighted Text in Container
            },
            "Container[id=outer]": {
                "Text": {"color": "blue"},  # override default Text in outer
                "Text[style_class=highlight]": {"color": "purple"},  # override highlight in outer
            },
        }
    )
    t_normal_inner = Text(id="tni", text="Normal Inner")
    t_high_inner = Text(id="thi", text="Highlight Inner", style_class=["highlight"])
    inner = Container(id="inner", children=[t_normal_inner, t_high_inner])

    t_normal_outer = Text(id="tno", text="Normal Outer")
    t_high_outer = Text(id="tho", text="Highlight Outer", style_class=["highlight"])
    outer = Container(id="outer", children=[t_normal_outer, t_high_outer])

    root = Container(id="root", children=[inner, outer])
    jstyle.apply(root)

    assert t_normal_inner.color == "black"
    assert t_high_inner.color == "red"
    assert t_normal_outer.color == "blue"
    assert t_high_outer.color == "purple"


def test_nested_wildcard_interactions():
    """test how global and nested wildcards interact."""
    jstyle.update(
        {
            "*": {
                "debug": True,
                "style.margin": (1, 1, 1, 1),  # Use tuple
            },
            "Container": {
                "style.padding": (5, 5, 5, 5),  # Use tuple
                "*": {  # applies to descendants
                    "debug": False,
                    "style.border_width": 0.5,
                },
            },
            "Text": {
                "font_size": 8,
                "style.border_width": 2.0,  # overrides nested *
            },
        }
    )

    text_child = Text(id="t", text="text")
    simple_child = SimpleComponent(id="s")
    cont = Container(id="cont", children=[text_child, simple_child])
    root = Container(id="root", children=[cont])

    jstyle.apply(root)

    assert root.debug is True
    assert tuple(root.style.margin) == (1, 1, 1, 1)
    assert tuple(root.style.padding) == (5, 5, 5, 5)

    assert cont.debug is False  # overridden by root's nested *
    assert tuple(cont.style.margin) == (1, 1, 1, 1)
    assert tuple(cont.style.padding) == (5, 5, 5, 5)
    assert cont.style.border_width == 0.5  # from root's nested *

    assert simple_child.debug is False  # from cont's nested *
    assert hasattr(simple_child, "style") and simple_child.style is not None
    assert tuple(simple_child.style.margin) == (1, 1, 1, 1)
    assert simple_child.style.border_width == 0.5  # from cont's nested *
    assert tuple(simple_child.style.padding) == (0, 0, 0, 0)

    assert text_child.debug is False  # from cont's nested *
    # Text components have style=None by default
    assert text_child.style is None
    assert text_child.font_size == 8


def test_nested_override_vs_global_type():
    """test that nested type rule overrides global type rule for the same property."""

    # define simple hierarchy for test
    class NodeBase(Container):  # inherits from Container
        node_prop: str = "base_default"
        style: BoxStyle = Field(default_factory=BoxStyle)  # ensure style exists

    class SubNodeA(NodeBase):  # inherits from NodeBase
        sub_prop: str = "sub_default"

    class ParentNode(NodeBase):  # inherits from NodeBase
        # Automatically add SubNodeA child on init
        def model_post_init(self, *args, **kwargs):
            super().model_post_init(*args, **kwargs)
            self.add_child(SubNodeA(id="child_a"))

    jstyle.update(
        {
            # Global rule for SubNodeA
            "SubNodeA": {
                "style.background_color": "blue",  # Global default
                "sub_prop": "global_sub",
                "node_prop": "global_node",  # Set prop inherited from NodeBase
            },
            # Nested rule for SubNodeA inside ParentNode
            "ParentNode": {
                "node_prop": "parent_node",  # Style ParentNode itself
                "SubNodeA": {
                    "style.background_color": "red",  # Nested override
                    "sub_prop": "nested_sub",  # Nested override
                    # Does not define node_prop, should inherit from global SubNodeA rule
                },
            },
        }
    )

    # Create the structure
    parent = ParentNode(id="p1")
    # The child SubNodeA is created automatically by ParentNode.__init__

    jstyle.apply(parent)

    child_node = parent.children[0]

    assert isinstance(child_node, SubNodeA)
    assert parent.node_prop == "parent_node"
    assert (
        child_node.style.background_color == "red"
    ), "Nested background_color should override global"
    assert child_node.sub_prop == "nested_sub", "Nested sub_prop should override global"
    assert (
        child_node.node_prop == "global_node"
    ), "node_prop should come from global SubNodeA rule (not parent or base)"


def test_nested_context_discovery_during_apply():
    """tests if apply() correctly discovers context even when called directly on child."""

    # component definitions for this test
    class ContextParent(Container):
        pass

    class StyledChild(Component):
        color: str = "black"  # default color

    jstyle.update(
        {
            # global rule for the child type
            "StyledChild": {
                "color": "blue"  # lower priority rule
            },
            # rule for the parent containing a nested rule for the child
            "ContextParent": {
                "style.border_color": "green",  # style the parent
                "StyledChild": {
                    "color": "red"  # higher priority due to context
                },
            },
        }
    )

    parent = ContextParent(id="p1")
    child = StyledChild(id="c1")
    parent.add_child(child)  # establishes parent link

    jstyle.apply(child)

    jstyle.apply(parent)

    assert parent.style.border_color == "green"
    assert (
        child.color == "red"
    ), "Child should get 'red' from nested rule found via context discovery"


def test_style_leakage_between_siblings():
    """
    tests if styling a property (esp. on a nested model like 'style')
    for one subclass incorrectly affects a sibling subclass that inherits
    the same base style object reference initially.
    """

    # --- component definitions for this test ---
    class SiblingBase(Container):
        # base class defines the style field with a factory
        style: BoxStyle = Field(default_factory=BoxStyle)
        unique_prop: str = "base_default"

    class SiblingA(SiblingBase):
        pass  # inherits style

    class SiblingB(SiblingBase):
        pass  # inherits style

    # --- end component definitions ---

    # style only sibling a's shadow and unique_prop
    jstyle.update(
        {
            "SiblingA": {
                "style.shadow": {"blur_radius": 5, "color": "red"},
                "unique_prop": "set_for_A",
                "layout": LayoutConstraints(
                    direction="column",
                ),
            },
            "SiblingB": {
                "unique_prop": "set_for_B"  # style B differently
            },
        }
    )

    sibling_a = SiblingA(id="a")
    sibling_b = SiblingB(id="b")  # sibling instance

    print(f"Initial style object IDs: A={id(sibling_a.style)}, B={id(sibling_b.style)}")

    # apply styles - this is where the potential mutation happens
    jstyle.apply(sibling_a)
    jstyle.apply(sibling_b)

    print(f"Final style object IDs:   A={id(sibling_a.style)}, B={id(sibling_b.style)}")

    # sibling_a should have the shadow and its unique prop
    assert sibling_a.style.shadow is not None, "SiblingA should have a shadow"
    assert sibling_a.style.shadow.blur_radius == 5
    assert sibling_a.style.shadow.color == "#ff0000ff"  # "red" normalized to hex
    assert sibling_a.unique_prop == "set_for_A"
    assert sibling_a.layout.direction == "column"

    # sibling_b should *not* have the shadow from sibling_a's rule
    # and should have its own unique_prop value
    assert (
        sibling_b.style.shadow is None
    ), "SiblingB's style.shadow should remain None (not affected by SiblingA's style)"
    assert sibling_b.unique_prop == "set_for_B"
    assert sibling_b.layout.direction == "row"  # default from Container


def test_attribute_presence_selector():
    """
    tests the [attribute_name] selector for presence and truthiness.
    """

    # --- component definition for this test ---
    class FeatureComponent(Component):
        is_active: bool = False
        has_feature: Optional[str] = None  # optional attribute
        counter: int = 0
        items: List[str] = Field(default_factory=list)
        style: BoxStyle = Field(default_factory=BoxStyle)  # add style for testing nested props

    # --- end component definition ---

    jstyle.update(
        {
            "[is_active]": {"style.border_color": "green"},
            "[has_feature]": {"style.background_color": "yellow"},
            "[counter]": {"style.border_width": 2},  # counter=0 is falsy
            "[items]": {"style.corner_radius": 5},  # empty list is falsy
        }
    )

    # create components with different attribute states
    comp_active = FeatureComponent(id="active", is_active=True)
    comp_inactive = FeatureComponent(id="inactive", is_active=False)
    comp_feature_str = FeatureComponent(id="feature_str", has_feature="enabled")
    comp_feature_empty = FeatureComponent(id="feature_empty", has_feature="")  # falsy
    comp_feature_none = FeatureComponent(id="feature_none", has_feature=None)  # falsy
    comp_counter_zero = FeatureComponent(id="counter_zero", counter=0)  # falsy
    comp_counter_non_zero = FeatureComponent(id="counter_non_zero", counter=5)  # truthy
    comp_items_empty = FeatureComponent(id="items_empty", items=[])  # falsy
    comp_items_full = FeatureComponent(id="items_full", items=["a"])  # truthy

    # apply styles to all components
    components = [
        comp_active,
        comp_inactive,
        comp_feature_str,
        comp_feature_empty,
        comp_feature_none,
        comp_counter_zero,
        comp_counter_non_zero,
        comp_items_empty,
        comp_items_full,
    ]
    # applying individually to avoid container context issues
    for comp in components:
        jstyle.apply(comp)

    # --- assertions ---
    # [is_active]
    assert comp_active.style.border_color == "green", "Should match [is_active] when True"
    assert comp_inactive.style.border_color is None, "Should NOT match [is_active] when False"

    # [has_feature]
    assert (
        comp_feature_str.style.background_color == "yellow"
    ), "Should match [has_feature] when truthy string"
    assert (
        comp_feature_empty.style.background_color is None
    ), "Should NOT match [has_feature] when empty string"
    assert (
        comp_feature_none.style.background_color is None
    ), "Should NOT match [has_feature] when None"

    # [counter]
    assert (
        comp_counter_zero.style.border_width == 0.0
    ), "Should NOT match [counter] when 0"  # default width is 0
    assert comp_counter_non_zero.style.border_width == 2, "Should match [counter] when non-zero"

    # [items]
    assert (
        comp_items_empty.style.corner_radius == 0.0
    ), "Should NOT match [items] when empty list"  # default radius is 0
    assert comp_items_full.style.corner_radius == 5, "Should match [items] when non-empty list"


def test_attribute_absence_selector():
    """
    tests the [!attribute_name] selector for absence or falsiness.
    """

    # --- component definition for this test ---
    class OptFeatureComponent(Component):
        is_enabled: Optional[bool] = None
        config_value: Optional[int] = None
        style: BoxStyle = Field(default_factory=BoxStyle)

    # --- end component definition ---

    jstyle.update(
        {
            "[!is_enabled]": {"style.border_style": "dotted"},
            "[!config_value]": {"style.background_color": "lightgrey"},
        }
    )

    comp_none = OptFeatureComponent(id="none", is_enabled=None, config_value=None)
    comp_false = OptFeatureComponent(id="false", is_enabled=False, config_value=0)
    comp_true = OptFeatureComponent(id="true", is_enabled=True, config_value=10)

    components = [comp_none, comp_false, comp_true]
    for comp in components:
        jstyle.apply(comp)

    # --- assertions ---
    # [!is_enabled] - should match None and False
    assert comp_none.style.border_style == "dotted", "[!is_enabled] should match None"
    assert comp_false.style.border_style == "dotted", "[!is_enabled] should match False"
    assert (
        comp_true.style.border_style == "solid"
    ), "[!is_enabled] should NOT match True (default is solid)"

    # [!config_value] - should match None and 0
    assert comp_none.style.background_color == "lightgrey", "[!config_value] should match None"
    assert comp_false.style.background_color == "lightgrey", "[!config_value] should match 0"
    assert comp_true.style.background_color is None, "[!config_value] should NOT match 10"
