"""`!Text "hello"` is sugar for `!Text { text: "hello" }`."""

import dracon

from jeanplot import Text, make_plot_context


def test_text_model_validate_from_string():
    t = Text.model_validate("hello")
    assert t.text == "hello"


def test_text_model_validate_from_dict_still_works():
    t = Text.model_validate({"text": "hi", "color": "red"})
    assert t.text == "hi"
    assert t.color == "red"


def test_yaml_text_tag_with_scalar():
    t = dracon.loads("!Text hello world", context=make_plot_context())
    assert isinstance(t, Text)
    assert t.text == "hello world"


def test_yaml_text_tag_with_quoted_scalar_preserves_markup():
    t = dracon.loads('!Text "X_2(eBFP2)"', context=make_plot_context())
    assert t.text == "X_2(eBFP2)"


def test_yaml_text_tag_mapping_form_still_works():
    t = dracon.loads("!Text { text: foo, color: red }", context=make_plot_context())
    assert t.text == "foo"
    assert t.color == "red"
