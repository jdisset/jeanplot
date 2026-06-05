import re
from typing import Any, Sequence

from dracon import parse_locator, resolve_one


class ComponentTreeAdapter:
    """TreeAdapter over the live Component scene-graph for the dracon locator.

    type_names is MRO class names ONLY (a bare word selector = a type match);
    parent reproduces the `.parent` walk the old Selector hardcoded."""

    def parent(self, node: Any) -> Any | None:
        return getattr(node, "parent", None)

    def children(self, node: Any) -> Sequence[Any]:
        # children + anchor_points: the navigable tree the old find_component_by_path
        # walked (anchors are valid connector/attachment endpoints). Forward-only —
        # styling matches walk parent, never this.
        kids = list(getattr(node, "children", None) or [])
        kids.extend(
            a for a in (getattr(node, "anchor_points", None) or []) if not any(a is k for k in kids)
        )
        return kids

    def type_names(self, node: Any) -> Sequence[str]:
        return [k.__name__ for k in type(node).__mro__]

    def attr(self, node: Any, name: str) -> Any:
        return getattr(node, name, None)


COMPONENT_ADAPTER = ComponentTreeAdapter()
_BARE_ID = re.compile(r"^[\w.-]+$")


def resolve_component(frame: Any, path: str) -> Any | None:
    """Resolve a locator string against the live component tree, framed at `frame`.
    A bare id (no axis tokens) is sugar for a rooted descendant-by-id match, so
    legacy `attached_to: "some_id"` keeps resolving."""
    s = path.strip()
    loc = parse_locator(f"/**[id={s}]" if _BARE_ID.fullmatch(s) else s)
    return resolve_one(frame, loc, COMPONENT_ADAPTER)
