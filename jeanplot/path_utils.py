# jeanplot/path_utils.py
from typing import Optional, Union, Dict, Any
from .component import Component


def find_component_by_path(root: Component, path: str) -> Optional[Component]:
    """find component by path relative to root container"""
    if not path:
        return None

    parts = path.split("/")
    current = root

    for part in parts:
        if part == "":
            continue

        if not hasattr(current, "children"):
            return None

        found = False
        for child in current.children:
            if getattr(child, "id", None) == part:
                current = child
                found = True
                break

        if not found:
            allids = [getattr(child, "id", None) for child in current.children]
            raise ValueError(f"Component {part} not found in {allids} at path {path}")

    return current


def resolve_component_ref(v, info):
    """validator for component references that accepts paths"""
    from .component import Component

    if isinstance(v, Component) or v is None:
        return v

    if isinstance(v, str):
        # if we have parent context, try to resolve the path
        values = info.data
        parent = values.get("parent")
        print(f"parent: {parent}")
        if parent is not None:
            # find the root container
            root = parent
            while root.parent is not None:
                root = root.parent

            # try to resolve path
            component = find_component_by_path(root, v)
            if component is not None:
                return component

    return v
