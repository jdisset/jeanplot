import difflib
from collections.abc import Iterable


def closest_name(name: str, candidates: Iterable[str], default: str | None = None) -> str | None:
    """Fuzzy lookup of a name in a set of candidates (case-insensitive).

    Returns the best match, or `default` if no candidate is close enough.
    """
    cand_list = list(candidates)
    lookup = {c.lower(): c for c in cand_list}
    matches = difflib.get_close_matches(name.lower(), lookup.keys(), n=1)
    if not matches:
        return default
    return lookup[matches[0]]
