"""Load research.py in unit tests without requiring network/model libraries."""

import importlib.util
import sys
import types
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))


def _stub_if_missing(name: str, attributes: dict) -> None:
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        sys.modules[name] = module


class _Unused:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("This dependency should not be used by unit tests")


_stub_if_missing("chromadb", {"EphemeralClient": _Unused})
_stub_if_missing("ollama", {})
_stub_if_missing("requests", {})
_stub_if_missing("bs4", {"BeautifulSoup": _Unused})
_stub_if_missing("ddgs", {"DDGS": _Unused})
