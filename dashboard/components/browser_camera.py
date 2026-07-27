from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT_DIR = Path(__file__).resolve().parent
_browser_camera = components.declare_component(
    "browser_camera",
    path=str(_COMPONENT_DIR),
)


def browser_camera(
    *,
    key: str,
    height: int = 760,
) -> dict | None:
    return _browser_camera(
        key=key,
        default=None,
        height=height,
        storage_key=key,
    )
