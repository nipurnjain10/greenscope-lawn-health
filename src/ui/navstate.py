"""Page objects registered by the application before a page renders.

Streamlit links to callable pages need the Page object, not a file path.
"""

from __future__ import annotations

PAGES_BY_PATH: dict = {}


def register(pages) -> None:
    PAGES_BY_PATH.clear()
    for page in pages:
        PAGES_BY_PATH[page.url_path] = page
