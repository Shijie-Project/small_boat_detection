#!/usr/bin/env python
"""Launcher kept so ``python webui/webui.py`` still works; see ``webui/README.md``.

For hot reload while editing, run ``gradio webui/app.py`` instead.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webui.app import main


if __name__ == "__main__":
    main()
