import os
import sys

import pytest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Tkinter smoke test needs a display")
def test_main_ui_small_window_instantiates_notebook_tabs():
    import tkinter as tk
    import main_ui

    root = tk.Tk()
    try:
        root.geometry("520x420")
        app = main_ui.AppUI(root)
        notebooks = [child for child in root.winfo_children()[0].winfo_children() if child.winfo_class() == "TNotebook"]
        assert notebooks
        assert app.btn_extract_full.winfo_exists()
        assert app.btn_train_suite_all.winfo_exists()
        assert app.btn_live.winfo_exists()
        assert app.btn_rl_start.winfo_exists()
    finally:
        root.destroy()
