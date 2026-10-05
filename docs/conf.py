import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

project = "Syncretize"
copyright = "Synthesizer contributors"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
templates_path = []
exclude_patterns = ["_build"]
html_theme = "alabaster"
html_static_path = ["_static"]
html_logo = "_static/syncretize_logo.png"
autodoc_member_order = "bysource"
