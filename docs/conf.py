project = "agent-exam"
project_copyright = "2026, Zyte Group Ltd"
author = "Zyte Group Ltd"
release = "0.0.0"

extensions = [
    "sphinx_scrapy",
]

exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
language = "en"
nitpicky = True
source_suffix = {".rst": "restructuredtext"}

# `--no-llm`, `--without-skill` and friends are quoted all over the CLI
# reference; smart quotes would render them with en dashes.
smartquotes = False
