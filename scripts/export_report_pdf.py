"""Export the Chinese Markdown experiment report as a styled PDF."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import mathtext  # noqa: E402
import mistune  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "reports" / "实验报告.md"
STYLE = PROJECT_ROOT / "reports" / "pdf-style.css"
TEMP_DIR = PROJECT_ROOT / "tmp" / "pdfs" / "experiment-report"
HTML_OUTPUT = TEMP_DIR / "实验报告.html"
PDF_OUTPUT = PROJECT_ROOT / "output" / "pdf" / "实验报告.pdf"
NODE = Path("/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node")
NODE_MODULES = Path(
    "/Applications/ChatGPT.app/Contents/Resources/cua_node/lib/node_modules"
)
PRINT_SCRIPT = PROJECT_ROOT / "scripts" / "print_html_pdf.cjs"


def math_data_uri(expression: str) -> str:
    buffer = io.BytesIO()
    mathtext.math_to_image(
        f"${expression}$",
        buffer,
        format="svg",
        dpi=180,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def render_inline_math(renderer, text: str) -> str:
    del renderer
    return (
        f'<img class="math-inline" src="{math_data_uri(text)}" '
        f'alt="{mistune.escape(text)}">'
    )


def render_block_math(renderer, text: str) -> str:
    del renderer
    return (
        '<div class="math-block">'
        f'<img class="math-block-image" src="{math_data_uri(text)}" '
        f'alt="{mistune.escape(text)}">'
        "</div>"
    )


def build_html() -> str:
    renderer = mistune.HTMLRenderer(escape=False)
    markdown = mistune.create_markdown(
        renderer=renderer,
        plugins=["table", "math"],
    )
    renderer.register("inline_math", render_inline_math)
    renderer.register("block_math", render_block_math)

    body = markdown(SOURCE.read_text(encoding="utf-8"))
    css = STYLE.read_text(encoding="utf-8")
    base_url = SOURCE.parent.resolve().as_uri() + "/"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <base href="{base_url}">
  <title>CIFAR-10 训练 ResNet 实验报告</title>
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not STYLE.is_file():
        raise FileNotFoundError(STYLE)
    if not NODE.is_file():
        raise FileNotFoundError(NODE)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUTPUT.write_text(build_html(), encoding="utf-8")

    environment = dict(os.environ)
    environment["NODE_PATH"] = str(NODE_MODULES)
    subprocess.run(
        [str(NODE), str(PRINT_SCRIPT), str(HTML_OUTPUT), str(PDF_OUTPUT)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    print(PDF_OUTPUT)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PDF export failed: {error}", file=sys.stderr)
        raise
