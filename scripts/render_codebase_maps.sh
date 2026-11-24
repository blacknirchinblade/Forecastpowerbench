#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_DIR="$ROOT/docs/codebase_maps"
OUTPUT_DIR="$SOURCE_DIR/rendered"
TOOLS_DIR="$ROOT/.codebase_map_tools"
NODE_DIR="$TOOLS_DIR/node"
D2_DIR="$TOOLS_DIR/d2"

MARKMAP_VERSION=0.18.12
MERMAID_VERSION=11.16.0
VIZ_VERSION=3.28.0
D2_VERSION=0.7.1

mkdir -p "$OUTPUT_DIR" "$NODE_DIR" "$D2_DIR"

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "[MAPS] Node.js and npm are required for Markmap, Mermaid, and Viz.js rendering." >&2
    exit 1
fi

if [ ! -f "$NODE_DIR/package.json" ]; then
    npm --prefix "$NODE_DIR" init -y >/dev/null
fi
npm --prefix "$NODE_DIR" install --no-audit --no-fund \
    "markmap-cli@$MARKMAP_VERSION" \
    "@mermaid-js/mermaid-cli@$MERMAID_VERSION" \
    "@viz-js/viz@$VIZ_VERSION" >/dev/null

cat > "$TOOLS_DIR/puppeteer.json" <<'JSON'
{"args":["--no-sandbox","--disable-setuid-sandbox"]}
JSON

"$NODE_DIR/node_modules/.bin/markmap" \
    "$SOURCE_DIR/01_POWERBENCH_MINDMAP.md" \
    --offline --no-open -o "$OUTPUT_DIR/01_POWERBENCH_MINDMAP.html"
"$NODE_DIR/node_modules/.bin/markmap" \
    "$SOURCE_DIR/02_SYMBOL_MINDMAP.md" \
    --offline --no-open -o "$OUTPUT_DIR/02_SYMBOL_MINDMAP.html"

for source in "$SOURCE_DIR"/*.mmd; do
    name=$(basename "$source" .mmd)
    "$NODE_DIR/node_modules/.bin/mmdc" -q -p "$TOOLS_DIR/puppeteer.json" \
        -b transparent -i "$source" -o "$OUTPUT_DIR/$name.svg"
done

cat > "$TOOLS_DIR/render_dot.mjs" <<'JS'
import fs from "node:fs";
import { instance } from "./node/node_modules/@viz-js/viz/dist/viz.js";
const [input, output] = process.argv.slice(2);
const viz = await instance();
fs.writeFileSync(output, viz.renderString(fs.readFileSync(input, "utf8"), { format: "svg", engine: "dot" }));
JS
for source in "$SOURCE_DIR"/*.dot; do
    name=$(basename "$source" .dot)
    node "$TOOLS_DIR/render_dot.mjs" "$source" "$OUTPUT_DIR/$name.svg"
done

D2_BIN="$D2_DIR/bin/d2"
if [ ! -x "$D2_BIN" ]; then
    archive="$D2_DIR/d2.tar.gz"
    partial="$archive.part"
    rm -f "$partial"
    curl -fL --retry 5 --retry-all-errors --retry-delay 2 \
        "https://github.com/terrastruct/d2/releases/download/v${D2_VERSION}/d2-v${D2_VERSION}-linux-amd64.tar.gz" \
        -o "$partial"
    gzip -t "$partial"
    mv "$partial" "$archive"
    tar -xzf "$archive" -C "$D2_DIR" --strip-components=1
fi
"$D2_BIN" --theme 300 "$SOURCE_DIR/12_PUBLICATION_ARCHITECTURE.d2" "$OUTPUT_DIR/12_PUBLICATION_ARCHITECTURE.svg"

python3 - "$SOURCE_DIR" "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import html
import sys

source = Path(sys.argv[1])
output = Path(sys.argv[2])
items = []
for path in sorted(output.iterdir()):
    if path.name == "index.html":
        continue
    kind = "Interactive mind map" if path.suffix == ".html" else "Rendered diagram"
    items.append(f'<a class="card" href="{html.escape(path.name)}"><strong>{html.escape(path.stem)}</strong><span>{kind}</span></a>')
index = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PowerBench Codebase Atlas</title><style>
body{{font-family:Georgia,serif;background:#f6f1e7;color:#17202a;margin:0}}main{{max-width:1100px;margin:auto;padding:48px 24px}}h1{{font-size:48px;margin:0 0 8px}}p{{font:18px/1.5 sans-serif;color:#46515c}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:32px}}.card{{background:#fff;border:1px solid #d8cdb9;border-left:7px solid #0e7490;border-radius:8px;padding:20px;text-decoration:none;color:#17202a;box-shadow:0 5px 16px #594c3820}}.card:hover{{transform:translateY(-2px)}}strong,span{{display:block}}span{{font:14px sans-serif;color:#68737d;margin-top:8px}}</style></head><body><main><h1>PowerBench Codebase Atlas</h1><p>Interactive mind maps, runtime workflows, dependency graphs, call graphs, and publication architecture. Sources and coverage reports are one directory above.</p><div class="grid">{''.join(items)}</div></main></body></html>'''
(output / "index.html").write_text(index)
PY

echo "[MAPS] Rendered atlas: $OUTPUT_DIR/index.html"
