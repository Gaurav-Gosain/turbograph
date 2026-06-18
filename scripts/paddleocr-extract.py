#!/usr/bin/env python3
"""Extract plain text from an image or PDF using PaddleOCR (PP-OCRv6) and print
it to stdout, one recognized line per output line.

turbograph treats extraction as a pluggable external command: wire this script in
with `--ocr-cmd`, for example

    turbograph serve --ocr-cmd "/home/me/.turbograph-ocr/bin/python \
        /path/to/scripts/paddleocr-extract.py {in}"

The {in} token is replaced by turbograph with the path to the input file. The
language can be set with the PADDLE_LANG environment variable (default: en).

This script is intentionally outside the Go code: turbograph is the graph and RAG
layer, and parsers are brought by the user. Swap PaddleOCR for any tool that
reads a file path and writes text to stdout.
"""

import os
import sys


def texts_from(result):
    """Yield recognized text lines from one PaddleOCR result item, tolerating the
    shape differences between PaddleOCR releases."""
    if result is None:
        return
    if isinstance(result, dict):
        for line in result.get("rec_texts", []) or []:
            yield line
        return
    rec = getattr(result, "rec_texts", None)
    if rec:
        for line in rec:
            yield line
        return
    data = getattr(result, "json", None)
    if isinstance(data, dict):
        for line in data.get("res", {}).get("rec_texts", []) or []:
            yield line


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: paddleocr-extract.py <file>\n")
        sys.exit(2)
    from paddleocr import PaddleOCR

    # Some CPU builds of paddlepaddle crash in the oneDNN backend; disabling it
    # falls back to a stable path. Harmless where oneDNN works.
    try:
        ocr = PaddleOCR(lang=os.environ.get("PADDLE_LANG", "en"), enable_mkldnn=False)
    except TypeError:
        ocr = PaddleOCR(lang=os.environ.get("PADDLE_LANG", "en"))
    lines = []
    for item in ocr.predict(sys.argv[1]):
        lines.extend(texts_from(item))
    sys.stdout.write("\n".join(lines))


if __name__ == "__main__":
    main()
