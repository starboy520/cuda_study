#!/usr/bin/env python3
"""Check relative links in Markdown files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


LINK_RE = re.compile(
    r"(?<!!)\[[^\]\n]*\]\(\s*(<[^>\n]*>|(?:\\.|[^()\n]|\([^()\n]*\))+?)\s*\)"
)
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
IGNORED_SCHEMES = {"http", "https", "mailto", "tel"}


def display_path(path: Path, root: Path) -> str:
    """Return a stable repository-relative path when possible."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def is_excluded(path: Path, root: Path, prefixes: list[str]) -> bool:
    """Return whether path is under any repository-relative prefix."""
    relative = display_path(path, root)
    return any(
        relative == prefix.rstrip("/")
        or relative.startswith(prefix.rstrip("/") + "/")
        for prefix in prefixes
    )


def markdown_files(inputs: list[str], root: Path, prefixes: list[str]):
    """Yield unique Markdown files selected by files and directories."""
    seen: set[Path] = set()
    for value in inputs:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve()

        if candidate.is_dir():
            paths = sorted(candidate.rglob("*.md"))
        elif candidate.is_file() and candidate.suffix.lower() == ".md":
            paths = [candidate]
        else:
            continue

        for path in paths:
            path = path.resolve()
            if path not in seen and not is_excluded(path, root, prefixes):
                seen.add(path)
                yield path


def destination(raw: str) -> str:
    """Extract the destination from a Markdown inline-link target."""
    value = raw.strip()
    if value.startswith("<"):
        closing = value.find(">", 1)
        if closing != -1:
            return value[1:closing]
    return re.split(r"\s+", value, maxsplit=1)[0]


def check_file(source: Path, root: Path) -> list[tuple[int, str, Path]]:
    """Return broken links as (line, raw target, resolved path)."""
    broken: list[tuple[int, str, Path]] = []
    fence: tuple[str, int] | None = None

    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fence_match = FENCE_RE.match(line)
            if fence_match:
                marker = fence_match.group(1)
                marker_char = marker[0]
                if fence is None:
                    fence = (marker_char, len(marker))
                elif fence[0] == marker_char and len(marker) >= fence[1]:
                    fence = None
                continue
            if fence is not None:
                continue

            for match in LINK_RE.finditer(line):
                raw = match.group(1)
                target = destination(raw)
                if not target or target.startswith("#"):
                    continue

                parsed = urlsplit(target)
                if parsed.scheme.lower() in IGNORED_SCHEMES:
                    continue

                link_path = unquote(parsed.path).replace("\\ ", " ")
                if not link_path:
                    continue
                resolved = (source.parent / link_path).resolve()
                if not resolved.exists():
                    broken.append((line_number, raw, resolved))

    return broken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Markdown files or directories (default: repository root)",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="repository-relative path prefix to exclude (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    inputs = args.paths or [str(root)]
    found_broken = False

    for source in markdown_files(inputs, root, args.exclude_prefix):
        for line_number, raw, resolved in check_file(source, root):
            found_broken = True
            print(
                f"{display_path(source, root)}:{line_number}: "
                f"{raw} -> {display_path(resolved, root)}"
            )

    return 1 if found_broken else 0


if __name__ == "__main__":
    sys.exit(main())
