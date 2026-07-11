#!/usr/bin/env python3
"""Check relative inline Markdown links in Markdown files.

Supported syntax is single-line inline Markdown links. Images, fenced code
blocks, external URLs, and anchor-only links are ignored. Reference-style
links and multiline links are not supported.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


LINK_RE = re.compile(
    r"(?<!!)\[[^\]\n]*\]\(\s*(<[^>\n]*>|(?:\\.|[^()\n]|\([^()\n]*\))+?)\s*\)"
)
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
WINDOWS_DRIVE_ROOT_RE = re.compile(r"^[A-Za-z]:$")
WINDOWS_DRIVE_RELATIVE_PATH_RE = re.compile(r"^[A-Z]:[^\\/]")


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
            paths = sorted(
                path for path in candidate.rglob("*")
                if path.is_file() and path.suffix.lower() == ".md"
            )
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


def check_file(
    source: Path,
) -> tuple[list[tuple[int, str, Path]], list[tuple[int, str, str]]]:
    """Return broken links and input errors found in one Markdown file."""
    broken: list[tuple[int, str, Path]] = []
    input_errors: list[tuple[int, str, str]] = []
    fence: tuple[str, int] | None = None

    try:
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

                    if target.startswith("//"):
                        continue

                    if (
                        WINDOWS_DRIVE_PATH_RE.match(target)
                        or WINDOWS_DRIVE_RELATIVE_PATH_RE.match(target)
                    ):
                        input_errors.append(
                            (
                                line_number,
                                raw,
                                "Windows paths are not checkable",
                            )
                        )
                        continue

                    if WINDOWS_DRIVE_ROOT_RE.match(target):
                        input_errors.append(
                            (
                                line_number,
                                raw,
                                "Windows paths are not checkable",
                            )
                        )
                        continue

                    try:
                        parsed = urlparse(target)
                    except ValueError as error:
                        input_errors.append(
                            (line_number, raw, f"invalid URI: {error}")
                        )
                        continue

                    if parsed.scheme:
                        continue

                    link_path = parsed.path
                    link_path = unquote(link_path).replace("\\ ", " ")
                    if not link_path:
                        continue
                    resolved = (source.parent / link_path).resolve()
                    if not resolved.exists():
                        broken.append((line_number, raw, resolved))
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read Markdown file {source}: {error}") from None

    return broken, input_errors


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
    broken: list[tuple[Path, int, str, Path]] = []
    input_errors: list[tuple[Path, int, str, str]] = []
    scanned = 0
    had_input_error = False

    for value in args.paths:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            print(f"input path does not exist: {value}", file=sys.stderr)
            input_errors.append((candidate, 0, value, "input path does not exist"))
            had_input_error = True
        elif not candidate.is_dir() and not (
            candidate.is_file() and candidate.suffix.lower() == ".md"
        ):
            print(
                f"input path is neither a Markdown file nor a directory: {value}",
                file=sys.stderr,
            )
            input_errors.append(
                (
                    candidate,
                    0,
                    value,
                    "input path is neither a Markdown file nor a directory",
                )
            )
            had_input_error = True

    for source in markdown_files(inputs, root, args.exclude_prefix):
        scanned += 1
        try:
            file_broken, file_input_errors = check_file(source)
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            input_errors.append((source, 0, "", str(error)))
            continue
        for line_number, raw, resolved in file_broken:
            broken.append((source, line_number, raw, resolved))
            print(
                f"{display_path(source, root)}:{line_number}: "
                f"{raw} -> {display_path(resolved, root)}"
            )

        for line_number, raw, reason in file_input_errors:
            input_errors.append((source, line_number, raw, reason))
            print(
                f"{display_path(source, root)}:{line_number}: "
                f"{raw}: {reason}",
                file=sys.stderr,
            )

    had_input_error = had_input_error or bool(input_errors)

    print(f"broken relative Markdown targets: {len(broken)}", file=sys.stderr)
    if input_errors:
        print(f"checker input errors: {len(input_errors)}", file=sys.stderr)

    if broken:
        return 1

    if had_input_error:
        return 1

    if scanned == 0:
        print("no Markdown files were scanned", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
