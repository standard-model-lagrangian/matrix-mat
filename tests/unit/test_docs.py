"""
Unit tests for documentation truthfulness (WS4.2).
Validates that relative markdown links across docs and README point to existing files.
"""

from __future__ import annotations

import re
from pathlib import Path
import pytest
import matrix_mat


def test_docs_reference_existing_files():
    """Extract relative markdown links from README.md and docs/*.md and verify targets exist."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    markdown_files = [repo_root / "README.md"] + list((repo_root / "docs").glob("*.md"))

    # Regex to find markdown links: [text](path)
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    missing_links = []

    for md_file in markdown_files:
        if not md_file.exists():
            continue
        content = md_file.read_text(encoding="utf-8")
        for match in link_pattern.finditer(content):
            url = match.group(2).strip()
            # Ignore external URLs, anchors, mailto, badges
            if url.startswith(("http://", "https://", "#", "mailto:")):
                continue
            # Strip anchors like docs/foo.md#section
            clean_url = url.split("#")[0].strip()
            if not clean_url:
                continue

            target_path = (md_file.parent / clean_url).resolve()
            if not target_path.exists():
                missing_links.append(f"{md_file.relative_to(repo_root)}: '{url}' -> {target_path}")

    assert not missing_links, f"Broken relative links found in documentation:\n" + "\n".join(missing_links)
