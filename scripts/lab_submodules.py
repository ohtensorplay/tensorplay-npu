"""Fetch the pinned third-party checkouts into a source tree.

The lab container cannot use git's smart-HTTP transport (the connection resets
mid-negotiation), and a recovery checkout arrives as a tarball: no ``.git``,
no submodule metadata, only empty directories.  This reads the pinned commits
back off the hosting service's commit tree and downloads each checkout as a
tarball, which the archive endpoint serves quickly and completely.

Standard library only, because nothing is installed yet when this runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

API = "https://api.github.com/repos"
ARCHIVE = "https://codeload.github.com"

#: Only what a CPU-only build of the core actually compiles.  The optional
#: math kit is turned off in the configure step, so its checkouts stay out.
NEEDED = ("gloo", "sleef")


def read_manifest(text: str) -> dict[str, dict[str, str]]:
    """Parse the submodule manifest into ``{section: {key: value}}``.

    Each section is a ``[submodule "<name>"]`` header line followed by
    indented ``key = value`` lines until the next header.
    """

    blocks: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in text.splitlines():
        line = line.strip()
        header = re.match(r'\[submodule\s+"([^"]+)"\]', line)
        if header is not None:
            current = blocks[header.group(1)] = {}
            continue
        if current is not None and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            current[key] = value
    return blocks


def slug(url: str) -> str:
    """``owner/name`` of a hosting URL, with the scheme and suffix dropped."""

    tail = url.split("://", 1)[-1].split("/", 1)[-1]
    return tail.removesuffix(".git")


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def pinned(project: str, commit: str) -> dict[str, str]:
    """Every commit entry of a tree, by path, recursing into directories.

    The plain tree call lists ``third_party`` as a single directory entry,
    which hides the pinned commits, so the recursive form is required.
    """

    entries = json.loads(get(f"{API}/{project}/git/trees/{commit}?recursive=1"))["tree"]
    return {str(e["path"]): str(e["sha"]) for e in entries if e["type"] == "commit"}


def unpack(url: str, destination: str) -> None:
    """Expand an archive tarball into ``destination`` without its top folder."""

    shutil.rmtree(destination, ignore_errors=True)
    os.makedirs(destination, exist_ok=True)
    completed = subprocess.run(
        f"curl -fsSL --retry 3 --connect-timeout 20 '{url}' "
        f"| tar -xz --strip-components=1 -C '{destination}'",
        shell=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not fetch {url}")


def fetch_nested(project: str, commit: str, directory: str) -> None:
    """Fill the checkouts declared inside an already fetched checkout.

    The vector math library needs its float-helper checkout present to build
    the scalar fallback objects, and the configure step fails without it.
    """

    manifest = os.path.join(directory, ".gitmodules")
    if not os.path.exists(manifest):
        return
    with open(manifest) as handle:
        blocks = read_manifest(handle.read())
    child_pins = pinned(project, commit)
    for section, fields in blocks.items():
        url, path, sha = fields.get("url"), fields.get("path"), child_pins.get(section)
        if not url or not path:
            raise RuntimeError(f"{section} is declared but has no url/path")
        if not sha:
            print(f"SKIP NESTED {section}: not pinned by {commit}")
            continue
        unpack(f"{ARCHIVE}/{slug(url)}/tar.gz/{sha}", os.path.join(directory, path))
        print(f"FETCH NESTED {section} {sha[:8]}", flush=True)


def fetch_all(root: str, project: str, ref: str) -> None:
    with open(os.path.join(root, ".gitmodules")) as handle:
        blocks = read_manifest(handle.read())
    if not blocks:
        raise RuntimeError(f"{root}/.gitmodules declares no submodules")
    parent_pins = pinned(project, ref)
    for section, fields in blocks.items():
        name = section.rsplit("/", 1)[-1]
        if name not in NEEDED:
            continue
        url, path, sha = fields.get("url"), fields.get("path"), parent_pins.get(section)
        if not url or not path or not sha:
            raise RuntimeError(f"{name} is declared but not resolvable")
        directory = os.path.join(root, path)
        unpack(f"{ARCHIVE}/{slug(url)}/tar.gz/{sha}", directory)
        print(f"FETCH {name} {sha[:8]}", flush=True)
        fetch_nested(slug(url), sha, directory)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Fetch pinned third-party checkouts.")
    parser.add_argument("--root", required=True, help="checkout holding the manifest")
    parser.add_argument("--project", required=True, help="owner/name of that checkout")
    parser.add_argument("--ref", default="main", help="branch or commit to read pins from")
    args = parser.parse_args(argv)
    fetch_all(args.root, args.project, args.ref)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
