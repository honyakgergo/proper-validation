"""Which engine produced this report.

The footer has always promised that "re-running with the same inputs and seed
reproduces every number above", and until this module existed that promise had
no referent: nothing on the page said *which* `proper_validation` ran. A report
that cannot be tied to the code behind it is asking to be taken on trust, which
is the one thing this tool exists to stop doing.

Standard library only, following `qv/types.py`: the numeric core has to stay
testable in isolation, and identifying the source has no business dragging a
dependency in.

**Why a source digest and not a dirty flag.** The obvious design is a commit
hash plus a boolean saying whether the working tree was modified. It was cut,
because the boolean cannot be earned. Deriving it from file modification times
against `.git/index` reads a staged-but-uncommitted change as *clean* - a false
clean, the dangerous direction - and the only sound alternative puts a `git`
subprocess inside report generation. A digest over the package source needs
neither: it names the code that actually ran, and `qv --version` in a clean
checkout of the named commit prints the reference digest to compare against, so
the question "was this built from that commit?" is answerable rather than
asserted. It also avoids an off-by-one the boolean would have carried, since a
regenerated artifact necessarily names the commit *before* the one that ships
it, while the digest describes the source that really executed.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

__all__ = ["package_version", "git_commit", "source_digest", "source_identity"]

#: Length of the abbreviated commit hash. Seven is what git itself shows and
#: what a reader will recognise from a log line.
_SHORT_SHA = 7

#: Length of the abbreviated source digest. Twelve hex characters is 48 bits -
#: far more than enough to distinguish the handful of builds anyone compares,
#: and short enough to sit in a footer without dominating it.
_SHORT_DIGEST = 12


def _package_root() -> Path:
    """The `qv/` directory, resolved from this file rather than the cwd.

    Same anchor, and the same reason, as `qv.cli._repo_root`: the console
    script does not put the working directory on `sys.path`, so anything
    located relative to the cwd is wrong whenever the tool is doing its job -
    running inside the researcher's project rather than inside this clone.
    """
    return Path(__file__).resolve().parent


def package_version() -> str:
    """The installed distribution version, or `"unknown"`.

    Falls back rather than raising: a report rendered from a checkout that was
    never installed is unusual but not an error, and failing an audit over a
    footer string would be absurd.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("proper-validation")
    except PackageNotFoundError:
        return "unknown"


def _read_text(path: Path) -> str | None:
    """Read a file, or return `None` if it cannot be read.

    One helper rather than a `try` at each of the three places this module
    touches the filesystem, because they all want the same thing: a git
    directory that is missing, truncated or unreadable costs you the commit
    hash in the footer and nothing else. Failing an audit over an identifier
    would be absurd.
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _git_dir(root: Path) -> Path | None:
    """Locate the git directory for `root`, following a worktree pointer.

    In a linked worktree `.git` is a file holding `gitdir: <path>` rather than
    a directory, and the path it names may be relative to the worktree.
    """
    marker = root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        text = _read_text(marker)
        if text is None or not text.startswith("gitdir:"):
            return None
        pointed = Path(text[len("gitdir:"):].strip())
        if not pointed.is_absolute():
            pointed = (root / pointed).resolve()
        return pointed if pointed.is_dir() else None
    return None


def _resolve_ref(git_dir: Path, ref: str) -> str | None:
    """Read a ref, from a loose file first and then from `packed-refs`.

    A freshly cloned repository has no loose refs at all - they are all packed
    - so reading only `refs/heads/<branch>` would return nothing on exactly the
    checkout a new user has.
    """
    loose = git_dir / ref
    if loose.is_file():
        return _read_text(loose) or None

    packed = _read_text(git_dir / "packed-refs")
    if packed is None:
        return None
    for line in packed.splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        sha, _, name = line.partition(" ")
        if name.strip() == ref:
            return sha.strip()
    return None


def git_commit(root: Path | None = None) -> str | None:
    """The short commit hash of the checkout this package lives in.

    Pure file reads - no subprocess, so this works with no `git` binary on
    PATH and cannot hang an audit. Returns `None` when there is no checkout to
    read, which is the normal case for a wheel install and not an error: the
    caller reports what it has rather than inventing an identifier.
    """
    root = _package_root().parent if root is None else Path(root)
    git_dir = _git_dir(root)
    if git_dir is None:
        return None

    head = _read_text(git_dir / "HEAD")
    if not head:
        return None

    sha = _resolve_ref(git_dir, head[4:].strip()) if head.startswith("ref:") else head
    if sha is None:
        return None
    sha = sha.strip()
    # A detached HEAD holds the hash directly; anything else is not one.
    if len(sha) < _SHORT_SHA or not all(c in "0123456789abcdef" for c in sha.lower()):
        return None
    return sha[:_SHORT_SHA]


def source_digest(root: Path | None = None) -> str:
    """A digest over every `.py` file in the package.

    Hashes sorted `(relative path, file digest)` pairs, so adding, removing or
    renaming a module changes the result as surely as editing one does.

    Bytes are hashed, not decoded text, which means a CRLF checkout digests
    differently from an LF one. That is the honest answer - it genuinely is a
    different file - and this value identifies what ran rather than claiming to
    be portable across checkouts.
    """
    root = _package_root() if root is None else Path(root)
    outer = sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            body = path.read_bytes()
        except OSError:  # pragma: no cover - unreadable source is not worth a crash
            continue
        outer.update(path.relative_to(root).as_posix().encode("utf-8"))
        outer.update(b"\0")
        outer.update(sha256(body).digest())
    return outer.hexdigest()[:_SHORT_DIGEST]


def source_identity() -> dict[str, str | None]:
    """Version, commit and source digest, for the report footer.

    Each part degrades on its own: a checkout that was never installed still
    reports a commit, and a wheel install with no `.git` still reports a
    version and a digest.
    """
    return {
        "version": package_version(),
        "commit": git_commit(),
        "source_digest": source_digest(),
    }
