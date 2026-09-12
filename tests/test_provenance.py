"""Which engine produced a report, and how that is established.

Every helper here is pointed at a temporary tree rather than at this
repository's own checkout. Asserting against the live `.git` would make the
suite pass or fail on whichever branch happened to be out, which is exactly
the kind of unreproducible test a validator has no business shipping.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qv.provenance import git_commit, package_version, source_digest, source_identity

SHA = "ab68694f0e0d1c2b3a4958677889aabbccddeeff"


def _package(root: Path, body: str = "x = 1\n") -> Path:
    """A minimal package tree for the digest to walk."""
    pkg = root / "qv"
    (pkg / "stats").mkdir(parents=True)
    (pkg / "__init__.py").write_text(body, encoding="utf-8")
    (pkg / "stats" / "sharpe.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return pkg


class TestSourceDigest:
    def test_is_stable_across_calls(self, tmp_path):
        pkg = _package(tmp_path)
        assert source_digest(pkg) == source_digest(pkg)

    def test_changes_when_one_byte_of_source_changes(self, tmp_path):
        """The negative control.

        Without it a helper that returned a constant would satisfy every other
        test in this class, and the footer would carry an identifier that
        identified nothing.
        """
        pkg = _package(tmp_path, body="x = 1\n")
        before = source_digest(pkg)
        (pkg / "__init__.py").write_text("x = 2\n", encoding="utf-8")
        assert source_digest(pkg) != before

    def test_changes_when_a_module_is_added_or_removed(self, tmp_path):
        pkg = _package(tmp_path)
        before = source_digest(pkg)
        added = pkg / "stats" / "pbo.py"
        added.write_text("y = 2\n", encoding="utf-8")
        assert source_digest(pkg) != before
        added.unlink()
        assert source_digest(pkg) == before

    def test_changes_when_a_module_is_renamed(self, tmp_path):
        """Path as well as content, or a rename would digest identically."""
        pkg = _package(tmp_path)
        before = source_digest(pkg)
        (pkg / "stats" / "sharpe.py").rename(pkg / "stats" / "sortino.py")
        assert source_digest(pkg) != before

    def test_ignores_compiled_bytecode(self, tmp_path):
        """`__pycache__` is a build artifact, not the source that ran."""
        pkg = _package(tmp_path)
        before = source_digest(pkg)
        cache = pkg / "__pycache__"
        cache.mkdir()
        (cache / "__init__.cpython-312.pyc").write_bytes(b"\x00\x01")
        (cache / "shim.py").write_text("z = 3\n", encoding="utf-8")
        assert source_digest(pkg) == before

    def test_is_short_and_hexadecimal(self, tmp_path):
        digest = source_digest(_package(tmp_path))
        assert len(digest) == 12
        assert all(c in "0123456789abcdef" for c in digest)


class TestGitCommit:
    def test_reads_a_loose_branch_ref(self, tmp_path):
        git = tmp_path / ".git"
        (git / "refs" / "heads").mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git / "refs" / "heads" / "main").write_text(SHA + "\n", encoding="utf-8")
        assert git_commit(tmp_path) == SHA[:7]

    def test_reads_a_detached_head(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text(SHA + "\n", encoding="utf-8")
        assert git_commit(tmp_path) == SHA[:7]

    def test_falls_back_to_packed_refs(self, tmp_path):
        """A fresh clone has no loose refs at all - they are all packed.

        Reading only `refs/heads/<branch>` would therefore return nothing on
        exactly the checkout a new user has.
        """
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled sorted\n"
            f"{SHA} refs/heads/main\n"
            "0000000000000000000000000000000000000000 refs/tags/v0.1.0\n",
            encoding="utf-8",
        )
        assert git_commit(tmp_path) == SHA[:7]

    def test_follows_a_worktree_pointer(self, tmp_path):
        """In a linked worktree `.git` is a file holding `gitdir: <path>`."""
        real = tmp_path / "elsewhere" / "worktrees" / "audit"
        real.mkdir(parents=True)
        (real / "HEAD").write_text(SHA + "\n", encoding="utf-8")
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / ".git").write_text(f"gitdir: {real}\n", encoding="utf-8")
        assert git_commit(tree) == SHA[:7]

    def test_returns_none_without_a_checkout(self, tmp_path):
        """The normal case for a wheel install, and not an error."""
        assert git_commit(tmp_path) is None

    def test_returns_none_when_the_ref_is_unresolvable(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/never-created\n", encoding="utf-8")
        assert git_commit(tmp_path) is None

    def test_returns_none_rather_than_echoing_a_non_hash(self, tmp_path):
        """A HEAD that is neither a ref nor a hash is not an identifier."""
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("not a commit at all\n", encoding="utf-8")
        assert git_commit(tmp_path) is None

    def test_returns_none_when_head_is_missing(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert git_commit(tmp_path) is None

    def test_returns_none_when_packed_refs_lacks_the_ref(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git / "packed-refs").write_text(
            f"{SHA} refs/heads/other\n", encoding="utf-8"
        )
        assert git_commit(tmp_path) is None

    def test_a_relative_worktree_pointer_resolves_against_the_tree(self, tmp_path):
        """`git worktree` writes a relative gitdir in some layouts."""
        tree = tmp_path / "tree"
        real = tmp_path / "tree" / "nested" / "gitdir"
        real.mkdir(parents=True)
        (real / "HEAD").write_text(SHA + "\n", encoding="utf-8")
        (tree / ".git").write_text("gitdir: nested/gitdir\n", encoding="utf-8")
        assert git_commit(tree) == SHA[:7]

    def test_a_git_file_that_is_not_a_pointer_is_ignored(self, tmp_path):
        (tmp_path / ".git").write_text("something else entirely\n", encoding="utf-8")
        assert git_commit(tmp_path) is None

    def test_an_unreadable_git_directory_costs_only_the_hash(self, tmp_path):
        """A directory where HEAD should be: readable as a path, not as a file.

        The audit must survive it. Losing the commit from the footer is a
        stated gap; crashing an audit over an identifier would not be.
        """
        git = tmp_path / ".git"
        (git / "HEAD").mkdir(parents=True)
        assert git_commit(tmp_path) is None


class TestIdentity:
    def test_version_is_reported_and_never_raises(self):
        assert isinstance(package_version(), str)
        assert package_version()

    def test_version_falls_back_when_the_distribution_is_absent(self, monkeypatch):
        """A checkout that was never installed still has to render a report."""
        import importlib.metadata as meta

        def missing(_name):
            raise meta.PackageNotFoundError("proper-validation")

        monkeypatch.setattr(meta, "version", missing)
        assert package_version() == "unknown"

    def test_identity_carries_all_three_parts(self):
        ident = source_identity()
        assert set(ident) == {"version", "commit", "source_digest"}
        assert ident["source_digest"]
        assert ident["version"]

    def test_parts_degrade_independently(self, monkeypatch):
        """A missing `.git` must not cost the version or the digest.

        The three are recorded together, but a wheel install has no commit to
        read and a checkout that was never installed has no version. Either
        one going missing has to leave the others standing.
        """
        import qv.provenance as prov

        monkeypatch.setattr(prov, "git_commit", lambda root=None: None)
        ident = prov.source_identity()
        assert ident["commit"] is None
        assert ident["version"] and ident["source_digest"]

    def test_the_package_exports_a_version(self):
        import qv

        assert qv.__version__ == package_version()


class TestLineEndingsAreNormalised:
    """Every source file must be LF, because the digest hashes bytes.

    `source_digest` hashes file bytes deliberately - its docstring says so, and
    the reasoning is sound: a CRLF checkout genuinely is a different file. The
    consequence is that a working tree which drifts to CRLF produces reports
    whose footer digest no checkout of any commit can reproduce, which is
    exactly the identification the footer exists to provide.

    This is easy to do by accident on Windows: `Path.write_text` opens in text
    mode, so it translates `\n` to `\r\n`, and any script that rewrites a
    source file flips that file's line endings without saying so. It happened,
    it produced six committed reports nobody else could reproduce, and it was
    caught by CI rather than locally. Hence this test.

    `.gitattributes` sets `* text=auto eol=lf`, so LF is what every checkout
    should hold on every platform.
    """

    def test_no_python_file_has_carriage_returns(self):
        import subprocess

        root = Path(__file__).resolve().parents[1]
        listed = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=root, capture_output=True, text=True, check=False,
        )
        if listed.returncode != 0:  # pragma: no cover - not a git checkout
            pytest.skip("not a git checkout")

        crlf = [
            name for name in listed.stdout.split()
            if b"\r\n" in (root / name).read_bytes()
        ]
        assert not crlf, (
            "these files hold CRLF, which changes the source digest and makes "
            f"every committed report unreproducible: {crlf}"
        )
