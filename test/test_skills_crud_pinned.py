"""Descriptor-pinned skill CRUD in ``SkillsLoader`` (create/update/delete).

``create_skill``/``update_skill``/``delete_skill`` address the skill directory and
its ``SKILL.md`` relative to a parent descriptor pinned by ``pinned_fs``, so an
ancestor swapped for a link after the by-name existence check cannot redirect the
write; ``update_skill`` additionally routes through ``atomic_write`` with the ACL
carry, closing the gap where a plain ``write_text`` dropped a named POSIX ACL.

NOT EXECUTED IN THE INTEGRATIONS_ONLY SANDBOX. Importing ``kiro_crew.skills`` pulls
``kiro_crew.cron`` -> ``croniter`` and ``kiro_crew.vector_memory`` ->
``snowballstemmer``, neither installable offline (pip 403), so these run in CI only.

CI invocation:

    python -m pytest test/test_skills_crud_pinned.py
"""

from __future__ import annotations

import errno
import os

import pytest

from kiro_crew.skills import SkillsLoader


@pytest.fixture()
def loader(tmp_path):
    return SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False)


def test_create_writes_skill_md_byte_exact(loader):
    body = "---\nname: demo\n---\n\n## Steps\ndo it\n"
    assert loader.create_skill("demo", body) is True
    written = (loader._dir / "demo" / "SKILL.md").read_text(encoding="utf-8")
    assert written == body


def test_create_refuses_a_pre_existing_name(loader):
    assert loader.create_skill("dup", "first") is True
    assert loader.create_skill("dup", "second") is False
    # The original content is untouched by the refused second create.
    assert (loader._dir / "dup" / "SKILL.md").read_text(encoding="utf-8") == "first"


def test_update_replaces_content_and_returns_true(loader):
    assert loader.create_skill("edit", "before") is True
    assert loader.update_skill("edit", "after") is True
    assert (loader._dir / "edit" / "SKILL.md").read_text(encoding="utf-8") == "after"


def test_update_refuses_a_missing_skill(loader):
    assert loader.update_skill("nope", "x") is False


def test_update_carries_the_acl(loader, monkeypatch):
    """update_skill routes through the ACL carry, so the source xattrs are read.

    Monkeypatches the xattr syscalls so the assertion holds on any filesystem: the
    captured source ACL value must reach ``setxattr`` on the replacement inode.
    """
    if not all(hasattr(os, a) for a in ("listxattr", "getxattr", "setxattr")):
        pytest.skip("platform without xattr syscalls")
    assert loader.create_skill("acl", "before") is True

    monkeypatch.setattr(os, "listxattr", lambda *a, **k: ["system.posix_acl_access"], raising=False)
    monkeypatch.setattr(os, "getxattr", lambda *a, **k: b"acl-bytes", raising=False)
    recorded: list[tuple[str, bytes]] = []
    monkeypatch.setattr(
        os,
        "setxattr",
        lambda fd, attr, value, *a, **k: recorded.append((attr, value)),
        raising=False,
    )

    assert loader.update_skill("acl", "after") is True
    monkeypatch.undo()
    assert ("system.posix_acl_access", b"acl-bytes") in recorded


def test_update_is_atomic_no_temp_residue(loader):
    assert loader.create_skill("atomic", "before") is True
    assert loader.update_skill("atomic", "after") is True
    names = sorted(p.name for p in (loader._dir / "atomic").iterdir())
    assert names == ["SKILL.md"]


def test_update_refuses_a_skill_md_swapped_to_a_symlink(loader, tmp_path):
    """A SKILL.md that is a symlink must not be written through.

    The pinned open uses ``O_NOFOLLOW`` and the by-name floor's ``atomic_write``
    replaces the link's directory entry rather than following it, so the link's
    target keeps its old bytes either way.
    """
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("O_NOFOLLOW required")
    assert loader.create_skill("linked", "seed") is True
    skill_md = loader._dir / "linked" / "SKILL.md"
    outside = tmp_path / "outside.txt"
    outside.write_text("protected", encoding="utf-8")
    skill_md.unlink()
    skill_md.symlink_to(outside)

    loader.update_skill("linked", "attacker body")
    # Whatever the outcome token, the file the link pointed at is not overwritten.
    assert outside.read_text(encoding="utf-8") == "protected"


@pytest.mark.parametrize("pinned", [True, False])
def test_update_rejects_the_target_when_the_acl_source_open_fails(loader, monkeypatch, pinned):
    """A failed ACL-source open rejects the update; it does not write without it.

    ``open_access_control_source`` documents that its ``OSError`` propagates so
    the caller can treat the leaf as a rejected target, and both sibling update
    surfaces (steering, file-write) answer it with a not-found token. Swallowing
    it here would publish a fresh inode carrying permission BITS only, silently
    dropping a named POSIX ACL -- on the path whose whole purpose is to keep it.
    Pinned and by-name floor both, so neither branch regains the swallow.
    """
    import kiro_crew.skills as skills_mod

    assert loader.create_skill("acl-src", "before") is True
    monkeypatch.setattr(skills_mod, "_DIR_FD_SUPPORTED", pinned)

    def refuse(_path):
        raise OSError(errno.ELOOP, "leaf swapped for a link")

    monkeypatch.setattr(skills_mod, "open_access_control_source", refuse)
    assert loader.update_skill("acl-src", "after") is False
    assert (loader._dir / "acl-src" / "SKILL.md").read_text(encoding="utf-8") == "before"
    # No temp file was staged and left behind by the refused update.
    assert sorted(p.name for p in (loader._dir / "acl-src").iterdir()) == ["SKILL.md"]


def test_delete_removes_the_skill(loader):
    assert loader.create_skill("gone", "x") is True
    assert loader.delete_skill("gone") is True
    assert not (loader._dir / "gone").exists()


def test_delete_refuses_a_symlinked_skill_dir(loader, tmp_path):
    """A skill directory that is a link is refused, not followed into an rmtree."""
    from kiro_crew.platform_compat import symlink_or_junction

    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep", encoding="utf-8")
    loader._dir.mkdir(parents=True, exist_ok=True)
    link = loader._dir / "linkskill"
    symlink_or_junction(str(victim), str(link))

    assert loader.delete_skill("linkskill") is False
    # The link's target and its contents survive.
    assert (victim / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_by_name_floor_when_capability_probe_is_false(loader, monkeypatch):
    """With the probe forced False, create/update/delete use the by-name path.

    Pins that the by-name floor still produces correct results, so the platform
    without openat (Windows) keeps working.
    """
    import kiro_crew.skills as skills_mod

    monkeypatch.setattr(skills_mod, "_DIR_FD_SUPPORTED", False)
    assert loader.create_skill("floor", "one") is True
    assert (loader._dir / "floor" / "SKILL.md").read_text(encoding="utf-8") == "one"
    assert loader.update_skill("floor", "two") is True
    assert (loader._dir / "floor" / "SKILL.md").read_text(encoding="utf-8") == "two"
    assert loader.delete_skill("floor") is True
    assert not (loader._dir / "floor").exists()
