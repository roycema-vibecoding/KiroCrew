"""Direct unit tests for the backend-only frontend-skip decision.

``frontend_skip`` is the pure, stdlib-only helper the sync runner consults at
runtime to decide whether a backend-only Pull+Build may skip the frontend BUILD
(``npm ci`` is deliberately not a candidate), plus the record that makes such a
skip safe. These tests pin the decision: it fires only when a record derived
after a COMPLETED earlier build names the very ``website/`` tree this sync is
landing, that build's bundle is still staged, and ``website/`` is clean on disk
-- and it is conservative everywhere else.

The record is passed to :func:`may_skip_frontend` as two VALUES, and this module
has no function that reads one off disk. That is the invariant the signatures
enforce: a record stored anywhere a sync step can write is a record a sync step
can forge, and the steps that run before the gated build execute code from the
revision being landed at the same uid as the process that reads it. So the
record lives in the Dev Fleet backend's memory and reaches the runner inside its
own script text -- see ``runtime._FRONTEND_BUILD_RECORDS``.

They run against a REAL git repository rather than a stubbed ``subprocess.run``.
Every falsifier this module exists to close is a statement about git's actual
behaviour -- that ``merge --ff-only`` succeeds over a dirty ``website/``, that
two revisions with identical ``website/`` trees share a tree OID, that
``--untracked-files=all`` is what makes a new file visible -- and a fake that
replays canned output cannot fail when one of those is misunderstood. Only the
"git is unavailable" paths use a stub, because that condition cannot be
provoked from a working repository.

The module is loaded BY FILE PATH, exactly as the sync runner loads its
snapshot, rather than through ``kiro_crew.apps.builtins.dev_fleet`` -- the
dotted import would execute the package ``__init__`` chain (which pulls in
croniter and the rest of the runtime), and the module imports nothing but the
standard library, so it needs no package context. Loading by path is also what
makes these runnable without the project's runtime dependencies installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

_HELPER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "kiro_crew"
    / "apps"
    / "builtins"
    / "dev_fleet"
    / "frontend_skip.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("frontend_skip", _HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fs = _load()

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(
    GIT is None, reason="git is required to exercise the real predicates"
)

#: A vite-shaped index.html: it names the content-hashed bundle chunk, which is
#: what makes two builds of different website/ sources produce different bytes.
BUNDLE = b'<!doctype html><script src="/assets/index-aaa111.js"></script>\n'
BUNDLE_CHUNK = "index-aaa111.js"
#: The same page emitted by a DIFFERENT build -- a different chunk hash.
OTHER_BUNDLE = b'<!doctype html><script src="/assets/index-bbb222.js"></script>\n'
OTHER_BUNDLE_CHUNK = "index-bbb222.js"
#: The chunk each index.html above references. A staged bundle is a TREE, and
#: these are the files a digest of index.html alone cannot describe.
_CHUNK_OF = {BUNDLE: BUNDLE_CHUNK, OTHER_BUNDLE: OTHER_BUNDLE_CHUNK}
CHUNK_BODY = b"export const app = 1;\n"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        [str(GIT), "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"
    return proc.stdout.strip()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_bundle(root: Path, bundle: bytes) -> None:
    """A bundle as vite emits one: index.html plus the chunk it references."""
    _write(root / "index.html", bundle)
    _write(root / "assets" / _CHUNK_OF[bundle], CHUNK_BODY)


def _stage(repo: Path, bundle: bytes = BUNDLE) -> None:
    """Put a served bundle at src/kiro_crew/static/dist, as staging does."""
    _write_bundle(repo / "src" / "kiro_crew" / "static" / "dist", bundle)


def _build_output(repo: Path, bundle: bytes = BUNDLE) -> None:
    """Put a build output at website/dist, as ``vite build`` does."""
    _write_bundle(repo / "website" / "dist", bundle)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A committed checkout shaped like this one's frontend half.

    The ignore rules mirror the real ones -- ``website/.gitignore`` covers
    ``node_modules`` and ``dist``, and the root one covers
    ``src/kiro_crew/static/dist`` -- which is what lets a fully BUILT checkout
    still read as clean, the property the cleanliness gate depends on to fire at
    all.
    """
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root.parent, "init", "-q", str(root))
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "commit.gpgsign", "false")
    _write(root / ".gitignore", b"src/kiro_crew/static/dist/\n")
    _write(root / "website" / ".gitignore", b"node_modules/\ndist/\n")
    _write(root / "website" / "package-lock.json", b'{"name":"website","lockfileVersion":3}\n')
    _write(root / "website" / "src" / "App.tsx", b"export const App = () => null;\n")
    _write(root / "src" / "kiro_crew" / "server.py", b"# backend\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    return root


def _built_and_recorded(repo: Path) -> tuple[str, str]:
    """Model a frontend half that ran to completion: build output, staged, recorded.

    Returns the ``(website tree OID, staged dist digest)`` pair the Dev Fleet
    backend would hold in memory afterwards. Nothing is written anywhere.
    """
    _build_output(repo)
    _stage(repo)
    record = fs.build_record(str(GIT), str(repo))
    assert record is not None
    return record


def _backend_only_commit(repo: Path) -> None:
    """Land a commit touching nothing under website/, as a backend-only sync does.

    Committed rather than merged from a second ref, because the decision asks
    about HEAD: after ``merge --ff-only`` HEAD is the revision the sync landed,
    so a commit on HEAD is that state.
    """
    _write(repo / "src" / "kiro_crew" / "server.py", b"# backend, changed\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "backend only")


def _frontend_commit(repo: Path) -> None:
    """Land a commit that DOES change website/."""
    _write(repo / "website" / "src" / "App.tsx", b"export const App = () => 'new';\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "frontend change")


# --- the skip fires, and on what ---


def test_skips_a_backend_only_sync_after_a_recorded_build(repo):
    """The whole point: a sync landing the same website/ tree skips the build."""
    record = _built_and_recorded(repo)
    _backend_only_commit(repo)

    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is True


def test_a_built_checkout_still_reads_as_clean(repo):
    """The cleanliness gate must not make the skip unreachable.

    ``website/.gitignore`` covers ``node_modules`` and ``dist``, so the artifacts
    a completed frontend half leaves behind are invisible to
    ``status --porcelain --untracked-files=all``. If that were not so, the gate
    would never pass on any real checkout and the optimization would be a silent
    no-op -- the failure mode that is easy to ship and impossible to notice.
    """
    _build_output(repo)
    _write(repo / "website" / "node_modules" / ".package-lock.json", b"{}\n")
    _write(repo / "website" / "node_modules" / "react" / "index.js", b"//\n")

    assert fs.website_worktree_is_clean(str(GIT), str(repo)) is True


def test_deriving_the_record_writes_nothing_at_all(repo):
    """The record is a return value, so it leaves no trace anywhere on disk.

    Two properties in one, and the module has no way to violate either: a record
    written inside the checkout would show up as an untracked file and make the
    very cleanliness gate it feeds fail forever after the first build, and a record
    written ANYWHERE is one a same-uid sync step can rewrite to steer the verdict.
    Pinned by walking the whole checkout before and after, because the second
    property is invisible in a test that only checks the paths it thought of.
    """
    _build_output(repo)
    _stage(repo)

    def _snapshot() -> dict[str, bytes]:
        return {
            str(p.relative_to(repo)): p.read_bytes() for p in sorted(repo.rglob("*")) if p.is_file()
        }

    before = _snapshot()
    record = fs.build_record(str(GIT), str(repo))

    assert record is not None
    assert _snapshot() == before, "build_record must not create or modify any file"
    assert fs.website_worktree_is_clean(str(GIT), str(repo)) is True
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""


# --- a website/ change is a different tree ---


def test_does_not_skip_when_the_landed_revision_changes_website(repo):
    record = _built_and_recorded(repo)
    _frontend_commit(repo)

    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is False


def test_does_not_skip_when_head_is_ahead_of_the_ref_the_sync_merges(repo):
    """The falsifier that makes asking about the INCOMING REF fail open.

    ``git merge --ff-only <ref>`` also exits ZERO -- printing "Already up to
    date" -- when the ref is an ancestor of HEAD, which is what a checkout
    carrying local commits looks like. A ``website/`` change committed there is
    invisible to the ref, whose tree is still the one the recorded build read,
    while HEAD (the source the build reads) has moved. Asking about the ref would
    skip, never build the change, and report success -- and stickily, because
    every later sync repeats it while the ref's tree stays put.

    So the decision asks about HEAD. Both revisions are checked here, to pin that
    the two answers really do differ in this shape rather than the test passing
    for some other reason.
    """
    record = _built_and_recorded(repo)
    _git(repo, "update-ref", "refs/kirocrew/sync-base-test", "HEAD")
    _frontend_commit(repo)

    lagging = fs.website_tree_oid(str(GIT), str(repo), "refs/kirocrew/sync-base-test")
    assert lagging == record[0], "the lagging ref cannot see the committed change"
    assert fs.website_tree_oid(str(GIT), str(repo), "HEAD") != record[0]
    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is False


def test_website_tree_oid_is_identity_not_a_delta(repo):
    """Two revisions share the OID exactly when their website/ trees are equal.

    That is what lets the decision be an identity check on the ONE revision that
    matters rather than a diff against a base: a backend-only commit leaves the
    OID alone and a frontend one moves it, with no second revision to plumb
    through. A diff needs that base, and inside a runner that has already
    fast-forwarded the only base at hand is HEAD itself, which makes the diff
    vacuously empty on every successful sync.
    """
    before = fs.website_tree_oid(str(GIT), str(repo), "HEAD")
    _backend_only_commit(repo)
    after_backend = fs.website_tree_oid(str(GIT), str(repo), "HEAD")
    _frontend_commit(repo)
    after_frontend = fs.website_tree_oid(str(GIT), str(repo), "HEAD")

    assert before is not None
    assert after_backend == before, "a backend-only commit must not move the website tree"
    assert after_frontend != before


# --- provenance: a failed build never records, so its retry rebuilds ---


def test_does_not_skip_when_the_frontend_half_never_completed(repo):
    """No record at all -- a cold backend, or a checkout it has never synced.

    This is the cold-start path, and it is reached on the FIRST backend-only sync
    of every Dev Fleet backend lifetime now that the record is held in memory. It
    must read as "build", never as a vacuous match: an empty pair is what the
    caller has when it holds nothing, so the verdict has to refuse it explicitly
    rather than compare two empty strings and agree.
    """
    _stage(repo)
    _build_output(repo)
    _backend_only_commit(repo)

    assert fs.may_skip_frontend(str(GIT), str(repo), "", "") is False


def test_does_not_skip_when_a_tsc_failure_left_the_build_output_intact(repo):
    """The falsifier no artifact comparison can see.

    ``npm run build`` is ``tsc -b && vite build`` and ``emptyOutDir`` lives
    INSIDE vite, so a TypeScript error in freshly merged code means vite never
    runs and ``website/dist`` is left fully intact -- byte-identical to the
    ``static/dist`` copy it was staged from. The merge has landed by then, so a
    diff-based gate is empty too, and the retry would skip and report success
    having never built the merged frontend.

    Modelled exactly: a good build is recorded, a frontend change lands, the
    build fails (so nothing is recorded and BOTH dist trees still hold the old
    bundle), and the retry must still build.
    """
    record = _built_and_recorded(repo)
    _frontend_commit(repo)

    # The failed build wrote nothing: both trees are still the old bundle, so
    # every artifact-to-artifact comparison reads "provenance holds".
    assert (repo / "website" / "dist" / "index.html").read_bytes() == BUNDLE
    assert (repo / "src" / "kiro_crew" / "static" / "dist" / "index.html").read_bytes() == BUNDLE

    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is False


def test_does_not_skip_when_the_stage_failed_after_a_good_build(repo):
    """Build ok, stage failed: the step exited non-zero, so nothing was recorded.

    The served bundle is then older than the source, and the retry -- whose own
    merge has already landed -- is precisely the shape a delta-based gate would
    skip.
    """
    record = _built_and_recorded(repo)
    _frontend_commit(repo)
    # The build succeeded and emitted a new bundle; the stage never copied it.
    _build_output(repo, OTHER_BUNDLE)

    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is False


def test_build_record_declines_when_it_cannot_vouch_for_the_build(repo):
    """``None`` when the pair cannot honestly be formed, so the caller drops it.

    A build this function cannot describe (a dirty ``website/`` -- the build read a
    working tree no commit names) must not leave the caller's older pair standing as
    if it described the bundle now on disk. Answering ``None`` is what makes the
    caller REPLACE rather than keep, which is the in-memory equivalent of removing a
    stale record.
    """
    record = _built_and_recorded(repo)
    assert record is not None

    _write(repo / "website" / "src" / "App.tsx", b"export const App = () => 'dirty';\n")

    assert fs.build_record(str(GIT), str(repo)) is None


def test_build_record_declines_when_no_bundle_was_staged(repo):
    """Nothing to fingerprint means nothing to vouch for."""
    _build_output(repo)

    assert fs.build_record(str(GIT), str(repo)) is None


# --- the working tree, which a tree OID cannot describe ---


def test_does_not_skip_over_an_uncommitted_website_edit(repo):
    """`merge --ff-only` succeeds over a dirty website/ it does not touch.

    So the committed tree can match the stamp while the source a build would
    read does not: an uncommitted edit would be silently never built, with the
    sync reporting success.
    """
    record = _built_and_recorded(repo)
    _backend_only_commit(repo)
    _write(repo / "website" / "src" / "App.tsx", b"export const App = () => 'edited';\n")

    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is False


def test_does_not_skip_over_an_untracked_website_file(repo):
    """A NEW file is exactly as invisible to a tree OID as an edited one.

    This is what ``--untracked-files=all`` is for; without it a new component
    added beside the tracked ones is never built.
    """
    record = _built_and_recorded(repo)
    _backend_only_commit(repo)
    _write(repo / "website" / "src" / "NewThing.tsx", b"export const N = () => null;\n")

    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is False


# --- the served bundle ---


def test_does_not_skip_when_the_served_bundle_is_gone(repo):
    """static/dist and node_modules are independent artifacts.

    Running the build+stage step is what repairs an absent dist, so its absence
    must withhold the skip or the dashboard is left with no assets.
    """
    record = _built_and_recorded(repo)
    _backend_only_commit(repo)
    shutil.rmtree(repo / "src" / "kiro_crew" / "static" / "dist")

    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is False
    assert fs.staged_dist_digest(str(repo)) is None


def test_does_not_skip_when_the_bundle_was_restaged_out_of_band(repo):
    """A peer flow's own build must not be inherited by the skip.

    ``_stage_dist`` has callers other than the sync (the dashboard's own update,
    pod provisioning), so the bundle on disk can stop being the one the recorded
    build staged. The stamp fingerprints it for exactly this reason.
    """
    record = _built_and_recorded(repo)
    _backend_only_commit(repo)
    _stage(repo, OTHER_BUNDLE)

    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is False


def test_staged_dist_digest_follows_the_source_tree_symlink(repo):
    """On a source-tree install static/dist is a symlink to website/dist.

    Both layouts must fingerprint, and to the SAME value for the same tree, so
    the skip is not silently unreachable on the very install Dev Fleet exists to
    manage -- and so a layout change alone cannot withhold it.
    """
    static = repo / "src" / "kiro_crew" / "static" / "dist"
    _stage(repo)
    real_directory = fs.staged_dist_digest(str(repo))
    assert real_directory is not None

    _build_output(repo)
    shutil.rmtree(static)
    try:
        static.symlink_to(repo / "website" / "dist", target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - unprivileged Windows
        pytest.skip("this platform does not allow creating a directory symlink")

    assert fs.staged_dist_digest(str(repo)) == real_directory


def test_does_not_skip_when_a_killed_build_emptied_the_bundle_around_index(repo):
    """A chunk lost since the recorded build must withhold the skip.

    ``emptyOutDir`` deletes the previous bundle in directory order before vite
    writes the new one, so a build+stage step killed mid-empty -- the run
    watchdog's timeout kill, or gateway shutdown reaping the process tree -- can
    leave ``index.html`` byte-identical to the recorded one while the chunks it
    references are already gone. That step exits non-zero, so it never records a
    new stamp: the OLD stamp stands, and on a backend-only retry the tree OID and
    cleanliness gates both still pass. A digest of ``index.html`` alone reads that
    as "provenance holds", skips, and serves a shell whose every asset 404s.
    """
    record = _built_and_recorded(repo)
    _backend_only_commit(repo)
    static = repo / "src" / "kiro_crew" / "static" / "dist"
    (static / "assets" / BUNDLE_CHUNK).unlink()

    # The marker file a shallower check would have consulted is untouched, so
    # every artifact comparison reaching only it would say the bundle is fresh.
    assert (static / "index.html").read_bytes() == BUNDLE
    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is False


def test_the_staged_digest_covers_paths_and_bytes_not_just_index(repo):
    """The fingerprint moves on any file of the tree, added or edited.

    Vite copies ``website/public/`` verbatim under stable UNHASHED names -- the
    importmap's ``/vendor/*.mjs`` shims, icons, ``manifest.json``, ``sw.js`` --
    so those files never appear in a content-hashed reference and a digest of
    ``index.html`` describes none of them.
    """
    _stage(repo)
    static = repo / "src" / "kiro_crew" / "static" / "dist"
    staged = fs.staged_dist_digest(str(repo))

    _write(static / "vendor" / "react.mjs", b"export default 1;\n")
    added = fs.staged_dist_digest(str(repo))
    assert added is not None and added != staged

    _write(static / "vendor" / "react.mjs", b"export default 2;\n")
    edited = fs.staged_dist_digest(str(repo))
    assert edited is not None and edited != added

    assert (static / "index.html").read_bytes() == BUNDLE


def _separator_framed(entries: list[tuple[str, bytes]]) -> str:
    """The digest a SEPARATED framing (``path SEP body SEP``) would produce.

    Present so the test below states a property of the framing rather than of
    the bytes it happens to pick: it first proves the two trees are
    indistinguishable to a separated walk, and only then requires the shipped
    one to tell them apart.
    """
    d = hashlib.sha256()
    for name, body in entries:
        d.update(name.encode("utf-8"))
        d.update(b"\0")
        d.update(body)
        d.update(b"\0")
    return d.hexdigest()


@pytest.mark.parametrize(
    "pair,other",
    [
        ([("index.html", b"X"), ("z.js", b"Y")], [("index.html", b"X\0z.js\0Y")]),
        ([("index.html", b""), ("z.js", b"")], [("index.html", b"\0z.js\0")]),
    ],
    ids=["asset-merged-into-index", "empty-asset-merged-into-index"],
)
def test_the_staged_digest_cannot_be_framed_into_another_trees_value(repo, pair, other):
    """Two DIFFERENT bundles must not share a digest, whatever the bytes spell.

    A separator only delimits values that cannot contain it, and a file body can
    contain anything -- so under ``path SEP body SEP`` a single file whose bytes
    spell out the framing of two collides with the pair. Both cases here are that
    collision, the second with the merged-away asset EMPTY, which is what makes
    "an empty file" and "no file at all" the same tree to a separated walk.

    A collision is the one error this digest must not make: it reads as "the
    recorded build's bundle is still staged" while an asset is gone, so the build
    is skipped and the missing asset 404s under a sync that reported success.
    """
    static = repo / "src" / "kiro_crew" / "static" / "dist"
    assert _separator_framed(pair) == _separator_framed(other)

    def _digest_of(entries: list[tuple[str, bytes]]) -> str | None:
        shutil.rmtree(static, ignore_errors=True)
        for name, body in entries:
            _write(static / name, body)
        return fs.staged_dist_digest(str(repo))

    first = _digest_of(pair)
    second = _digest_of(other)
    assert first is not None and second is not None
    assert first != second


def test_the_staged_digest_separates_a_symlinked_asset_from_a_real_one(repo):
    """A link to identical bytes is not the file the recorded build staged.

    The digest frames each entry's KIND, so swapping one of the bundle's own
    files for a link that resolves to the same bytes moves the digest instead of
    being read as the same tree. Whatever the link resolves to today is not
    covered by the walk and can be repointed without touching the bundle.
    """
    static = repo / "src" / "kiro_crew" / "static" / "dist"
    _stage(repo)
    real = fs.staged_dist_digest(str(repo))
    assert real is not None

    chunk = static / "assets" / BUNDLE_CHUNK
    elsewhere = repo / "elsewhere.js"
    _write(elsewhere, CHUNK_BODY)
    chunk.unlink()
    try:
        chunk.symlink_to(elsewhere)
    except (OSError, NotImplementedError):  # pragma: no cover - unprivileged Windows
        pytest.skip("this platform does not allow creating a file symlink")

    assert chunk.read_bytes() == CHUNK_BODY
    assert fs.staged_dist_digest(str(repo)) != real


def test_the_staged_digest_sees_a_directory_the_build_never_staged(repo):
    """The digest is a function of the tree's SHAPE, not only of its file bytes.

    An added directory carries no served bytes of its own, so a file-only walk
    reads the tree as unchanged -- and then cannot see a directory replaced by a
    symlink into a tree of someone else's choosing either, since the walk does
    not descend a link. Framing directory entries is what makes both visible; the
    cost is the conservative direction, an unnecessary build.
    """
    static = repo / "src" / "kiro_crew" / "static" / "dist"
    _stage(repo)
    staged = fs.staged_dist_digest(str(repo))
    assert staged is not None

    (static / "vendor").mkdir()
    assert fs.staged_dist_digest(str(repo)) != staged


# --- npm ci is deliberately NOT part of the verdict ---


def test_the_verdict_ignores_node_modules_because_npm_ci_is_unconditional(repo):
    """A damaged node_modules does not withhold the BUILD skip -- by design.

    Whether the on-disk tree is what ``npm ci`` would produce cannot be verified
    below the cost of running it: npm's ``node_modules/.package-lock.json`` is
    metadata nothing reconciles with the files it describes, so it stays
    byte-identical while a package is deleted or a file inside one is truncated.
    So ``npm ci`` is left unconditional and REPAIRS the tree on every sync,
    which is what makes it safe for this verdict to say nothing about it.
    """
    record = _built_and_recorded(repo)
    _backend_only_commit(repo)
    # A tree npm's own metadata still describes as complete, with the package
    # itself removed.
    _write(repo / "website" / "node_modules" / ".package-lock.json", b'{"packages":{}}\n')

    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is True


# --- the record is unreachable from disk, and unobtainable evidence ---


def test_no_record_planted_on_disk_can_reach_the_verdict(repo):
    """The class both blocking findings named, closed structurally.

    An install script running before the gated build -- ``pip install -e .``
    executing the merged revision's build backend, ``npm ci`` running every
    dependency's lifecycle scripts -- is same-uid code from the revision being
    landed, and both fields of the record are computable by it. So it plants a
    plausible record wherever a record could live, having ALSO reverted the served
    bundle so the live digest matches. Every one of those files is inert: the
    verdict is a function of its two arguments and of git, so there is no path a
    planted file could travel.

    Pinned as a regression fence rather than as a behaviour: the way this breaks is
    someone re-adding a parameter that names a file, at which point the forged
    record below starts being read.
    """
    record = _built_and_recorded(repo)
    _backend_only_commit(repo)
    stale_tree, stale_dist = record

    forged = json.dumps({"website_tree": stale_tree, "dist_tree": stale_dist})
    plausible = [
        repo / ".git" / "kirocrew-frontend-skip" / "build-stamp.json",
        repo / ".git" / "kirocrew-frontend-build-stamp.json",
        repo / ".git" / "build-stamp.json",
    ]
    for path in plausible:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(forged, encoding="utf-8")

    # The honest verdict on this state is True (a backend-only commit over a
    # recorded build), so the forged files cannot be what produced it...
    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is True
    # ...and with the pair the caller actually holds replaced by an empty one, the
    # verdict is False even though every forged record is still on disk. That is
    # the module having no way to read one.
    assert fs.may_skip_frontend(str(GIT), str(repo), "", "") is False
    assert all(p.is_file() for p in plausible), "the forged records are still there"
    # No public name of this module takes a record PATH, so none can be handed one.
    assert not [name for name in dir(fs) if not name.startswith("_") and "stamp" in name.lower()]


def test_does_not_skip_when_git_cannot_be_run(repo, monkeypatch):
    """Unobtainable evidence is weak evidence: build."""
    record = _built_and_recorded(repo)
    _backend_only_commit(repo)

    def _boom(*a, **k):
        raise OSError("no git here")

    monkeypatch.setattr(fs.subprocess, "run", _boom)

    assert fs.may_skip_frontend(str(GIT), str(repo), *record) is False
    assert fs.website_tree_oid(str(GIT), str(repo), "HEAD") is None
    assert fs.website_worktree_is_clean(str(GIT), str(repo)) is False


def test_does_not_skip_when_head_identifies_no_website_tree(tmp_path):
    """An unborn HEAD identifies no tree, so nothing the record names can match.

    The record here is well-formed and its bundle digest matches what is staged,
    so the only reason left to refuse is the one under test.
    """
    empty = tmp_path / "unborn"
    empty.mkdir()
    _git(empty.parent, "init", "-q", str(empty))
    _stage(empty)
    staged = fs.staged_dist_digest(str(empty))

    assert staged is not None
    assert fs.website_tree_oid(str(GIT), str(empty), "HEAD") is None
    assert fs.may_skip_frontend(str(GIT), str(empty), "0" * 40, staged) is False


def test_website_tree_oid_rejects_a_non_oid_answer(repo, monkeypatch):
    """Only a hex digest is an answer to "which tree?".

    Anything else must not be allowed to compare equal to another non-answer.
    """

    class _Proc:
        returncode = 0
        stdout = b"fatal: something\n"

    monkeypatch.setattr(fs.subprocess, "run", lambda *a, **k: _Proc())

    assert fs.website_tree_oid(str(GIT), str(repo), "HEAD") is None
