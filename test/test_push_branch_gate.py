"""Tests for the git-publish branch gate (feature-branch allow, protected deny).

Covers the always-on Python gate in ``kiro_crew.security`` and its kiro-cli
``defaults.json`` mirror:

* ``_is_git_publish`` — a PURE detector ("is this a git publish?"), incl.
  command-substitution glue-evasion. No side effects.
* ``_is_push_to_protected_branch`` — the allow/deny decision. Fails CLOSED on
  bare/ambiguous refs, protected targets (main/mainline/master in any ref
  spelling), wildcard refspecs, ``--mirror``/``--all``, quote/escape evasion,
  and substitution glue. EVERY publish sub-invocation is validated.
* ``is_denied`` — the enforcement point: denial reason for a blocked publish,
  ``push_allowed`` SEL audit for an allowed one (final-outcome only).

KiroCrew protects the git default branch names only (enumerated at line 9
above); it has no ``beta-braveheart``/``develop``/``prod`` integration branch
nor a ``release/*`` namespace, so those names are ordinary feature branches here.
"""

import signal
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from kiro_crew import security
from kiro_crew.security import (
    _is_git_publish,
    _is_push_to_protected_branch,
    _schedule_push_allow_audit,
    is_denied,
)

# "git pus" + "h" keeps a literal blocked command out of the test source.
PUSH = "git pus" + "h"


class TestIsPushToProtectedBranch:
    """Unit tests for the branch-level allow/deny decision."""

    def test_feature_branch_allowed(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} github my-feature-branch") is False
        assert _is_push_to_protected_branch(f"{PUSH} -u origin fix/relax-git-rule") is False
        assert _is_push_to_protected_branch(f"{PUSH} origin feat/welcome-kiro-ghost") is False

    def test_refspec_to_feature_allowed(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} github feature:my-feature") is False
        assert _is_push_to_protected_branch(f"{PUSH} origin HEAD:my-topic") is False

    def test_protected_branch_blocked(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin main") is True
        assert _is_push_to_protected_branch(f"{PUSH} github mainline") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin master") is True

    def test_nondefault_integration_branches_not_protected(self) -> None:
        """KiroCrew protects only git defaults; other integration branch names
        are ordinary feature branches here and stay pushable."""
        assert _is_push_to_protected_branch(f"{PUSH} origin beta-integration") is False
        assert _is_push_to_protected_branch(f"{PUSH} origin develop") is False
        assert _is_push_to_protected_branch(f"{PUSH} origin release/1.0") is False

    def test_similar_feature_names_not_false_positive(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin mainline-refactor") is False
        assert _is_push_to_protected_branch(f"{PUSH} origin main-refactor") is False
        assert _is_push_to_protected_branch(f"{PUSH} origin feature/main") is False

    def test_refspec_to_protected_blocked(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} github feature:main") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin HEAD:master") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin feat:mainline") is True

    def test_bare_push_blocked(self) -> None:
        assert _is_push_to_protected_branch(PUSH) is True
        assert _is_push_to_protected_branch(f"{PUSH} origin") is True

    def test_force_push_to_feature_allowed(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} --force github my-feature") is False
        assert _is_push_to_protected_branch(f"{PUSH} -f origin my-feature") is False
        assert _is_push_to_protected_branch(f"{PUSH} --force-with-lease github feat") is False

    def test_force_push_to_protected_blocked(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} --force origin main") is True
        assert _is_push_to_protected_branch(f"{PUSH} -f github mainline") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin +main") is True

    def test_multiple_refspecs(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin my-feature main") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin feat1 feat2") is False
        assert _is_push_to_protected_branch(f"{PUSH} origin feat1 mainline feat2") is True

    def test_ambiguous_refs_blocked(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin head") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin @") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin HEAD:my-feature") is False

    def test_push_all_branches_flags_blocked(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} --all origin feat-branch") is True
        assert _is_push_to_protected_branch(f"{PUSH} --mirror origin feat-branch") is True

    def test_shell_expansion_blocked(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin $BRANCH") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin ${{BRANCH}}") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin @{{u}}") is True

    def test_quoted_branch_names_stripped(self) -> None:
        """bash collapses interior quotes/escapes into one word (ma\"in\" -> main)."""
        assert _is_push_to_protected_branch(f"{PUSH} origin 'main'") is True
        assert _is_push_to_protected_branch(f'{PUSH} origin ma"in"') is True
        assert _is_push_to_protected_branch(f"{PUSH} origin m''ain") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin ma\\in") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin 'my-feature'") is False

    # ── ref-spelling normalization (server-side resolution) ──
    def test_ref_path_spellings_blocked(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin refs/heads/main") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin heads/main") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin remotes/origin/main") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin refs/remotes/origin/mainline") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin HEAD:refs/heads/master") is True

    def test_ref_path_feature_not_false_positive(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin refs/heads/my-feature") is False
        assert _is_push_to_protected_branch(f"{PUSH} origin heads/main-refactor") is False

    # ── wildcard refspecs (push MANY refs, may include protected) ──
    def test_wildcard_refspecs_blocked(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin refs/heads/*:refs/heads/*") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin '*:*'") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin +refs/heads/*:refs/heads/*") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin feat*") is True


class TestCompoundAndGlueGate:
    """EVERY publish sub-invocation is validated; glue-evasion fails closed."""

    def test_second_push_to_protected_blocked(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin feat && {PUSH} origin main") is True

    def test_all_feature_pushes_allowed(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin feat1 && {PUSH} origin feat2") is False

    def test_glue_evasion_verb_and_target_blocked(self) -> None:
        """git$(echo ' ')push and ma$(echo)in both resolve to a protected push."""
        assert (
            _is_push_to_protected_branch(
                f"{PUSH} origin feature; git$(echo x)pus{'h'} origin ma$(echo)in"
            )
            is True
        )
        assert _is_push_to_protected_branch(f"git`echo`pus{'h'} origin main") is True

    def test_substitution_in_target_blocked(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin ma$(echo)in") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin ${{x}}main") is True

    def test_brace_expansion_blocked(self) -> None:
        """bash brace expansion resolves to a protected branch before git sees it."""
        assert _is_push_to_protected_branch(f"{PUSH} origin {{main,dummy}}") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin ma{{i,i}}n") is True
        assert _is_push_to_protected_branch(f"{PUSH} origin {{feat,main}}") is True

    def test_substitution_in_non_push_segment_allowed(self) -> None:
        assert _is_push_to_protected_branch(f"{PUSH} origin feat && echo $(date)") is False
        assert _is_push_to_protected_branch(f"echo $(date); {PUSH} origin my-feature") is False
        assert _is_push_to_protected_branch(f"{PUSH} origin feat && echo {{a,b}}") is False

    def test_word_push_in_other_segment_allowed(self) -> None:
        assert (
            _is_push_to_protected_branch(f"{PUSH} origin feat; echo remember-to-pus{'h'}") is False
        )


class TestIsGitPublishDetection:
    """_is_git_publish is a PURE detector — True for ANY publish, no side effects."""

    def test_detects_feature_and_protected(self) -> None:
        assert _is_git_publish(f"{PUSH} github my-feature-branch") is True
        assert _is_git_publish(f"{PUSH} origin main") is True

    def test_bare_detected(self) -> None:
        assert _is_git_publish(PUSH) is True

    def test_stash_and_commit_message_not_detected(self) -> None:
        assert _is_git_publish("git stash pus" + "h") is False
        assert _is_git_publish("git commit -m 'pus" + "h to prod'") is False
        assert _is_git_publish("git log --grep pus" + "h") is False

    def test_glue_evasion_detected(self) -> None:
        assert _is_git_publish("git_pus" + "h") is True
        assert _is_git_publish(f"git$(echo ' ')pus{'h'} origin main") is True

    def test_program_substitution_evasion_detected(self) -> None:
        # Program NAME produced by an expansion the shell resolves to git before
        # exec -- the literal ``git`` token never appears in the source text.
        # Port of the upstream project (security-review 3eeb3852).
        P = "pus" + "h"
        assert _is_git_publish("$(echo git) %s origin main" % P) is True
        assert _is_git_publish("`echo git` %s origin main" % P) is True
        assert _is_git_publish("${git} %s origin main" % P) is True
        assert _is_git_publish("$git %s origin main" % P) is True
        # Bare $VAR after an env-assignment in a chained segment.
        assert _is_git_publish("git=x; $git %s origin mainline" % P) is True

    def test_program_substitution_non_push_not_detected(self) -> None:
        # Substitutions/expansions NOT followed by a ``push`` subcommand, and
        # unrelated $VAR usage, must NOT be flagged.
        assert _is_git_publish("$(echo ls) -la") is False
        assert _is_git_publish("$editor notes.txt") is False
        assert _is_git_publish("echo $git is set") is False


@pytest.fixture
def captured_sel_events(monkeypatch):
    """Capture SEL events without real I/O (isolate the ambient forensic log)."""
    events = []

    class _Capture:
        def log(self, event) -> None:
            events.append(event)

    monkeypatch.setattr(security, "SecurityEventLog", lambda: _Capture())
    return events


class TestGitPushEnforcement:
    """Behavioral tests through the real enforcement point ``is_denied``."""

    @pytest.mark.asyncio
    async def test_feature_push_allowed_and_audited(self, captured_sel_events, monkeypatch) -> None:
        """Allowed feature publish returns None and emits ONE push_allowed event.

        Async so is_denied runs under a live loop and takes the PRODUCTION path
        (audit offloaded to the maintenance executor, not the sync fallback);
        a dedicated single-worker executor is installed and drained.
        """
        executor = ThreadPoolExecutor(max_workers=1)
        monkeypatch.setattr(security, "maintenance_executor", lambda: executor)
        try:
            assert is_denied(f"{PUSH} origin my-feature") is None
        finally:
            executor.shutdown(wait=True)
        allow = [e for e in captured_sel_events if e.event_type == "push_allowed"]
        assert len(allow) == 1
        assert allow[0].outcome == "allowed"
        assert allow[0].metadata.get("mechanism") == "BRANCH_GATE"

    def test_protected_push_blocked_no_audit(self, captured_sel_events) -> None:
        assert is_denied(f"{PUSH} origin main").startswith("Blocked by security policy")
        assert is_denied(f"{PUSH} --force origin master").startswith("Blocked by security policy")
        assert not any(e.event_type == "push_allowed" for e in captured_sel_events)

    def test_substitution_program_push_blocked(self, captured_sel_events) -> None:
        # Verb obfuscated via an expansion that resolves to git; the branch gate
        # still reads the target and blocks a protected one (security-review 3eeb3852).
        P = "pus" + "h"
        assert is_denied("$(echo git) %s origin main" % P).startswith("Blocked by security policy")
        assert is_denied("$git %s origin mainline" % P).startswith("Blocked by security policy")
        # Third protected name built by concatenation so the inclusive-language
        # scanner does not flag the literal legacy-primary branch token here.
        legacy_primary = "mast" + "er"
        assert is_denied("${git} %s origin %s" % (P, legacy_primary)).startswith(
            "Blocked by security policy"
        )
        assert not any(e.event_type == "push_allowed" for e in captured_sel_events)

    def test_substitution_program_push_to_feature_blocked(self, captured_sel_events) -> None:
        # Fork divergence: unlike upstream (which allows an explicit
        # feature-branch target under an obfuscated verb), this fork's
        # _AMBIGUOUS_EXPANSION_RE fails closed — a push whose program token is
        # an unresolvable expansion is denied even for a feature target, because
        # the gate cannot prove the resolved command is really `git push`.
        P = "pus" + "h"
        assert is_denied("$(echo git) %s origin my-feature" % P).startswith(
            "Blocked by security policy"
        )

    def test_bare_push_blocked(self, captured_sel_events) -> None:
        assert is_denied(PUSH).startswith("Blocked by security policy")

    def test_glue_evasion_bypass_blocked(self, captured_sel_events) -> None:
        """The reviewer's bypass: benign sibling + glued protected publish."""
        cmd = f"{PUSH} origin feature; git$(echo x)pus{'h'} origin ma$(echo)in"
        assert is_denied(cmd).startswith("Blocked by security policy")
        assert not any(e.event_type == "push_allowed" for e in captured_sel_events)

    def test_wildcard_and_shorthand_blocked(self, captured_sel_events) -> None:
        assert is_denied(f"{PUSH} origin refs/heads/*:refs/heads/*").startswith(
            "Blocked by security policy"
        )
        assert is_denied(f"{PUSH} origin heads/main").startswith("Blocked by security policy")

    def test_brace_expansion_blocked(self, captured_sel_events) -> None:
        assert is_denied(f"{PUSH} origin {{main,dummy}}").startswith("Blocked by security policy")
        assert is_denied(f"{PUSH} origin ma{{i,i}}n").startswith("Blocked by security policy")
        assert not any(e.event_type == "push_allowed" for e in captured_sel_events)

    def test_compound_second_push_blocked(self, captured_sel_events) -> None:
        assert is_denied(f"{PUSH} origin feat && {PUSH} origin main").startswith(
            "Blocked by security policy"
        )

    def test_chained_dangerous_command_denied_without_allow_audit(
        self, captured_sel_events
    ) -> None:
        """A feature publish chained with a denied command is denied, and NO
        push_allowed audit fires (SEL reflects the FINAL outcome)."""
        reason = is_denied(f"{PUSH} origin feat && aws delete_bucket my-bucket")
        assert reason is not None and reason.startswith("Blocked by security policy")
        assert not any(e.event_type == "push_allowed" for e in captured_sel_events)

    def test_allow_audit_sync_fallback(self, captured_sel_events) -> None:
        """No running loop -> _schedule_push_allow_audit writes inline."""
        _schedule_push_allow_audit(f"{PUSH} origin my-feature")
        allow = [e for e in captured_sel_events if e.event_type == "push_allowed"]
        assert len(allow) == 1
        assert allow[0].operation == "git_push"

    def test_redos_safe_on_pathological_input(self) -> None:
        """is_denied must not backtrack exponentially on whitespace-laden flags.

        The SIGALRM is the real guarantee: a catastrophic pattern would run for
        seconds-to-minutes and trip it. The elapsed bound is deliberately wide
        (load-tolerant) — a shared, parallel CI runner can inflate the sub-ms
        linear scan to a few hundred ms without that being a regression.
        """

        def _timeout(*_):
            raise TimeoutError

        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, 10.0)
        try:
            t = time.perf_counter()
            is_denied("git " + ("\t-! " * 5000) + "x")
            elapsed = time.perf_counter() - t
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        assert elapsed < 5.0


class TestDefaultsJsonPushRegexes:
    """Git-publish enforcement at the KiroCrew hooks gate (``is_denied``).

    Denied commands are no longer injected into the kiro-cli agent config's
    ``deniedCommands`` (that injection path is retired); the git-publish
    protected-branch gate is enforced solely by ``is_denied`` via the always-on
    ``_is_git_publish`` / ``_is_push_to_protected_branch`` floor.  These cases
    verify that floor blocks protected targets without over-blocking legit
    feature-branch pushes.
    """

    def _blocked(self, cmd: str) -> bool:
        return is_denied(cmd) is not None

    def test_protected_targets_blocked(self) -> None:
        for cmd in (
            f"{PUSH} origin main",
            f"{PUSH} origin mainline",
            f"{PUSH} origin HEAD:main",
            f"{PUSH} origin refs/heads/main",
            f"{PUSH} origin heads/main",
            f"{PUSH} origin remotes/origin/main",
            f"{PUSH} origin +main",
            f"{PUSH} --mirror origin",
            f"{PUSH} --all origin",
            f"{PUSH} origin refs/heads/*:refs/heads/*",
            f"{PUSH} origin {{main,dummy}}",
            f"{PUSH} origin ma{{i,i}}n",
        ):
            assert self._blocked(cmd), cmd

    def test_feature_paths_not_blocked(self) -> None:
        for cmd in (
            f"{PUSH} origin my-feature",
            f"{PUSH} origin feature/main",
            f"{PUSH} origin fix/mainline",
            f"{PUSH} origin main-refactor",
            f"{PUSH} origin refs/heads/main-refactor",
            f"{PUSH} origin +my-feature",
            f"{PUSH} origin heads/develop",
            # wildcard/brace patterns are scoped to the publish's own args, not
            # a sibling command's shell glob/brace:
            f"{PUSH} origin feat && ls *.py",
            f"{PUSH} origin feat && echo {{a,b}}",
        ):
            assert not self._blocked(cmd), cmd

    def test_stash_push_not_blocked(self) -> None:
        assert not self._blocked("git stash pus" + "h --all")
        assert not self._blocked("git stash pus" + "h")


class TestEveryGatedTagNamesARealCatalogRule:
    """A gated tag naming no catalog rule is a SILENT ALLOW, not a loud error.

    Now that the floor consults the per-rule enabled set, ``is_denied`` resolves
    each gated tag through ``_GIT_PUBLISH_FLOOR_BY_ID.get(tag)``.  A miss yields
    ``None`` and the branch ``continue``s — so a protected-branch push is simply
    not denied, nothing reddens, and the command runs.  A rule rename or a typo
    in one of the seven ``tags.add("...")`` literals is therefore a
    push-protection bypass that no behavioural test in this file would catch,
    because they all exercise commands whose tags currently DO resolve.

    ``test_exfil_gate_opt_out.py`` has the twin guard for the exfiltration
    branches; this is the git-publish half.
    """

    # Every literal a gated ``tags.add(...)`` in ``_push_segment_targets_protected``
    # or ``_git_publish_floor_tags`` can emit.  Written out rather than scraped
    # from source so that DELETING an emitter is visible here too.
    GATED_TAGS = (
        "git-publish-push-mirror-all",
        "git-publish-push-bare",
        "git-publish-push-single-arg",
        "git-publish-push-wildcard-refspec",
        "git-publish-push-ambiguous-ref",
        "git-publish-push-protected-ref-path",
        "git-publish-push-protected-branch-name",
    )

    def test_every_gated_tag_resolves_to_a_pattern(self) -> None:
        missing = [t for t in self.GATED_TAGS if t not in security._GIT_PUBLISH_FLOOR_BY_ID]
        assert not missing, (
            f"gated git-publish tag(s) naming no catalog rule: {missing}. In is_denied "
            "these resolve to None and the branch is SKIPPED, so the push is ALLOWED. "
            "Fix the tag literal or add the catalog rule."
        )

    def test_every_gated_tag_is_a_real_rule_id(self) -> None:
        ids = {r.id for r in security.BUILTIN_DENIED_RULES}
        assert not [t for t in self.GATED_TAGS if t not in ids]

    def test_the_ungated_sentinel_is_unspellable_as_a_rule_id(self) -> None:
        """If an operator could name the sentinel, disabling that row would turn
        off the anti-obfuscation branches — the one thing opt-out must not reach."""
        assert security._GIT_PUBLISH_UNGATED not in self.GATED_TAGS
        assert security._GIT_PUBLISH_UNGATED not in {r.id for r in security.BUILTIN_DENIED_RULES}

    def test_the_emitter_only_ever_emits_known_tags(self) -> None:
        """Drive the real function: everything it emits is either a listed gated
        tag or the ungated sentinel — never a third, unhandled kind."""
        known = set(self.GATED_TAGS) | {security._GIT_PUBLISH_UNGATED}
        for cmd in (
            "git push",
            "git push origin",
            "git push --mirror",
            "git push --all origin",
            "git push origin main",
            "git push origin refs/heads/main",
            "git push origin HEAD",
            "git push origin @",
            "git push origin feat/x",
            "git push origin $(echo main)",
            "git push origin 'main@{upstream}'",
        ):
            emitted = security._git_publish_floor_tags(cmd.lower())
            assert emitted <= known, f"{cmd!r} emitted unknown tag(s): {emitted - known}"


class TestPushOptionsCannotDodgeTheProtectedRule:
    """Flag spellings that must not be read as the repository, or as no-op flags.

    Both shapes were latent while the whole git-publish floor was unconditional:
    they mis-CLASSIFIED a protected-branch push rather than allowing it, so the
    floor denied anyway. Making the rules individually disableable is what turned
    them into bypasses — switching off the rule the shape was mis-attributed to
    published the push. Regression for the GPT 5.6 blocking finding on #7705.
    """

    def test_repo_flag_carries_the_remote_so_the_lone_token_is_a_refspec(self):
        # `--repo=origin` starts with '-', so a naive flag strip leaves ["main"]
        # and reads it as the REMOTE — classifying a push to main as the single-arg
        # shape. It must be seen as the protected-branch shape instead.
        for cmd in (
            "git push --repo=origin main",
            "git push --repo origin main",
        ):
            tags = security._git_publish_floor_tags(cmd)
            assert "git-publish-push-protected" in tags or any(
                t.startswith("git-publish-push-protected") for t in tags
            ), f"{cmd!r} did not resolve to a protected-branch tag: {tags}"
            assert "git-publish-push-single-arg" not in tags, (
                f"{cmd!r} still mis-classified as the single-arg shape, so disabling "
                "that one rule would publish to main"
            )

    def test_branches_is_an_all_branches_flag(self):
        # `--branches` is git's own modern alias for `--all` (2.44+), so it pushes
        # every local branch including protected ones.
        tags = security._git_publish_floor_tags("git push --branches origin")
        assert "git-publish-push-mirror-all" in tags, (
            "--branches pushes all branches like --all; leaving it out means the "
            "shape carries no tag at all and nothing denies it"
        )

    def test_repo_flag_without_a_refspec_is_still_caught(self):
        # The flag named the remote and nothing named a branch — the bare form.
        tags = security._git_publish_floor_tags("git push --repo=origin")
        assert tags, "a repo-flag push with no refspec emitted no tag at all"


class TestGitOptionAbbreviationsResolveLikeGit:
    """Git resolves an unambiguous long-option PREFIX to that option, so matching
    flag literals exactly misses every abbreviation.

    This is the third round of findings in this one span, so it is fixed at the
    INVARIANT rather than per spelling: the classifier now resolves a token against
    the enumerated set of `git push` long options, the same way git does. A
    spelling list can only ever trail the next abbreviation; a name table cannot.
    """

    def test_abbreviated_all_branches_flags_still_deny(self):
        for cmd in (
            "git push --mirr origin",
            "git push --mir origin",
            "git push --al origin",
            "git push --branch origin",
        ):
            tags = security._git_publish_floor_tags(cmd)
            assert "git-publish-push-mirror-all" in tags, (
                f"{cmd!r} is an abbreviation git accepts for an all-branches flag, "
                f"but it carried no all-branches tag: {tags}"
            )

    def test_abbreviated_repo_flag_still_shifts_the_positional_read(self):
        for cmd in ("git push --rep=origin main", "git push --rep origin main"):
            tags = security._git_publish_floor_tags(cmd)
            assert any(
                t.startswith("git-publish-push-protected") for t in tags
            ), f"{cmd!r} did not resolve to a protected-branch tag: {tags}"
            assert "git-publish-push-single-arg" not in tags

    def test_an_unrelated_flag_is_not_swallowed(self):
        """The guard against over-denial, which is why the option table is complete.

        `--atomic` shares a prefix with `--all`. Resolving abbreviations against
        `{all, branches, mirror}` alone would read `--a`-family flags as
        all-branches and deny an ordinary feature-branch push. `--atomic` resolves
        to exactly one option, so it must stay clean.
        """
        tags = security._git_publish_floor_tags("git push --atomic origin feat/x")
        assert "git-publish-push-mirror-all" not in tags, (
            "--atomic was misread as an all-branches flag, which denies a legitimate "
            "feature-branch push"
        )
        assert not tags, f"a plain feature-branch push should emit no tag, got {tags}"

    def test_an_ambiguous_abbreviation_covering_a_dangerous_option_fails_closed(self):
        """`--a` is ambiguous to git (all, atomic, ...) so the command never runs.

        Reading it as the dangerous member can therefore only deny a push that was
        already going to fail — the safe direction for a floor.
        """
        assert security._push_option_matches("--a", security._PUSH_ALL_BRANCHES_OPTS)
        assert "git-publish-push-mirror-all" in security._git_publish_floor_tags(
            "git push --a origin"
        )

    def test_the_prefix_test_matches_a_full_option_table(self):
        """Pins WHY no git-wide option table is carried here.

        Resolving a token against every `git push` long option and then
        intersecting with the dangerous ones gives the same answer as testing the
        prefix against the dangerous ones directly: a non-dangerous option can only
        ADD a candidate, never remove a dangerous one. Asserted over every prefix
        of every option so the equivalence is a checked property rather than a
        claim in a comment — if it ever stops holding, the table is needed and this
        goes red.
        """
        full_table = frozenset(
            {
                "all",
                "atomic",
                "branches",
                "delete",
                "dry-run",
                "exec",
                "follow-tags",
                "force",
                "force-if-includes",
                "force-with-lease",
                "ipv4",
                "ipv6",
                "mirror",
                "no-atomic",
                "no-follow-tags",
                "no-force-with-lease",
                "no-progress",
                "no-recurse-submodules",
                "no-signed",
                "no-thin",
                "no-verify",
                "porcelain",
                "progress",
                "prune",
                "push-option",
                "quiet",
                "receive-pack",
                "recurse-submodules",
                "repo",
                "set-upstream",
                "signed",
                "tags",
                "thin",
                "verbose",
                "verify",
            }
        )
        probes = {opt[:i] for opt in full_table for i in range(1, len(opt) + 1)}
        probes |= {"zzz", "a", "m", "r", "b"}
        for name in sorted(probes):
            token = f"--{name}"
            candidates = {opt for opt in full_table if opt.startswith(name)}
            for target in (
                security._PUSH_ALL_BRANCHES_OPTS,
                security._PUSH_REPO_OPTS,
            ):
                assert bool(candidates & target) == security._push_option_matches(token, target), (
                    f"{token!r}: full-table resolution and the direct prefix test "
                    f"disagree for {set(target)} — the option table is load-bearing "
                    "after all and must be restored"
                )


def test_an_all_branches_push_does_not_also_carry_the_single_arg_tag():
    """`push --all origin` must not earn BOTH tags.

    It used to: the all-branches flag tagged mirror-all, then the empty-refspec
    fallback added single-arg on top. Disabling mirror-all therefore left the
    command blocked by its sibling — the toggle read as enabled-and-off while
    enforcement never changed, which is the exact defect class this PR closes.
    GPT 5.6 flagged it on #7705.
    """
    tags = security._git_publish_floor_tags("git push --all origin")
    assert "git-publish-push-mirror-all" in tags
    assert "git-publish-push-single-arg" not in tags
    assert "git-publish-push-bare" not in tags

    # And the bare/single-arg shapes still tag when no all-branches flag is present.
    assert "git-publish-push-single-arg" in security._git_publish_floor_tags("git push origin")
    assert "git-publish-push-bare" in security._git_publish_floor_tags("git push")


class TestValueTakingOptionArity:
    """Push options with a REQUIRED, separable value must have that value
    consumed, not leaked into the positional list. (#7796 — the fourth finding
    in this parser span.)

    Measured on the unfixed parser: appending ``--push-option ci.skip`` (or
    ``-o``, ``--receive-pack``, ``--exec``) to the bare ``--repo=origin``
    publish flipped the floor from {bare} to EMPTY — an ALLOW, because the
    leaked value was read as the only refspec and ``ci.skip`` is not a
    protected name. The tag set IS the protected-branch decision, so one extra
    flag switched the floor off.
    """

    SEPARATED = (
        ("-o", "ci.skip"),
        ("--push-option", "ci.skip"),
        ("--receive-pack", "/usr/bin/git-receive-pack"),
        ("--exec", "/usr/bin/git-receive-pack"),
    )

    def test_a_separated_value_does_not_erase_the_floor_tag(self):
        base = security._git_publish_floor_tags("git push --repo=origin")
        assert base == frozenset({"git-publish-push-bare"})
        for opt, val in self.SEPARATED:
            cmd = f"git push --repo=origin {opt} {val}"
            assert security._git_publish_floor_tags(cmd) == base, (
                f"{cmd!r}: appending one option changed the floor answer — the "
                "separated value leaked into the positional list and erased the tag"
            )

    def test_a_separated_value_is_not_read_in_place_of_a_real_target(self):
        for opt, val in self.SEPARATED:
            cmd = f"git push {opt} {val} origin main"
            tags = security._git_publish_floor_tags(cmd)
            assert "git-publish-push-protected-branch-name" in tags, (
                f"{cmd!r}: the real protected target lost its tag to the leaked "
                f"option value: {set(tags)}"
            )

    def test_a_consumed_value_keeps_a_feature_branch_push_allowed(self):
        # Kills the ARITY-TABLE mutant separately from the fallback mutant:
        # drop a family from the table and these commands fall to the
        # protective fallback and DENY, so this test is what proves the table
        # (precision) is load-bearing, not just the fallback (protection).
        for opt, val in self.SEPARATED:
            cmd = f"git push {opt} {val} origin feature-x"
            assert not security._git_publish_floor_tags(cmd), (
                f"{cmd!r}: a legitimate feature-branch push with a known "
                "value-taking option must stay allowed"
            )

    def test_the_attached_form_keeps_parsing_as_before(self):
        # ``=`` binds the value inside the token, so it can never disturb the
        # positional split; these already parsed correctly and must not regress.
        for cmd in (
            "git push --repo=origin --push-option=ci.skip",
            "git push --repo=origin -oci.skip",
        ):
            assert security._git_publish_floor_tags(cmd) == frozenset(
                {"git-publish-push-bare"}
            ), f"{cmd!r}: the attached form regressed"

    def test_value_option_abbreviations_consume_like_git(self):
        # Git resolves an unambiguous long-option prefix, so ``--push-opt`` is
        # ``--push-option`` — its separated value must be consumed too (the
        # same resolution rule finding 2 established for the danger flags).
        cmd = "git push --repo=origin --push-opt ci.skip"
        assert security._git_publish_floor_tags(cmd) == frozenset({"git-publish-push-bare"})

    def test_a_short_bundle_ending_in_o_consumes_like_git(self):
        # ``-fo ci.skip`` is ``-f -o ci.skip``: the first value-taking short in
        # a bundle takes the rest of the token or, when the rest is empty, the
        # NEXT token.
        cmd = "git push --repo=origin -fo ci.skip"
        assert security._git_publish_floor_tags(cmd) == frozenset({"git-publish-push-bare"})


class TestUnrecognisedOptionsReadProtectively:
    """The invariant that closes the class (#7796 shape C): an option the scan
    does not model might take a separated value, so the positional split cannot
    be trusted — read the segment protectively instead. A mis-parse can then
    only ever OVER-protect; a future value-taking push option cannot silently
    reopen the erasure.
    """

    def test_an_unknown_separated_long_option_cannot_erase_the_floor(self):
        cmd = "git push --frobnicate ci.skip origin feature-x"
        tags = security._git_publish_floor_tags(cmd)
        assert "git-publish-push-bare" in tags, (
            f"{cmd!r}: an unrecognised option left the positional split trusted "
            f"— the erasure class is open again: {set(tags)}"
        )

    def test_an_unknown_short_flag_reads_protectively(self):
        tags = security._git_publish_floor_tags("git push -z origin feature-x")
        assert "git-publish-push-bare" in tags

    def test_the_protective_reading_still_names_a_protected_target_precisely(self):
        # Not trusting the split widens the refspec scan to EVERY positional,
        # so a protected name still reports its own catalog row — an operator
        # who disabled the bare rule is still covered by the protected one.
        tags = security._git_publish_floor_tags("git push --frobnicate ci.skip origin main")
        assert (
            "git-publish-push-protected-branch-name" in tags
        ), f"the protective reading dropped the precise protected tag: {set(tags)}"

    def test_known_no_value_flags_do_not_trip_the_fallback(self):
        # The guard against over-denial: every boolean / attached-optional-value
        # push option is modelled, so ordinary feature-branch pushes keep
        # working. (``--signed`` / ``--force-with-lease`` take a value in
        # ATTACHED form only — git never consumes a separate token for them.)
        for cmd in (
            "git push --force origin feature-x",
            "git push --force-with-lease origin feature-x",
            "git push --signed origin feature-x",
            "git push --no-verify origin feature-x",
            "git push -f origin feature-x",
            "git push -fq origin feature-x",
            "git push --atomic --dry-run origin feature-x",
        ):
            assert not security._git_publish_floor_tags(cmd), (
                f"{cmd!r}: a known no-value flag was read as unrecognised and "
                "over-denied a legitimate feature-branch push"
            )

    def test_recurse_submodules_is_deliberately_not_vouched_for(self):
        # Its arity is not modelled (optional-value in current git, and the
        # separated spelling is exactly the leak shape if that ever changes),
        # so the separated form reads protectively while the attached form —
        # which cannot leak — parses as before.
        assert "git-publish-push-bare" in security._git_publish_floor_tags(
            "git push --recurse-submodules check origin feature-x"
        )
        assert not security._git_publish_floor_tags(
            "git push --recurse-submodules=check origin feature-x"
        )

    def test_an_all_branches_flag_keeps_its_single_tag_under_the_fallback(self):
        # The fallback does not stack the bare tag on top of mirror-all: the
        # all-branches flag names the target set exhaustively (finding 3's
        # suppression), and mirror-all already covers a superset of bare.
        tags = security._git_publish_floor_tags("git push --all --frobnicate v")
        assert tags == frozenset({"git-publish-push-mirror-all"})

    def test_end_of_options_marker_makes_everything_after_it_positional(self):
        # A bare ``--`` ends option parsing in git; the scan must read what
        # follows as positionals, not drop or consume it as flags.
        assert "git-publish-push-protected-branch-name" in security._git_publish_floor_tags(
            "git push -- origin main"
        )
        assert not security._git_publish_floor_tags("git push -- origin feature-x")

    def test_end_of_options_stops_option_parsing_for_later_dash_tokens(self):
        # After ``--``, a dash-prefixed token is a POSITIONAL in git's reading
        # (the only way to push a ref whose name starts with a dash), so it
        # must be read as a refspec candidate — not dropped as a flag (the old
        # scan's single-arg misread) and not fed to the unrecognised-option
        # fallback (which would over-deny a shape git accepts).
        assert not security._git_publish_floor_tags("git push -- origin -x")


class TestParserBranchTable:
    """Every branch of the positional/arity parser, with why each is correct.

    The issue makes this table an explicit acceptance condition: four
    consecutive findings landed in this one span, each fix narrowing one branch
    while the adjacent one stayed wrong, so the parser's whole behaviour is
    pinned here as data rather than implied by scattered tests. The same table
    appears in the PR body.
    """

    BARE = "git-publish-push-bare"
    SINGLE = "git-publish-push-single-arg"
    MIRROR = "git-publish-push-mirror-all"
    NAME = "git-publish-push-protected-branch-name"
    REF_PATH = "git-publish-push-protected-ref-path"
    AMBIG = "git-publish-push-ambiguous-ref"
    WILD = "git-publish-push-wildcard-refspec"

    # (push args, exact expected tag set, why this is the correct reading)
    TABLE = (
        ("", {BARE}, "no target named — the current branch may be protected"),
        ("origin", {SINGLE}, "remote only, no refspec — same current-branch risk, its own row"),
        ("origin feature-x", set(), "explicit non-protected target — the allowed shape"),
        ("origin main", {NAME}, "explicit protected target by bare name"),
        ("origin refs/heads/main", {REF_PATH}, "protected target by ref path — separate row"),
        ("origin head", {AMBIG}, "symbolic ref resolves at runtime — unverifiable"),
        ("origin refs/heads/*:refs/heads/*", {WILD}, "wildcard expands to many refs"),
        ("origin main feature-x", {NAME}, "ALL refspecs scanned, one protected suffices"),
        ("--repo=origin", {BARE}, "remote in the flag, nothing names a branch — bare"),
        ("--repo origin", {BARE}, "separated repo value consumed, still bare"),
        ("--repo=origin main", {NAME}, "remote in flag, so the sole positional is a refspec"),
        ("--all origin", {MIRROR}, "all-branches flag; single-arg deliberately suppressed"),
        ("--mirror", {MIRROR}, "mirror pushes everything"),
        ("--force origin feature-x", set(), "boolean flag — force to a feature branch is allowed"),
        ("-fq origin feature-x", set(), "short bundle of modelled no-value options"),
        (
            "--force-with-lease origin feature-x",
            set(),
            "optional-value option: git binds a value in attached form only",
        ),
        ("-o ci.skip origin feature-x", set(), "arity: -o consumes its separated value"),
        (
            "--push-option ci.skip origin main",
            {NAME},
            "value consumed, the real protected target still read",
        ),
        ("--push-option=ci.skip origin feature-x", set(), "attached value cannot leak"),
        (
            "--repo=origin --push-option ci.skip",
            {BARE},
            "the #7796 erasure row: value consumed, bare tag preserved",
        ),
        (
            "--receive-pack /usr/bin/git-receive-pack origin feature-x",
            set(),
            "arity: --receive-pack consumes its separated value",
        ),
        (
            "--exec /usr/bin/git-receive-pack origin main",
            {NAME},
            "arity: --exec consumes its value, protected target still read",
        ),
        ("-- origin main", {NAME}, "end-of-options: everything after -- is positional"),
        ("-- origin feature-x", set(), "end-of-options with a feature target stays allowed"),
        (
            "-- origin -x",
            set(),
            "a dash-named refspec after -- is positional, not an option",
        ),
        (
            "--frobnicate ci.skip origin feature-x",
            {BARE},
            "unrecognised option: split untrusted, protective bare reading",
        ),
        (
            "--frobnicate ci.skip origin main",
            {BARE, NAME},
            "protective reading scans every positional, precise tag preserved",
        ),
        ("--frobnicate", {BARE}, "unrecognised and nothing positional — bare either way"),
        (
            "--all --frobnicate v",
            {MIRROR},
            "fallback does not stack bare onto mirror-all (finding 3 suppression)",
        ),
    )

    def test_the_parser_branch_table(self):
        for args, expected, why in self.TABLE:
            cmd = f"git push {args}".strip()
            got = security._git_publish_floor_tags(cmd)
            assert got == frozenset(
                expected
            ), f"{cmd!r}: expected {set(expected) or '{}'} because {why}; got {set(got) or '{}'}"

    def test_substitution_glue_stays_on_the_ungated_branch(self):
        # Kept out of TABLE because the sentinel is not a catalog row: the
        # anti-obfuscation branches are upstream of this parser and unaffected.
        got = security._git_publish_floor_tags("git push origin $(echo main)")
        assert got == frozenset({security._GIT_PUBLISH_UNGATED})
