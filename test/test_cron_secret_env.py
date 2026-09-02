"""Tests for operator-granted vault secrets on script/command crons.

Covers the grant validator, the code pin, fail-closed resolution, env
injection in both sandboxed runners, and the persistence-layer gate in
CronService (agent jobs refused, pin required, revoke clears).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kiro_crew.cron import CronService
from kiro_crew.cron_script import (
    _inject_secret_env,
    _secret_env_precheck,
    compute_secret_env_pin,
    run_command_sandboxed,
    run_script_sandboxed,
    validate_secret_env_grant,
)
from kiro_crew.secrets import SecretVault


@pytest.fixture(autouse=True)
def _cron_caller_is_named(named_cron_caller):
    """Cron writes require a nameable caller; these tests assume it."""


@pytest.fixture(autouse=True)
def _crons_dir_tracks_patched_home(monkeypatch, tmp_path):
    """Point cron_script.config_dir at the per-test patched home.

    Same redirect as test_cron_script.py: scripts are written under
    ``<home>/.kirocrew/crons`` and the vault lives beside them, so the
    resolver, the pin computation, and the vault all agree on one root.

    Tests that do NOT patch ``Path.home`` still reach this resolver through
    the grant-pin key (every ``compute_secret_env_pin`` call), so those fall
    back to a per-test tmp dir — never the operator's real ``~/.kirocrew``,
    which this fixture must not create, read, or key.
    """
    real_home = Path.home()
    fallback = tmp_path / "kirocrew-home-fallback"

    def _dir() -> Path:
        home = Path.home()
        if home == real_home:
            return fallback
        return home / ".kirocrew"

    monkeypatch.setattr("kiro_crew.cron_script.config_dir", _dir)


def _make_script(tmp_path: Path, body: str, name: str = "job.py") -> Path:
    crons_dir = tmp_path / ".kirocrew" / "crons"
    crons_dir.mkdir(parents=True, exist_ok=True)
    script = crons_dir / name
    script.write_text(body)
    return script


class TestValidateSecretEnvGrant:
    def test_valid_grant_passes(self):
        validate_secret_env_grant({"MY_SANDBOX_TOKEN": "slack-sandbox"})

    def test_lowercase_key_rejected(self):
        with pytest.raises(ValueError, match="valid env-var name"):
            validate_secret_env_grant({"my_token": "x"})

    @pytest.mark.parametrize(
        "key",
        [
            "PATH",
            "SLACK_BOT_TOKEN",  # _CRON_ENV_DENY member
            "KIROCREW_INTERNAL_SECRET",
            "KIROCREW_ANYTHING",
            "_KIROCREW_SECRET_FILE",
            "LD_PRELOAD",
            "DYLD_INSERT_LIBRARIES",
            "PYTHONPATH",
        ],
    )
    def test_protected_names_rejected(self, key):
        # A leading-underscore name (_KIROCREW_*) fails the grammar check
        # first; the rest hit the protected-name check. Both refuse.
        with pytest.raises(ValueError, match="env-var name"):
            validate_secret_env_grant({key: "x"})

    @pytest.mark.parametrize("name", ["", " padded ", "trailing "])
    def test_bad_vault_name_rejected(self, name):
        with pytest.raises(ValueError, match="vault"):
            validate_secret_env_grant({"MY_TOKEN": name})

    def test_entry_cap_enforced(self):
        grant = {f"KEY_{i}": "v" for i in range(17)}
        with pytest.raises(ValueError, match="max 16"):
            validate_secret_env_grant(grant)


class TestComputeSecretEnvPin:
    def test_command_pin_deterministic_and_text_bound(self):
        a1 = compute_secret_env_pin("", "echo hi")
        a2 = compute_secret_env_pin("", "echo hi")
        b = compute_secret_env_pin("", "echo bye")
        assert a1 == a2
        assert a1 != b

    def test_script_pin_tracks_body(self, tmp_path):
        script = _make_script(tmp_path, "def run(ctx): pass\n")
        spec = str(script) + ":run"
        with patch("pathlib.Path.home", return_value=tmp_path):
            pin1 = compute_secret_env_pin(spec, "")
            script.write_text("def run(ctx): return 1\n")
            pin2 = compute_secret_env_pin(spec, "")
        assert pin1 != pin2

    def test_neither_script_nor_command_raises(self):
        with pytest.raises(ValueError, match="script or command"):
            compute_secret_env_pin("", "")


class TestSecretEnvPrecheck:
    def test_empty_grant_is_noop(self):
        resolved, err = _secret_env_precheck(None, "", command="echo hi")
        assert resolved == {} and err is None
        resolved, err = _secret_env_precheck({}, "", command="echo hi")
        assert resolved == {} and err is None

    def test_missing_pin_fails_closed(self):
        resolved, err = _secret_env_precheck({"T": "x"}, "", command="echo hi")
        assert resolved == {}
        assert err is not None and "re-approve" in err

    def test_changed_command_fails_closed(self):
        pin = compute_secret_env_pin("", "echo hi")
        resolved, err = _secret_env_precheck({"T": "x"}, pin, command="echo bye")
        assert resolved == {}
        assert err is not None and "code changed" in err

    def test_missing_vault_entry_fails_closed(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            pin = compute_secret_env_pin("", "echo hi", grant={"MY_TOKEN": "not-stored"})
            resolved, err = _secret_env_precheck({"MY_TOKEN": "not-stored"}, pin, command="echo hi")
        assert resolved == {}
        assert err is not None and "does not exist" in err
        # The vault secret NAME must not be echoed (CWE-117 discipline is
        # key-only messages); the env-var key is operator config and may be.
        assert "not-stored" not in err
        assert "MY_TOKEN" in err

    def test_resolves_from_vault(self, tmp_path):
        SecretVault(tmp_path / ".kirocrew").set_sync("slack-sandbox", "xoxb-123")
        with patch("pathlib.Path.home", return_value=tmp_path):
            pin = compute_secret_env_pin("", "echo hi", grant={"MY_SANDBOX_TOKEN": "slack-sandbox"})
            resolved, err = _secret_env_precheck(
                {"MY_SANDBOX_TOKEN": "slack-sandbox"}, pin, command="echo hi"
            )
        assert err is None
        assert resolved == {"MY_SANDBOX_TOKEN": "xoxb-123"}


class TestRunCommandSandboxedSecretEnv:
    @pytest.fixture(autouse=True)
    def _passthrough_sandbox(self, monkeypatch, posix_test_shell):
        monkeypatch.setattr("kiro_crew.cron_script.wrap_argv", lambda argv, **k: (list(argv), None))
        monkeypatch.setattr(
            "kiro_crew.cron_script._resolve_command_shell", lambda: posix_test_shell
        )

    def test_granted_secret_reaches_child_env(self, tmp_path):
        SecretVault(tmp_path / ".kirocrew").set_sync("slack-sandbox", "xoxb-inject")
        cmd = 'printf "%s" "$MY_SANDBOX_TOKEN"'
        with patch("pathlib.Path.home", return_value=tmp_path):
            pin = compute_secret_env_pin("", cmd, grant={"MY_SANDBOX_TOKEN": "slack-sandbox"})
            result = run_command_sandboxed(
                cmd,
                timeout=30,
                secret_env={"MY_SANDBOX_TOKEN": "slack-sandbox"},
                secret_env_pin=pin,
            )
        assert result["status"] == "ok"
        assert result["output"] == "xoxb-inject"

    def test_pending_pin_copied_to_active_fields_fails_closed(self, tmp_path):
        """The F-exploit: a store writer copies a valid PENDING pin into the
        active grant fields. Domain separation must refuse it."""
        SecretVault(tmp_path / ".kirocrew").set_sync("slack-sandbox", "xoxb-inject")
        cmd = 'printf "%s" "$MY_SANDBOX_TOKEN"'
        grant = {"MY_SANDBOX_TOKEN": "slack-sandbox"}
        with patch("pathlib.Path.home", return_value=tmp_path):
            pending_pin = compute_secret_env_pin(
                "", cmd, job_id="job1", grant=grant, domain="pending"
            )
            result = run_command_sandboxed(
                cmd,
                timeout=30,
                job_id="job1",
                secret_env=grant,
                secret_env_pin=pending_pin,
            )
        assert result["status"] == "error"
        assert "code changed" in result["output"]

    def test_foreign_job_pin_fails_closed(self, tmp_path):
        """A pin minted for one job must not authorize another job's grant."""
        SecretVault(tmp_path / ".kirocrew").set_sync("slack-sandbox", "xoxb-inject")
        cmd = 'printf "%s" "$MY_SANDBOX_TOKEN"'
        grant = {"MY_SANDBOX_TOKEN": "slack-sandbox"}
        with patch("pathlib.Path.home", return_value=tmp_path):
            other_pin = compute_secret_env_pin("", cmd, job_id="other-job", grant=grant)
            result = run_command_sandboxed(
                cmd, timeout=30, job_id="job1", secret_env=grant, secret_env_pin=other_pin
            )
        assert result["status"] == "error"

    def test_pin_mismatch_blocks_execution(self, tmp_path):
        SecretVault(tmp_path / ".kirocrew").set_sync("slack-sandbox", "xoxb-inject")
        pin = compute_secret_env_pin("", "echo original")
        marker = tmp_path / "ran"
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = run_command_sandboxed(
                f"touch {marker}",
                timeout=30,
                secret_env={"MY_SANDBOX_TOKEN": "slack-sandbox"},
                secret_env_pin=pin,
            )
        assert result["status"] == "error"
        assert "code changed" in result["output"]
        # Fail-closed means the command never ran at all.
        assert not marker.exists()

    def test_no_grant_leaves_env_clean(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = run_command_sandboxed('printf "%s" "${MY_SANDBOX_TOKEN:-unset}"')
        assert result["status"] == "ok"
        assert result["output"] == "unset"


class TestRunScriptSandboxedSecretEnv:
    @pytest.fixture(autouse=True)
    def _passthrough_sandbox(self, monkeypatch):
        monkeypatch.setattr("kiro_crew.cron_script.wrap_argv", lambda argv, **k: (list(argv), None))

    def test_granted_secret_reaches_script(self, tmp_path):
        SecretVault(tmp_path / ".kirocrew").set_sync("slack-sandbox", "xoxb-script")
        script = _make_script(
            tmp_path,
            "import os\n"
            "from kiro_crew.cron_script import Done\n"
            "def run(ctx):\n"
            "    raise Done(os.environ.get('MY_SANDBOX_TOKEN', 'MISSING'))\n",
        )
        spec = str(script) + ":run"
        with patch("pathlib.Path.home", return_value=tmp_path):
            pin = compute_secret_env_pin(
                spec, "", job_id="job1", grant={"MY_SANDBOX_TOKEN": "slack-sandbox"}
            )
            result = run_script_sandboxed(
                spec,
                "job1",
                timeout=120,
                secret_env={"MY_SANDBOX_TOKEN": "slack-sandbox"},
                secret_env_pin=pin,
            )
        assert result["status"] == "done", result
        assert result["message"] == "xoxb-script"

    def test_body_rewrite_after_grant_fails_closed(self, tmp_path):
        SecretVault(tmp_path / ".kirocrew").set_sync("slack-sandbox", "xoxb-script")
        script = _make_script(tmp_path, "def run(ctx): pass\n")
        spec = str(script) + ":run"
        with patch("pathlib.Path.home", return_value=tmp_path):
            pin = compute_secret_env_pin(spec, "")
            # The agent-writeable script is swapped after the operator granted.
            script.write_text(
                "import os\n"
                "def run(ctx):\n"
                "    open('/tmp/exfil', 'w').write(os.environ.get('MY_SANDBOX_TOKEN',''))\n"
            )
            result = run_script_sandboxed(
                spec,
                "job1",
                timeout=120,
                secret_env={"MY_SANDBOX_TOKEN": "slack-sandbox"},
                secret_env_pin=pin,
            )
        assert result["status"] == "error"
        assert "code changed" in result["error"]


class TestCronServiceSecretEnvGate:
    def _service(self, tmp_path: Path) -> CronService:
        return CronService(base_dir=tmp_path / "crons-store")

    def test_grant_persists_and_round_trips(self, tmp_path):
        svc = self._service(tmp_path)
        job = svc.add_job("j", "m", every_secs=3600, command="echo hi")
        pin = compute_secret_env_pin("", "echo hi")
        updated = svc.update_job(
            job.id, secret_env={"MY_TOKEN": "slack-sandbox"}, secret_env_pin=pin
        )
        assert updated is not None
        assert updated.secret_env == {"MY_TOKEN": "slack-sandbox"}
        assert updated.secret_env_pin == pin
        # Round-trip through a fresh service instance (disk reload).
        svc2 = self._service(tmp_path)
        reloaded = next(j for j in svc2.list_jobs() if j.id == job.id)
        assert reloaded.secret_env == {"MY_TOKEN": "slack-sandbox"}
        assert reloaded.secret_env_pin == pin

    def test_agent_job_refused(self, tmp_path):
        svc = self._service(tmp_path)
        job = svc.add_job("j", "check the queue", every_secs=3600)
        with pytest.raises(ValueError, match="script/command"):
            svc.update_job(job.id, secret_env={"MY_TOKEN": "x"}, secret_env_pin="deadbeef")

    def test_grant_without_pin_refused(self, tmp_path):
        svc = self._service(tmp_path)
        job = svc.add_job("j", "m", every_secs=3600, command="echo hi")
        with pytest.raises(ValueError, match="secret_env_pin"):
            svc.update_job(job.id, secret_env={"MY_TOKEN": "x"})

    def test_protected_name_refused_at_persistence(self, tmp_path):
        svc = self._service(tmp_path)
        job = svc.add_job("j", "m", every_secs=3600, command="echo hi")
        with pytest.raises(ValueError, match="protected env-var name"):
            svc.update_job(
                job.id,
                secret_env={"SLACK_BOT_TOKEN": "x"},
                secret_env_pin="deadbeef",
            )

    def test_revoke_clears_pin(self, tmp_path):
        svc = self._service(tmp_path)
        job = svc.add_job("j", "m", every_secs=3600, command="echo hi")
        pin = compute_secret_env_pin("", "echo hi")
        svc.update_job(job.id, secret_env={"MY_TOKEN": "n"}, secret_env_pin=pin)
        revoked = svc.update_job(job.id, secret_env={})
        assert revoked is not None
        assert revoked.secret_env == {}
        assert revoked.secret_env_pin == ""

    def test_unrelated_update_leaves_grant_alone(self, tmp_path):
        svc = self._service(tmp_path)
        job = svc.add_job("j", "m", every_secs=3600, command="echo hi")
        pin = compute_secret_env_pin("", "echo hi")
        svc.update_job(job.id, secret_env={"MY_TOKEN": "n"}, secret_env_pin=pin)
        updated = svc.update_job(job.id, name="renamed")
        assert updated is not None
        assert updated.secret_env == {"MY_TOKEN": "n"}
        assert updated.secret_env_pin == pin


class TestInjectionDefenseInDepth:
    def test_protected_keys_skipped_at_injection(self):
        """A hand-edited store could hold a protected key; injection skips it,
        so the product-internal keys set afterwards always win."""
        clean: dict[str, str] = {}
        _inject_secret_env(clean, {"_KIROCREW_SECRET_FILE": "evil", "OK_TOKEN": "fine"})
        assert "_KIROCREW_SECRET_FILE" not in clean
        assert clean == {"OK_TOKEN": "fine"}

    def test_forged_plain_hash_pin_is_rejected(self, tmp_path):
        """The pin is keyed HMAC: a store writer who recomputes the old plain
        sha256 shape (or any digest without the vault-fenced key) cannot mint
        a pin the runner accepts."""
        import hashlib

        forged = hashlib.sha256(b"command\x00" + b"echo hi").hexdigest()
        with patch("pathlib.Path.home", return_value=tmp_path):
            resolved, err = _secret_env_precheck({"T": "x"}, forged, command="echo hi")
        assert resolved == {}
        assert err is not None and "code changed" in err

    def test_granted_script_cannot_import_live_crons_sibling(self, tmp_path):
        """The pin covers one approved body; a sibling module in the live,
        agent-writeable crons/ dir must not be importable at fire time."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            script = _make_script(
                tmp_path,
                "import helper\n"
                "from kiro_crew.cron_script import Done\n"
                "def run(ctx):\n"
                "    raise Done(helper.VALUE)\n",
            )
            _make_script(tmp_path, "VALUE = 'sibling ran'\n", name="helper.py")
            spec = str(script) + ":run"
            pin = compute_secret_env_pin(
                spec, "", job_id="job1", grant={"MY_TOKEN": "slack-sandbox"}
            )
            with patch("kiro_crew.cron_script.wrap_argv", lambda argv, **k: (list(argv), None)):
                SecretVault(tmp_path / ".kirocrew").set_sync("slack-sandbox", "xoxb-1")
                result = run_script_sandboxed(
                    spec,
                    "job1",
                    timeout=120,
                    secret_env={"MY_TOKEN": "slack-sandbox"},
                    secret_env_pin=pin,
                )
        # The sibling import fails (ModuleNotFoundError surfaces as the
        # script error), rather than the unpinned helper running with the
        # secret in env.
        assert result["status"] == "error", result
        assert "helper" in result.get("error", "")


class TestMcpSecretRequest:
    """The agent-reachable half: cron_secret_request writes PENDING only."""

    def _svc_and_job(self, *, command: str = "echo hi", agent: bool = False):
        from kiro_crew.config.loader import config_dir as loader_config_dir

        svc = CronService(base_dir=loader_config_dir())
        job = svc.add_job(
            "j",
            "m" if agent else "args",
            every_secs=3600,
            command="" if agent else command,
            session_key="dashboard:conftest-slot",
        )
        return svc, job

    def _vault(self):
        from kiro_crew.config.loader import config_dir as loader_config_dir

        return SecretVault(loader_config_dir())

    def test_request_records_pending_not_active(self):
        from kiro_crew import mcp_cron

        svc, job = self._svc_and_job()
        self._vault().set_sync("slack-sandbox", "xoxb-1")
        out = mcp_cron._call_tool(
            "cron_secret_request",
            {"job_id": job.id, "secrets": {"MY_TOKEN": "slack-sandbox"}},
        )
        assert "PENDING" in out
        reloaded = CronService(base_dir=svc._dir).get_job(job.id)
        assert reloaded is not None
        assert reloaded.secret_env_pending == {"MY_TOKEN": "slack-sandbox"}
        assert reloaded.secret_env_pending_pin
        # The active grant is untouched — the tool cannot grant.
        assert reloaded.secret_env == {}
        assert reloaded.secret_env_pin == ""

    def test_request_refuses_unknown_vault_name(self):
        from kiro_crew import mcp_cron

        svc, job = self._svc_and_job()
        out = mcp_cron._call_tool(
            "cron_secret_request",
            {"job_id": job.id, "secrets": {"MY_TOKEN": "never-stored"}},
        )
        assert out.startswith("Error:")
        reloaded = CronService(base_dir=svc._dir).get_job(job.id)
        assert reloaded is not None and reloaded.secret_env_pending == {}

    def test_request_refuses_agent_job(self):
        from kiro_crew import mcp_cron

        svc, job = self._svc_and_job(agent=True)
        self._vault().set_sync("slack-sandbox", "xoxb-1")
        out = mcp_cron._call_tool(
            "cron_secret_request",
            {"job_id": job.id, "secrets": {"MY_TOKEN": "slack-sandbox"}},
        )
        assert out.startswith("Error:")

    def test_request_refuses_protected_env_name(self):
        from kiro_crew import mcp_cron

        svc, job = self._svc_and_job()
        self._vault().set_sync("slack-sandbox", "xoxb-1")
        out = mcp_cron._call_tool(
            "cron_secret_request",
            {"job_id": job.id, "secrets": {"SLACK_BOT_TOKEN": "slack-sandbox"}},
        )
        assert out.startswith("Error:")

    def test_empty_request_withdraws_pending(self):
        from kiro_crew import mcp_cron

        svc, job = self._svc_and_job()
        self._vault().set_sync("slack-sandbox", "xoxb-1")
        mcp_cron._call_tool(
            "cron_secret_request",
            {"job_id": job.id, "secrets": {"MY_TOKEN": "slack-sandbox"}},
        )
        out = mcp_cron._call_tool("cron_secret_request", {"job_id": job.id, "secrets": {}})
        assert "Withdrew" in out
        reloaded = CronService(base_dir=svc._dir).get_job(job.id)
        assert reloaded is not None
        assert reloaded.secret_env_pending == {}
        assert reloaded.secret_env_pending_pin == ""

    def test_cron_update_cannot_write_active_grant(self):
        """The general-purpose MCP update tool must never carry the grant."""
        from kiro_crew import mcp_cron

        svc, job = self._svc_and_job()
        mcp_cron._call_tool(
            "cron_update",
            {
                "job_id": job.id,
                "name": "renamed",
                "secret_env": {"MY_TOKEN": "slack-sandbox"},
                "secret_env_pin": "deadbeef",
            },
        )
        reloaded = CronService(base_dir=svc._dir).get_job(job.id)
        assert reloaded is not None
        assert reloaded.secret_env == {}
        assert reloaded.secret_env_pin == ""


class TestGrantEndpointPendingFlow:
    """The operator half: approve/deny via the dashboard endpoint."""

    @pytest.fixture(autouse=True)
    def _owner_view(self, monkeypatch):
        """These tests exercise grant flow, not identity; grant is owner-only."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
            lambda request: True,
        )

    def _app_and_svc(self):
        from types import SimpleNamespace

        from aiohttp import web

        from kiro_crew.config.loader import config_dir as loader_config_dir
        from kiro_crew.dashboard.handlers.cron import api_cron_secret_grant

        svc = CronService(base_dir=loader_config_dir())
        app = web.Application()
        app["state"] = SimpleNamespace(crons=svc, push_refresh=lambda *a, **k: None)
        app.router.add_put("/api/crons/{job_id}/secrets", api_cron_secret_grant)
        return app, svc

    def _request_pending(self, svc, *, command: str = "echo hi") -> str:
        job = svc.add_job(
            "j",
            "args",
            every_secs=3600,
            command=command,
            session_key="dashboard:conftest-slot",
        )
        pin = compute_secret_env_pin(
            "",
            command,
            "args",
            job_id=job.id,
            grant={"MY_TOKEN": "slack-sandbox"},
            domain="pending",
        )
        svc.update_job(
            job.id,
            secret_env_pending={"MY_TOKEN": "slack-sandbox"},
            secret_env_pending_pin=pin,
            secret_env_pending_ts=1.0,
        )
        return job.id

    @pytest.mark.asyncio
    async def test_approve_promotes_pending_to_active(self):
        from aiohttp.test_utils import TestClient, TestServer

        from kiro_crew.config.loader import config_dir as loader_config_dir

        app, svc = self._app_and_svc()
        SecretVault(loader_config_dir()).set_sync("slack-sandbox", "xoxb-1")
        jid = self._request_pending(svc)
        with patch("kiro_crew.dashboard.handlers.cron._sel"):
            async with TestClient(TestServer(app)) as client:
                resp = await client.put(f"/api/crons/{jid}/secrets", json={"approve_pending": True})
        assert resp.status == 200
        job = CronService(base_dir=svc._dir).get_job(jid)
        assert job is not None
        assert job.secret_env == {"MY_TOKEN": "slack-sandbox"}
        assert job.secret_env_pin
        assert job.secret_env_pending == {}

    @pytest.mark.asyncio
    async def test_approve_refuses_when_code_changed_after_request(self):
        from aiohttp.test_utils import TestClient, TestServer

        from kiro_crew.config.loader import config_dir as loader_config_dir

        app, svc = self._app_and_svc()
        SecretVault(loader_config_dir()).set_sync("slack-sandbox", "xoxb-1")
        # A pending request whose pin was taken over DIFFERENT code than the
        # job now carries — the shape a post-request command rewrite produces.
        job = svc.add_job(
            "j",
            "args",
            every_secs=3600,
            command="echo current",
            session_key="dashboard:conftest-slot",
        )
        svc.update_job(
            job.id,
            secret_env_pending={"MY_TOKEN": "slack-sandbox"},
            secret_env_pending_pin=compute_secret_env_pin("", "echo what-was-requested"),
            secret_env_pending_ts=1.0,
        )
        with patch("kiro_crew.dashboard.handlers.cron._sel"):
            async with TestClient(TestServer(app)) as client:
                resp = await client.put(
                    f"/api/crons/{job.id}/secrets", json={"approve_pending": True}
                )
                body = await resp.json()
        assert resp.status == 409
        assert body["code"] == "code_changed"
        reloaded = CronService(base_dir=svc._dir).get_job(job.id)
        assert reloaded is not None and reloaded.secret_env == {}

    @pytest.mark.asyncio
    async def test_deny_with_stale_expected_refuses(self):
        """A denial restating request A must not clear a replacement B."""
        from aiohttp.test_utils import TestClient, TestServer

        app, svc = self._app_and_svc()
        jid = self._request_pending(svc)
        svc.update_job(
            jid,
            secret_env_pending={"MY_TOKEN": "other-secret"},
            secret_env_pending_pin=compute_secret_env_pin(
                "",
                "echo hi",
                "args",
                job_id=jid,
                grant={"MY_TOKEN": "other-secret"},
                domain="pending",
            ),
            secret_env_pending_ts=2.0,
        )
        with patch("kiro_crew.dashboard.handlers.cron._sel"):
            async with TestClient(TestServer(app)) as client:
                resp = await client.put(
                    f"/api/crons/{jid}/secrets",
                    json={
                        "deny_pending": True,
                        "expected_secret_env": {"MY_TOKEN": "slack-sandbox"},
                    },
                )
                body = await resp.json()
        assert resp.status == 409
        assert body["code"] == "stale_request"
        job = CronService(base_dir=svc._dir).get_job(jid)
        assert job is not None and job.secret_env_pending == {"MY_TOKEN": "other-secret"}

    @pytest.mark.asyncio
    async def test_deny_clears_pending(self):
        from aiohttp.test_utils import TestClient, TestServer

        app, svc = self._app_and_svc()
        jid = self._request_pending(svc)
        with patch("kiro_crew.dashboard.handlers.cron._sel"):
            async with TestClient(TestServer(app)) as client:
                resp = await client.put(f"/api/crons/{jid}/secrets", json={"deny_pending": True})
        assert resp.status == 200
        job = CronService(base_dir=svc._dir).get_job(jid)
        assert job is not None
        assert job.secret_env_pending == {}
        assert job.secret_env == {}


class TestInlineApprovalCard:
    """Machine asks, human answers: the inline card endpoint boundaries."""

    @pytest.fixture(autouse=True)
    def _owner_view(self, monkeypatch):
        """These tests exercise card flow, not identity; grant is owner-only."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
            lambda request: True,
        )

    @pytest.mark.asyncio
    async def test_replaced_pending_request_is_not_promoted_by_stale_approval(self):
        """The agent can overwrite a pending request at any moment; an
        approval that restates request A must not promote a replacement B."""
        from aiohttp.test_utils import TestClient, TestServer

        from kiro_crew.config.loader import config_dir as loader_config_dir

        app, svc, _state = self._apps(internal_auth=False)
        SecretVault(loader_config_dir()).set_sync("slack-sandbox", "xoxb-1")
        SecretVault(loader_config_dir()).set_sync("jira-token", "fake-2")
        jid = self._pending_job(svc)  # pending: MY_TOKEN <- slack-sandbox
        # Agent replaces the request AFTER the operator's view rendered.
        svc.update_job(
            jid,
            secret_env_pending={"MY_TOKEN": "jira-token"},
            secret_env_pending_pin=compute_secret_env_pin("", "echo hi"),
            secret_env_pending_ts=2.0,
        )
        with patch("kiro_crew.dashboard.handlers.cron._sel"):
            async with TestClient(TestServer(app)) as client:
                resp = await client.put(
                    f"/api/crons/{jid}/secrets",
                    json={
                        "approve_pending": True,
                        "expected_secret_env": {"MY_TOKEN": "slack-sandbox"},
                    },
                )
                body = await resp.json()
        assert resp.status == 409
        assert body["code"] == "stale_request"
        job = CronService(base_dir=svc._dir).get_job(jid)
        assert job is not None and job.secret_env == {}

    @pytest.mark.asyncio
    async def test_non_owner_dashboard_token_cannot_grant(self, monkeypatch):
        """A dashboard token minted for an allowed Slack user (not the owner)
        must not approve grants — the vault boundary is owner-only."""
        from aiohttp.test_utils import TestClient, TestServer

        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
            lambda request: False,
        )
        app, svc, _state = self._apps(internal_auth=False)
        jid = self._pending_job(svc)
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(f"/api/crons/{jid}/secrets", json={"approve_pending": True})
        assert resp.status == 403
        job = CronService(base_dir=svc._dir).get_job(jid)
        assert job is not None and job.secret_env == {}

    @pytest.mark.asyncio
    async def test_non_owner_cannot_resolve_secret_card_via_approvals_endpoint(self, monkeypatch):
        """The generic approvals-resolve endpoint is the card's only resolve
        surface; for cron-secret cards it must be owner-only too — otherwise a
        non-owner could answer the card and drive the promotion the grant
        endpoint's own owner gate refuses them."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from kiro_crew.dashboard.handlers.sessions import api_approval_resolve

        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
            lambda request: False,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.stale_owner_session_response",
            lambda request: None,
        )
        state = SimpleNamespace(resolve_approval=MagicMock(return_value=True))
        app = web.Application()
        app["state"] = state
        app.router.add_post("/api/approvals/{id}/{action}", api_approval_resolve)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/approvals/cron-secret:abc123:deadbeef/approve")
            # An ordinary tool approval is untouched by the gate.
            resp_other = await client.post("/api/approvals/tool-xyz/approve")
        assert resp.status == 403
        assert resp_other.status == 200
        state.resolve_approval.assert_called_once()  # only the ordinary one

    def _apps(self, *, internal_auth: bool):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from aiohttp import web

        from kiro_crew.config.loader import config_dir as loader_config_dir
        from kiro_crew.dashboard.handlers.cron import (
            api_cron_secret_grant,
            api_cron_secret_request_card,
        )

        svc = CronService(base_dir=loader_config_dir())

        @web.middleware
        async def _mark_internal(request, handler):
            if internal_auth:
                request["internal_auth"] = True
            return await handler(request)

        state = SimpleNamespace(
            crons=svc,
            push_refresh=lambda *a, **k: None,
            _slots={"chat-1-abc": object()},
            request_approval=AsyncMock(return_value=True),
        )
        app = web.Application(middlewares=[_mark_internal])
        app["state"] = state
        app.router.add_put("/api/crons/{job_id}/secrets", api_cron_secret_grant)
        app.router.add_post("/api/crons/{job_id}/secret-request-card", api_cron_secret_request_card)
        return app, svc, state

    def _pending_job(self, svc) -> str:
        job = svc.add_job(
            "j",
            "args",
            every_secs=3600,
            command="echo hi",
            session_key="dashboard:chat-1-abc",
        )
        svc.update_job(
            job.id,
            secret_env_pending={"MY_TOKEN": "slack-sandbox"},
            secret_env_pending_pin=compute_secret_env_pin(
                "",
                "echo hi",
                "args",
                job_id=job.id,
                grant={"MY_TOKEN": "slack-sandbox"},
                domain="pending",
            ),
            secret_env_pending_ts=1.0,
        )
        return job.id

    @pytest.mark.asyncio
    async def test_grant_endpoint_refuses_machine_credential(self):
        """/api/crons is a prefix internal path, so the grant route IS
        reachable with X-Internal-Secret — the handler must refuse it."""
        from aiohttp.test_utils import TestClient, TestServer

        app, svc, _state = self._apps(internal_auth=True)
        jid = self._pending_job(svc)
        with patch("kiro_crew.dashboard.handlers.cron._sel"):
            async with TestClient(TestServer(app)) as client:
                resp = await client.put(f"/api/crons/{jid}/secrets", json={"approve_pending": True})
                body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "operator_only"
        job = CronService(base_dir=svc._dir).get_job(jid)
        assert job is not None and job.secret_env == {}

    @pytest.mark.asyncio
    async def test_card_endpoint_requires_machine_credential(self):
        from aiohttp.test_utils import TestClient, TestServer

        app, svc, _state = self._apps(internal_auth=False)
        jid = self._pending_job(svc)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                f"/api/crons/{jid}/secret-request-card",
                json={"session_key": "dashboard:chat-1-abc"},
            )
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_card_approval_promotes_via_pin_verified_path(self):
        import asyncio

        from aiohttp.test_utils import TestClient, TestServer

        from kiro_crew.config.loader import config_dir as loader_config_dir

        app, svc, state = self._apps(internal_auth=True)
        SecretVault(loader_config_dir()).set_sync("slack-sandbox", "xoxb-1")
        jid = self._pending_job(svc)
        with patch("kiro_crew.dashboard.handlers.cron._sel"):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    f"/api/crons/{jid}/secret-request-card",
                    json={"session_key": "dashboard:chat-1-abc"},
                )
                body = await resp.json()
                assert resp.status == 202 and body["card"] is True
                # Let the background decision task run to completion.
                from kiro_crew.dashboard.handlers.cron import _SECRET_CARD_TASKS

                await asyncio.gather(*_SECRET_CARD_TASKS)
        state.request_approval.assert_awaited_once()
        # The card summary carries names only, never secret values.
        summary = state.request_approval.await_args.kwargs["tool_input"]
        assert "MY_TOKEN" in summary and "xoxb-1" not in summary
        job = CronService(base_dir=svc._dir).get_job(jid)
        assert job is not None
        assert job.secret_env == {"MY_TOKEN": "slack-sandbox"}
        assert job.secret_env_pending == {}

    @pytest.mark.asyncio
    async def test_card_denial_keeps_pending_record(self):
        import asyncio

        from aiohttp.test_utils import TestClient, TestServer

        app, svc, state = self._apps(internal_auth=True)
        state.request_approval.return_value = False
        jid = self._pending_job(svc)
        with patch("kiro_crew.dashboard.handlers.cron._sel"):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    f"/api/crons/{jid}/secret-request-card",
                    json={"session_key": "dashboard:chat-1-abc"},
                )
                assert resp.status == 202
                from kiro_crew.dashboard.handlers.cron import _SECRET_CARD_TASKS

                await asyncio.gather(*_SECRET_CARD_TASKS)
        job = CronService(base_dir=svc._dir).get_job(jid)
        assert job is not None
        # Deny/timeout are indistinguishable on the card — the durable pending
        # record stays, approvable or revocable from the Schedule page.
        assert job.secret_env == {} and job.secret_env_pending == {"MY_TOKEN": "slack-sandbox"}

    @pytest.mark.asyncio
    async def test_card_endpoint_without_slot_reports_no_card(self):
        from aiohttp.test_utils import TestClient, TestServer

        app, svc, state = self._apps(internal_auth=True)
        jid = self._pending_job(svc)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                f"/api/crons/{jid}/secret-request-card",
                json={"session_key": "dashboard:not-a-real-slot"},
            )
            body = await resp.json()
        assert resp.status == 200
        assert body == {"ok": True, "card": False}
        state.request_approval.assert_not_awaited()
