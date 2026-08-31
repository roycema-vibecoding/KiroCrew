"""Tests for the personal cloud drive publish provider.

No AWS is reached: the deploy engine's call surface is stubbed, so these assert the
provider's OWN decisions -- account resolution and binding, visibility semantics, the
served-prefix bucket policy, the content type and digest on upload, the two dead-link
states and which paths may be blocked by them, and what the public edition registers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from kiro_crew import publish_provider as pp
from kiro_crew.publish import personal_drive as pd
from kiro_crew.publish_provider import Capability, CapabilityNotSupportedError, PublishError

BUCKET = f"{pd.BUCKET_PREFIX}abc123def456"
POLICY_NEW = "RHP-NEW"
POLICY_EXISTING = "RHP-EXISTING"
DIST = "E123"
DOMAIN = "d1.cloudfront.net"


@pytest.fixture(autouse=True)
def _clean_registry():
    """Full isolation, and the registry is put BACK.

    ``reset_providers()`` clears cached INSTANCES only, so without also clearing the
    factory map a registration would leak into the next test. Clearing alone is not
    enough either: the map is process-wide, so a bare clear on teardown leaves every
    later test in the worker looking at an empty registry -- the leak inverted.
    """
    saved = dict(pp._FACTORIES)
    pp._FACTORIES.clear()
    pp.reset_providers()
    yield
    pp._FACTORIES.clear()
    pp._FACTORIES.update(saved)
    pp.reset_providers()


@pytest.fixture
def payload(tmp_path: Path) -> str:
    f = tmp_path / "index.html"
    f.write_text("<h1>hi</h1>", encoding="utf-8")
    return str(f)


def _registry(names: list[str], default: str = "") -> dict:
    return {
        "version": 2,
        "profiles": [{"name": n, "region": "us-west-2"} for n in names],
        "default": default or (names[0] if names else ""),
    }


def _stub_profiles(monkeypatch, names: list[str], default: str = "") -> dict:
    reg = _registry(names, default)
    monkeypatch.setattr(pd.profiles, "load_registry", lambda: reg)

    def _resolve(requested: str = ""):
        name = requested or reg["default"]
        if not name:
            return None
        for entry in reg["profiles"]:
            if entry["name"] == name:
                return entry["name"], entry["region"]
        return None

    monkeypatch.setattr(pd.profiles, "resolve_profile", _resolve)
    return reg


class _EngineSpy:
    """Records every engine call the provider makes.

    ``existing_keys=None`` means "every object exists"; a set makes ``head-object``
    key-aware so a test can put the drive in a specific state. ``drive=False`` makes the
    drive not exist yet, exercising the create branch. ``head_error`` makes
    ``head-object`` fail with a NON-not-found error.
    """

    def __init__(
        self,
        *,
        status: str = "Deployed",
        enabled: bool = True,
        existing_keys: set[str] | None = None,
        drive: bool = True,
        head_error: str = "",
        stored_digest: str = "",
        content_type: str = "text/html",
        existing_policy: bool = False,
        # The CSP directive / Override the pre-existing same-named policy carries. Default
        # to the module's exact sandbox values so existing_policy=True models a correct
        # policy that should be REUSED; a test overrides these to model a pre-created
        # policy WITHOUT the sandbox (wrong/absent CSP, or Override false) that must fail
        # closed. ``None`` drops the ContentSecurityPolicy block entirely (absent CSP).
        existing_policy_csp: str | None = pd._SANDBOX_CSP,
        existing_policy_override: bool = True,
        buckets: list[str] | None = None,
        # Does the TAGGING API report the distribution (only ever in us-east-1)?
        cf_tags_visible: bool = True,
        # Does CloudFront's OWN tag API report the drive tag on it?
        cf_own_tags: bool = True,
    ):
        self.status = status
        self.enabled = enabled
        self.existing_keys = existing_keys
        self.drive = drive
        self.cf_tags_visible = cf_tags_visible
        self.cf_own_tags = cf_own_tags
        self.head_error = head_error
        self.stored_digest = stored_digest
        self.content_type = content_type
        self.existing_policy = existing_policy
        self.existing_policy_csp = existing_policy_csp
        self.existing_policy_override = existing_policy_override
        self.policy_ids: list[str] = []
        self.buckets = buckets
        self.checked: list[list[str]] = []
        self.invalidated: list[str] = []
        self.created_buckets: list[str] = []
        self.created_distributions: list[str] = []
        self.created_site_ids: list[str] = []
        self.created_tags: list[object] = []
        self.tag_queries = 0

    def _tag_payload(self, region: str = "") -> str:
        """Mimic the real tagging API's REGION scoping.

        `resourcegroupstaggingapi` returns only resources "located in the specified AWS
        Region", and CloudFront is not a regional resource -- its own docs say Tag Editor
        and Resource Groups do not cover it. A spy that answered the same rows for every
        region hid the defect that made a healthy drive look half-built outside us-east-1,
        so the distribution row is served only for us-east-1, and `cf_tags_visible=False`
        models the stricter reading where the tagging API never reports it at all.
        """
        if not self.drive:
            return json.dumps({"ResourceTagMappingList": []})
        rows: list[dict] = []
        if region != "us-east-1":
            names = self.buckets if self.buckets is not None else [BUCKET]
            rows += [{"ResourceARN": f"arn:aws:s3:::{n}"} for n in names]
        if region == "us-east-1" and self.cf_tags_visible:
            rows.append({"ResourceARN": f"arn:aws:cloudfront::1:distribution/{DIST}"})
        return json.dumps({"ResourceTagMappingList": rows})

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(pd.engine, "distribution_status", lambda d, p: (self.status, DOMAIN))
        monkeypatch.setattr(pd.engine, "invalidate", lambda d, p: self.invalidated.append(d))
        monkeypatch.setattr(pd.engine, "_harden_bucket", lambda b, p, tags: None)
        monkeypatch.setattr(pd.engine, "_oac_name", lambda site_id: f"oac-{site_id}"[:64])
        monkeypatch.setattr(pd.engine, "_trimmed_stderr", lambda err, limit=200: err[:limit])

        def _create_oac(name, profile):
            return "OAC1"

        def _create_distribution(
            bucket, region, oac_id, site_id, profile, tags=None, response_headers_policy_id=""
        ):
            self.created_distributions.append(bucket)
            self.created_site_ids.append(site_id)
            self.created_tags.append(tags)
            self.policy_ids.append(response_headers_policy_id)
            return {"id": "E999", "arn": "arn:aws:cloudfront::1:distribution/E999"}

        monkeypatch.setattr(pd.engine, "create_oac", _create_oac)
        monkeypatch.setattr(pd.engine, "create_distribution", _create_distribution)

        def _checked(args, profile, action="", timeout=0):
            self.checked.append(list(args))
            if args[:2] == ["resourcegroupstaggingapi", "get-resources"]:
                self.tag_queries += 1
                asked = args[args.index("--region") + 1] if "--region" in args else ""
                return self._tag_payload(asked)
            if args[:2] == ["cloudfront", "list-distributions"]:
                if not self.drive:
                    return json.dumps({"DistributionList": {"Items": []}})
                return json.dumps(
                    {
                        "DistributionList": {
                            "Items": [
                                {"Id": DIST, "ARN": f"arn:aws:cloudfront::1:distribution/{DIST}"}
                            ]
                        }
                    }
                )
            if args[:2] == ["cloudfront", "list-tags-for-resource"]:
                if not self.cf_own_tags:
                    return json.dumps({"Tags": {"Items": []}})
                return json.dumps(
                    {"Tags": {"Items": [{"Key": pd.TAG_DRIVE, "Value": pd.TAG_DRIVE_VALUE}]}}
                )
            if args[:2] == ["cloudfront", "list-response-headers-policies"]:
                if not self.existing_policy:
                    return json.dumps({"ResponseHeadersPolicyList": {"Items": []}})
                security_headers: dict = {}
                if self.existing_policy_csp is not None:
                    security_headers["ContentSecurityPolicy"] = {
                        "ContentSecurityPolicy": self.existing_policy_csp,
                        "Override": self.existing_policy_override,
                    }
                return json.dumps(
                    {
                        "ResponseHeadersPolicyList": {
                            "Items": [
                                {
                                    "ResponseHeadersPolicy": {
                                        "Id": POLICY_EXISTING,
                                        "ResponseHeadersPolicyConfig": {
                                            "Name": pd._SANDBOX_POLICY_NAME,
                                            "SecurityHeadersConfig": security_headers,
                                        },
                                    }
                                }
                            ]
                        }
                    }
                )
            if args[:2] == ["cloudfront", "create-response-headers-policy"]:
                return json.dumps({"ResponseHeadersPolicy": {"Id": POLICY_NEW}})
            if args[:2] == ["cloudfront", "get-distribution-config"]:
                return json.dumps({"DistributionConfig": {"Enabled": self.enabled}})
            return "{}"

        monkeypatch.setattr(pd.engine, "_checked", _checked)

        def _run_aws(args, profile):
            if args[:2] == ["s3api", "head-object"]:
                key = args[args.index("--key") + 1]
                if self.head_error:
                    return 1, "", self.head_error
                if self.existing_keys is not None and key not in self.existing_keys:
                    return 1, "", "An error occurred (404) ... Not Found"
                meta = {"sha256": self.stored_digest} if self.stored_digest else {}
                return 0, json.dumps({"ContentType": self.content_type, "Metadata": meta}), ""
            if args[:2] == ["s3api", "create-bucket"]:
                self.created_buckets.append(args[args.index("--bucket") + 1])
                return 0, "{}", ""
            return 0, "{}", ""

        monkeypatch.setattr(pd.engine, "run_aws", _run_aws)

    # helpers a test can read

    def arg(self, verb: str, flag: str) -> str:
        for a in self.checked:
            if a[:2] == ["s3api", verb]:
                return a[a.index(flag) + 1]
        raise AssertionError(f"no {verb} recorded")

    def put_keys(self) -> list[str]:
        return [a[a.index("--key") + 1] for a in self.checked if a[:2] == ["s3api", "put-object"]]

    def keys_deleted(self) -> list[str]:
        return [
            a[a.index("--key") + 1] for a in self.checked if a[:2] == ["s3api", "delete-object"]
        ]

    def keys_copied_to(self) -> list[str]:
        return [a[a.index("--key") + 1] for a in self.checked if a[:2] == ["s3api", "copy-object"]]

    def bucket_policy(self) -> dict:
        return json.loads(self.arg("put-bucket-policy", "--policy"))

    def harden_tagsets(self) -> list[str]:
        return [a for a in self.checked if a[:1] == ["resourcegroupstaggingapi"]]

    def created_policies(self) -> list[list[str]]:
        """Every create-response-headers-policy call recorded."""
        return [
            a for a in self.checked if a[:2] == ["cloudfront", "create-response-headers-policy"]
        ]


def _publish(provider, payload, visibility="public", shared_with=None):
    return asyncio.run(
        provider.publish(
            file_path=payload,
            content_type="text/html",
            title="t",
            summary="",
            tags=[],
            visibility=visibility,
            shared_with=shared_with or [],
        )
    )


# ── shape / contract ─────────────────────────────────────────────────────────


def test_provider_implements_the_whole_abc():
    """Instantiation is the assertion: an unimplemented abstractmethod raises here."""
    assert pd.PersonalDriveProvider().name == pd.DEFAULT_PROVIDER


def test_no_per_account_provider_key_is_registered(monkeypatch):
    """A provider id is validated at the HTTP boundary against ^[a-z0-9-]{1,32}$, so a
    colon-qualified per-account key would be listed in the picker and rejected on
    submit. A row that 400s when clicked is worse than no row."""
    import re as _re

    from kiro_crew.dashboard.handlers import artifacts as art_handlers

    _stub_profiles(monkeypatch, ["alpha", "beta"], default="alpha")
    pd.register_public_edition_providers()
    gate = art_handlers._ARTIFACT_PROVIDER_RE
    assert isinstance(gate, _re.Pattern)
    for key in pp._FACTORIES:
        assert gate.match(key), f"registered key {key!r} cannot pass the publish gate"


def test_display_name_names_the_destination_not_the_account():
    """The picker already carries a row reading "Publish to public web (your AWS)" for the
    per-artifact deploy path. Repeating "your AWS" here made the two rows read as variants
    of one destination; which account it lands in belongs in the remedy text. "drive"
    alone read as PRIVATE storage, so the label keeps a legible PUBLIC promise."""
    label = pd.PersonalDriveProvider().display_name
    assert label == "Public web (shared drive)"
    assert "aws" not in label.lower()
    assert "public" in label.lower()


def test_the_remedy_text_names_the_action_that_makes_it_available():
    """The panel renders this in place for an unavailable destination, so it has to name
    WHICH action fixes it -- a provider is the only thing that knows."""
    hint = pd.PersonalDriveProvider().install_hint
    assert hint and "register" in hint.lower()


def test_display_name_is_vendor_neutral():
    """The seam mandates user-facing strings come from display_name; it must not
    name a third-party product."""
    label = pd.PersonalDriveProvider().display_name.lower()
    assert "drive" in label
    for vendor in ("google", "dropbox", "onedrive", "box.com"):
        assert vendor not in label


def test_sharing_capability_is_declared():
    """The orchestration gates its visibility reconciliation on this capability, so
    withholding it would let a re-publish-as-private push new bytes into the SERVED
    prefix and skip the move -- content still public, record still saying public."""
    assert Capability.SHARING in pd.PersonalDriveProvider().capabilities()


def test_declaring_sharing_does_not_imply_a_grant_list(monkeypatch):
    """SHARING means 'can change visibility', not 'has a per-principal grant list' --
    the grant list is refused by sharing_model and by update_sharing itself."""
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy().install(monkeypatch)
    assert pd.PersonalDriveProvider().sharing_model().supports_shared is False
    with pytest.raises(CapabilityNotSupportedError):
        asyncio.run(
            pd.PersonalDriveProvider().update_sharing(
                external_id="abc~alpha", visibility="shared", shared_with=["someone"]
            )
        )


def test_sharing_model_declares_no_grant_list():
    m = pd.PersonalDriveProvider().sharing_model()
    assert m.supports_public and m.supports_private
    assert not m.supports_shared
    assert m.principal_kind == "none"
    assert not m.supports_roles


def test_sync_model_is_a_token_guarded_mirror():
    m = pd.PersonalDriveProvider().sync_model()
    assert m.collab_mode == "mirror"
    assert m.concurrency == "token"


# ── the drive is invisible to the deploy surface ──────────────────────────────


def test_drive_does_not_carry_the_deploy_site_tag(monkeypatch, payload):
    """The deploy engine groups sites by kirocrew:site and SKIPS resources without it.
    Carrying that tag would list the pooled drive as an ordinary one-off site, where a
    recall empties it and a destroy deletes it -- wiping every published artifact."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    tagsets: list[str] = []
    monkeypatch.setattr(pd.engine, "_harden_bucket", lambda b, p, tags: tagsets.append(tags))
    _publish(pd.PersonalDriveProvider(), payload)
    assert tagsets, "the drive's bucket must be tagged"
    for tags in tagsets:
        assert pd.TAG_DRIVE in tags
        assert pd.engine.TAG_SITE not in tags


def test_discovery_filters_on_the_drive_tag_alone(monkeypatch, payload):
    """One private tag identifies the drive, passed as a SINGLE filter -- requiring a
    second tag in the same call would make discovery depend on how repeated
    --tag-filters flags combine."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    query = next(a for a in spy.checked if a[:2] == ["resourcegroupstaggingapi", "get-resources"])
    filters = [query[i + 1] for i, tok in enumerate(query) if tok == "--tag-filters"]
    assert filters == [f"Key={pd.TAG_DRIVE},Values={pd.TAG_DRIVE_VALUE}"]


def test_a_drive_outside_us_east_1_is_found_whole(monkeypatch, payload):
    """The tagging API answers per REGION and does not cover CloudFront, so the regional
    call that finds the bucket cannot see the distribution. Reporting a half-built drive
    there sends a HEALTHY drive down partial-create recovery: a second distribution is
    built and the bucket policy is re-pointed at it, so every link already handed out
    answers 403 -- and the second tagged distribution then makes discovery ambiguous
    forever. The default region is not us-east-1, so this was the normal path."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()  # region defaults to us-west-2
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert spy.created_distributions == [], "a healthy drive must not be rebuilt"
    regions = [
        a[a.index("--region") + 1]
        for a in spy.checked
        if a[:2] == ["resourcegroupstaggingapi", "get-resources"] and "--region" in a
    ]
    assert "us-east-1" in regions, "the distribution must be looked for where CloudFront lives"


def test_a_distribution_invisible_to_the_tagging_api_is_found_through_cloudfront(
    monkeypatch, payload
):
    """CloudFront's own docs say Tag Editor and Resource Groups do not cover it, so the
    tag query may report nothing in EVERY region. Discovery then falls back to CloudFront's
    own tag API -- the path CloudFront documents -- rather than concluding the drive is
    half-built and rebuilding it."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(cf_tags_visible=False)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert spy.created_distributions == [], "a healthy drive must not be rebuilt"
    assert any(a[:2] == ["cloudfront", "list-tags-for-resource"] for a in spy.checked)


def test_a_genuinely_missing_distribution_is_still_created(monkeypatch, payload):
    """The partial-create recovery path is still needed for the state it was written for:
    a prior run tagged the bucket and died before the distribution existed. Neither tag
    source reports one, and CloudFront has none carrying the tag."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(cf_tags_visible=False, cf_own_tags=False)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert spy.created_distributions == [BUCKET], "the existing bucket must be reused"


def test_two_drives_in_one_account_refuse_to_guess(monkeypatch, payload):
    """What discovery returns is what gets written to and deleted from, so an
    ambiguous match must fail loud and name the candidates."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(buckets=[BUCKET, f"{pd.BUCKET_PREFIX}000000000000"])
    spy.install(monkeypatch)
    with pytest.raises(PublishError) as exc:
        _publish(pd.PersonalDriveProvider(), payload)
    msg = str(exc.value)
    assert "more than one personal drive" in msg
    assert BUCKET in msg and f"{pd.BUCKET_PREFIX}000000000000" in msg
    assert "nothing was touched" in msg


def test_discovery_ignores_a_bucket_outside_the_naming_scheme(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(buckets=["someone-elses-bucket"])
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    # No usable bucket found -> the create branch ran instead of writing to a stranger's.
    assert spy.created_buckets and spy.created_buckets[0].startswith(pd.BUCKET_PREFIX)


# ── account resolution and BINDING ───────────────────────────────────────────


def test_no_registered_account_is_unavailable_with_a_remedy(monkeypatch):
    _stub_profiles(monkeypatch, [])
    p = pd.PersonalDriveProvider()
    assert p.available() is False
    with pytest.raises(pd.PublishUnavailableError) as exc:
        p._resolve_profile()
    assert "register a profile" in str(exc.value).lower()


def test_unknown_bound_account_names_the_registered_ones(monkeypatch):
    _stub_profiles(monkeypatch, ["alpha", "beta"], default="alpha")
    with pytest.raises(pd.PublishUnavailableError) as exc:
        pd.PersonalDriveProvider()._resolve_profile("typo")
    msg = str(exc.value)
    assert "typo" in msg and "alpha" in msg and "beta" in msg


def test_a_new_publish_follows_the_registry_default(monkeypatch):
    _stub_profiles(monkeypatch, ["alpha", "beta"], default="beta")
    assert pd.PersonalDriveProvider()._resolve_profile() == ("beta", "us-west-2")


def test_publish_binds_the_resolved_account_into_the_external_id(monkeypatch, payload):
    """The seam persists a provider KEY and re-resolves it later, so a destination
    whose key means 'whatever is default now' cannot locate its own past publishes."""
    _stub_profiles(monkeypatch, ["alpha", "beta"], default="alpha")
    _EngineSpy().install(monkeypatch)
    res = _publish(pd.PersonalDriveProvider(), payload)
    _key, bound = pd.split_external_id(res.external_id)
    assert bound == "alpha"


def test_the_public_url_never_contains_the_account_name(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["secret-acct"], default="secret-acct")
    _EngineSpy().install(monkeypatch)
    res = _publish(pd.PersonalDriveProvider(), payload)
    assert res.view_url
    assert "secret-acct" not in res.view_url


def test_the_object_key_never_contains_the_account_name(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["secret-acct"], default="secret-acct")
    spy = _EngineSpy()
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert all("secret-acct" not in k for k in spy.put_keys())


def test_changing_the_default_does_not_move_an_existing_publication(monkeypatch, payload):
    """Publish under alpha, make beta the default, then unpublish: the delete must run
    against ALPHA. Otherwise the local record is cleared while alpha's object stays
    publicly served -- the un-recallable state this destination must never produce."""
    reg = _stub_profiles(monkeypatch, ["alpha", "beta"], default="alpha")
    spy = _EngineSpy()
    spy.install(monkeypatch)
    provider = pd.PersonalDriveProvider()
    res = _publish(provider, payload)

    seen: list[str] = []
    real_resolve = pd.profiles.resolve_profile
    monkeypatch.setattr(
        pd.profiles,
        "resolve_profile",
        lambda requested="": (seen.append(requested), real_resolve(requested))[1],
    )
    reg["default"] = "beta"
    asyncio.run(provider.unpublish(external_id=res.external_id))
    assert "alpha" in seen, "the removal must target the account the publish bound"


def test_an_unbound_legacy_id_falls_back_to_the_default(monkeypatch):
    """An id minted before binding existed has no separator; the pre-binding
    behaviour (registry default) is the right fallback, not a crash."""
    _stub_profiles(monkeypatch, ["alpha"], default="alpha")
    assert pd.PersonalDriveProvider()._profile_for("plainid") == ("alpha", "us-west-2")


def test_available_does_not_reach_aws(monkeypatch):
    """available() gates whether the destination is offered at all, so it must not
    make a network call."""
    _stub_profiles(monkeypatch, ["alpha"])

    def _boom(*a, **k):  # pragma: no cover - fails the test if called
        raise AssertionError("available() reached AWS")

    monkeypatch.setattr(pd.engine, "run_aws", _boom)
    monkeypatch.setattr(pd.engine, "_checked", _boom)
    assert pd.PersonalDriveProvider().available() is True


# ── visibility semantics ─────────────────────────────────────────────────────


def test_shared_visibility_is_refused_not_downgraded(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["alpha"])
    with pytest.raises(CapabilityNotSupportedError):
        _publish(pd.PersonalDriveProvider(), payload, visibility="shared")


def test_a_grant_list_is_refused_even_when_visibility_is_public(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["alpha"])
    with pytest.raises(CapabilityNotSupportedError):
        _publish(pd.PersonalDriveProvider(), payload, shared_with=["someone"])


def test_private_publish_yields_no_browsable_url(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    res = _publish(pd.PersonalDriveProvider(), payload, visibility="private")
    assert res.view_url == ""
    assert not spy.invalidated, "a private object is not served, so nothing to invalidate"
    assert spy.put_keys()[0].startswith(pd.PRIVATE_PREFIX)


# ── the load-bearing upload details ──────────────────────────────────────────


def test_upload_declares_the_content_type(monkeypatch, payload):
    """Without an explicit content type S3 stores binary/octet-stream, and the
    distribution's nosniff header then makes a browser download an HTML page instead
    of rendering it."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert spy.arg("put-object", "--content-type") == "text/html"


def test_upload_stores_the_content_digest_as_metadata(monkeypatch, payload):
    """The stored digest is the whole content of the token-guarded concurrency this
    provider declares; without it a push cannot tell the object changed."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    res = _publish(pd.PersonalDriveProvider(), payload)
    assert spy.arg("put-object", "--metadata") == f"sha256={res.concurrency_token}"


def test_public_publish_returns_a_link_and_does_not_invalidate(monkeypatch, payload):
    """No purge on a first publish: the key is freshly minted so no edge has served it.
    Issuing one anyway would add the only failure mode this call otherwise lacks -- the
    object is already uploaded, so a throttled invalidation would report a failed
    publish while leaving that object behind with no record pointing at it."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    res = _publish(pd.PersonalDriveProvider(), payload)
    key_part, _ = pd.split_external_id(res.external_id)
    assert res.view_url == f"https://{DOMAIN}/{pd.PUBLIC_PREFIX}{key_part}"
    assert res.concurrency_token, "a push guard token is required"
    assert spy.invalidated == []
    assert spy.put_keys() == [f"{pd.PUBLIC_PREFIX}{key_part}"]


def test_external_id_is_generated_not_derived(monkeypatch, payload):
    """The id is half of an unguessable public URL, so two publishes of the same
    bytes must not collide."""
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy().install(monkeypatch)
    a = _publish(pd.PersonalDriveProvider(), payload).external_id
    b = _publish(pd.PersonalDriveProvider(), payload).external_id
    assert a != b


def test_oversized_artifact_is_refused_with_its_size(monkeypatch, tmp_path):
    from kiro_crew.artifacts import MAX_CONTENT_BYTES

    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy().install(monkeypatch)
    big = tmp_path / "big.html"
    big.write_bytes(b"x" * (MAX_CONTENT_BYTES + 1))
    with pytest.raises(PublishError) as exc:
        _publish(pd.PersonalDriveProvider(), str(big))
    assert str(MAX_CONTENT_BYTES) in str(exc.value)


# ── the sandbox response-headers policy: verify-before-reuse, else fail closed ──


def test_same_name_policy_with_wrong_csp_fails_closed(monkeypatch, payload):
    """An agent holding the user's AWS creds can pre-create a policy under this
    account-global name WITHOUT the sandbox CSP. Reusing it on the name alone would
    silently drop the opaque-origin sandbox and let published docs share an origin, so a
    name match with a wrong/absent CSP must FAIL CLOSED and create NOTHING."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False, existing_policy=True, existing_policy_csp="default-src 'self'")
    spy.install(monkeypatch)
    with pytest.raises(PublishError) as excinfo:
        _publish(pd.PersonalDriveProvider(), payload)
    msg = str(excinfo.value)
    assert pd._SANDBOX_POLICY_NAME in msg
    assert POLICY_EXISTING in msg
    # Failed closed: no distribution built, no duplicate policy created.
    assert spy.created_distributions == []
    assert spy.created_policies() == []


def test_same_name_policy_with_exact_csp_is_reused(monkeypatch, payload):
    """Genuine find-or-create: a same-named policy carrying the EXACT sandbox CSP with
    Override true is reused -- its id reaches create_distribution and no new policy is
    created."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False, existing_policy=True)  # defaults to exact CSP + Override true
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert spy.created_policies() == []  # reused, not re-created
    assert POLICY_EXISTING in spy.policy_ids  # the reused id was handed to create_distribution


def test_same_name_policy_with_override_false_fails_closed(monkeypatch, payload):
    """A CSP that matches the directive but is not enforced (Override false) does not
    isolate anything, so it must fail closed exactly like a wrong CSP -- not be reused."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False, existing_policy=True, existing_policy_override=False)
    spy.install(monkeypatch)
    with pytest.raises(PublishError) as excinfo:
        _publish(pd.PersonalDriveProvider(), payload)
    assert pd._SANDBOX_POLICY_NAME in str(excinfo.value)
    assert spy.created_distributions == []
    assert spy.created_policies() == []


def test_no_existing_policy_is_created_as_before(monkeypatch, payload):
    """With no same-named policy present, the sandbox policy is created and its new id
    flows into the distribution -- the find-or-create create branch is intact."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False, existing_policy=False)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert len(spy.created_policies()) == 1
    assert POLICY_NEW in spy.policy_ids


# ── the two dead-link states, and HOW they are reported ──────────────────────
def test_still_rolling_out_is_reported_on_a_public_publish(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy(status="InProgress").install(monkeypatch)
    res = _publish(pd.PersonalDriveProvider(), payload)
    assert "still rolling out" in res.notice.lower()
    # The raw AWS state token (e.g. "InProgress") is deliberately NOT surfaced in user
    # copy -- the sentence already explains the state without leaking the literal.
    assert "InProgress" not in res.notice
    assert "nothing was lost" in res.notice.lower()


def test_externally_disabled_is_reported_on_a_public_publish(monkeypatch, payload):
    """An account security control can disable the distribution; the resulting dead
    link is invisible from the URL, so the publish must say so -- on the result, because
    the content is already uploaded and raising would strand it."""
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy(enabled=False).install(monkeypatch)
    res = _publish(pd.PersonalDriveProvider(), payload)
    assert "DISABLED" in res.notice
    assert "not a state Kiro Crew sets" in res.notice


def test_an_unreadable_rollout_state_still_returns_the_handle(monkeypatch, payload):
    """Making the VERDICT non-fatal was not enough -- the probe itself reaches CloudFront,
    so a throttle or a permissions gap raised after the object was already uploaded and the
    orchestration, which records the publication only on return, left it public with no
    withdrawal handle. Not knowing the state is a thing to report, not a reason to discard
    the handle."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    # Drive resolution reads the status too, so failing every call would abort BEFORE the
    # upload -- a different (and already covered) path. Only the post-upload read fails.
    real_status = pd.engine.distribution_status
    calls = {"n": 0}

    def _status(dist_id, profile):
        calls["n"] += 1
        if spy.put_keys():
            raise pd.engine.AWSError("throttled")
        return real_status(dist_id, profile)

    monkeypatch.setattr(pd.engine, "distribution_status", _status)
    res = _publish(pd.PersonalDriveProvider(), payload)
    assert res.external_id, "the handle for the uploaded object must come back"
    assert "did not report" in res.notice
    assert spy.put_keys(), "the object was uploaded before the state was read"


def test_an_upload_bounds_how_long_an_edge_may_serve_it(monkeypatch, payload):
    """Withdrawal deletes the object and SUBMITS an invalidation without awaiting it, so
    the invalidation is the fast path and this header is the guarantee. Without it objects
    inherit the distribution's cache policy default (Managed-CachingOptimized, 24 hours),
    so an edge that had served a public artifact could keep serving it for a day after the
    user withdrew it and was told it was gone."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert spy.arg("put-object", "--cache-control") == f"max-age={pd._EDGE_MAX_AGE_SECONDS}"
    # A day-long window is the thing being prevented; keep this honest if the value moves.
    assert pd._EDGE_MAX_AGE_SECONDS <= 900


def test_a_disabled_network_is_also_reported_rather_than_raised(monkeypatch, payload):
    """Same reasoning for the other dead-link state. A DISABLED distribution needs saying
    out loud, but not at the cost of the handle for content already uploaded."""
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy(drive=False, status="Deployed", enabled=False).install(monkeypatch)
    res = _publish(pd.PersonalDriveProvider(), payload)
    assert "DISABLED" in res.notice
    assert res.external_id


def test_removal_is_never_blocked_by_an_unhealthy_network(monkeypatch, payload):
    """THE critical one. The orchestration clears the local record whether or not the
    destination delete happened, so refusing here would report a withdrawal that never
    occurred while the object stayed publicly served."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(status="InProgress", enabled=False, existing_keys=set())
    spy.install(monkeypatch)
    key_part = "abc"
    spy.existing_keys = {f"{pd.PUBLIC_PREFIX}{key_part}"}
    asyncio.run(pd.PersonalDriveProvider().unpublish(external_id=f"{key_part}~alpha"))
    assert spy.keys_deleted() == [
        f"{pd.PUBLIC_PREFIX}{key_part}",
        f"{pd.PRIVATE_PREFIX}{key_part}",
    ]


def test_taking_something_private_is_never_blocked_by_an_unhealthy_network(monkeypatch):
    """Same reasoning: this path WITHDRAWS public content."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(status="InProgress", enabled=False, existing_keys={f"{pd.PUBLIC_PREFIX}abc"})
    spy.install(monkeypatch)
    asyncio.run(
        pd.PersonalDriveProvider().update_sharing(
            external_id="abc~alpha", visibility="private", shared_with=[]
        )
    )
    assert spy.keys_copied_to() == [f"{pd.PRIVATE_PREFIX}abc"]
    assert spy.keys_deleted() == [f"{pd.PUBLIC_PREFIX}abc"]


def test_making_something_public_IS_blocked_by_an_unhealthy_network(monkeypatch):
    """The asymmetry is the point: this path hands out a link, so it must refuse."""
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy(status="InProgress", existing_keys={f"{pd.PRIVATE_PREFIX}abc"}).install(monkeypatch)
    with pytest.raises(PublishError):
        asyncio.run(
            pd.PersonalDriveProvider().update_sharing(
                external_id="abc~alpha", visibility="public", shared_with=[]
            )
        )


def test_unreadable_distribution_config_does_not_block_publishing(monkeypatch, payload):
    """An unreadable enabled-flag is no objection: refusing here would make a
    permissions gap look like a disabled distribution."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    real = pd.engine._checked

    def _checked(args, profile, action="", timeout=0):
        if args[:2] == ["cloudfront", "get-distribution-config"]:
            raise pd.engine.AWSError("AccessDenied")
        return real(args, profile, action=action, timeout=timeout)

    monkeypatch.setattr(pd.engine, "_checked", _checked)
    assert _publish(pd.PersonalDriveProvider(), payload).view_url


# ── head must not turn an AWS failure into "already gone" ────────────────────


def test_a_throttled_head_is_an_error_not_an_absence(monkeypatch):
    """On the paths that DO probe, collapsing every failure to 'absent' would move or
    overwrite the wrong thing. (The removal path deliberately does not probe at all.)"""
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy(head_error="An error occurred (SlowDown) ... Please reduce").install(monkeypatch)
    with pytest.raises(PublishError) as exc:
        asyncio.run(
            pd.PersonalDriveProvider().update_sharing(
                external_id="abc~alpha", visibility="private", shared_with=[]
            )
        )
    assert "nothing was changed" in str(exc.value)


def test_an_expired_token_head_is_an_error_not_an_absence(monkeypatch):
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy(head_error="An error occurred (ExpiredToken)").install(monkeypatch)
    with pytest.raises(PublishError):
        asyncio.run(
            pd.PersonalDriveProvider().update_sharing(
                external_id="abc~alpha", visibility="private", shared_with=[]
            )
        )


def test_removal_never_probes_before_deleting(monkeypatch):
    """DeleteObject is idempotent, so asking first buys nothing -- and it costs
    everything: the probe can fail on a throttle or an expired token, and the
    orchestration clears the local record regardless, so a probe failure would discard
    the only handle to an object that is still served."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(head_error="An error occurred (SlowDown) ... Please reduce")
    spy.install(monkeypatch)
    asyncio.run(pd.PersonalDriveProvider().unpublish(external_id="abc~alpha"))
    assert spy.keys_deleted() == [f"{pd.PUBLIC_PREFIX}abc", f"{pd.PRIVATE_PREFIX}abc"]
    assert spy.invalidated == [DIST]


def test_removal_is_idempotent_when_nothing_is_there(monkeypatch):
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys=set())
    spy.install(monkeypatch)
    asyncio.run(pd.PersonalDriveProvider().unpublish(external_id="gone~alpha"))
    assert spy.keys_deleted() == [f"{pd.PUBLIC_PREFIX}gone", f"{pd.PRIVATE_PREFIX}gone"]
    assert spy.invalidated == [DIST], "the purge must run: it is the last call for this id"


# ── the push guard actually guards ───────────────────────────────────────────


def test_push_refuses_to_overwrite_an_out_of_band_change(monkeypatch, payload):
    """sync_model declares token concurrency, so a remote digest matching neither the
    token we were given nor the bytes we are about to write means someone else changed
    it -- overwriting would destroy that change while reporting success."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc"}, stored_digest="deadbeef" * 8)
    spy.install(monkeypatch)
    out = asyncio.run(
        pd.PersonalDriveProvider().push_version(
            external_id="abc~alpha", file_path=payload, expected_token="cafebabe" * 8
        )
    )
    assert out.conflict is True
    assert out.error
    assert spy.put_keys() == [], "nothing may be written on a conflict"


def test_push_treats_a_missing_digest_as_an_out_of_band_change(monkeypatch, payload):
    """Every object this module writes carries the sha256 metadata, so its ABSENCE means
    the object was replaced by something that is not us -- an upload from the S3 console
    or `aws s3 cp`, neither of which preserves user metadata. A missing digest is less
    evidence that the object is ours than a mismatching one, so it must not fail open."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc"}, stored_digest="")
    spy.install(monkeypatch)
    out = asyncio.run(
        pd.PersonalDriveProvider().push_version(
            external_id="abc~alpha", file_path=payload, expected_token="cafebabe" * 8
        )
    )
    assert out.conflict is True
    assert spy.put_keys() == [], "a metadata-less object may not be overwritten"


def test_push_proceeds_when_the_remote_matches_the_token(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["alpha"])
    token = "cafebabe" * 8
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc"}, stored_digest=token)
    spy.install(monkeypatch)
    out = asyncio.run(
        pd.PersonalDriveProvider().push_version(
            external_id="abc~alpha", file_path=payload, expected_token=token
        )
    )
    assert not out.error and not out.conflict
    assert spy.put_keys() == [f"{pd.PUBLIC_PREFIX}abc"]


def test_push_proceeds_when_the_remote_already_holds_the_new_bytes(monkeypatch, payload):
    """A retried push is not a conflict."""
    _stub_profiles(monkeypatch, ["alpha"])
    digest = pd.hashlib.sha256(Path(payload).read_bytes()).hexdigest()
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc"}, stored_digest=digest)
    spy.install(monkeypatch)
    out = asyncio.run(
        pd.PersonalDriveProvider().push_version(
            external_id="abc~alpha", file_path=payload, expected_token="cafebabe" * 8
        )
    )
    assert not out.conflict


def test_push_version_reports_instead_of_raising(monkeypatch, payload):
    _stub_profiles(monkeypatch, [])  # no account -> would raise inside
    out = asyncio.run(
        pd.PersonalDriveProvider().push_version(
            external_id="abc~alpha", file_path=payload, expected_token=""
        )
    )
    assert out.error, "push_version must never raise; it reports"


def test_push_version_missing_object_says_republish(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy(existing_keys=set()).install(monkeypatch)
    out = asyncio.run(
        pd.PersonalDriveProvider().push_version(
            external_id="gone~alpha", file_path=payload, expected_token=""
        )
    )
    assert "publish it again" in out.error.lower()


def test_a_withdrawal_never_builds_a_replacement_drive(monkeypatch, payload):
    """A discovery miss -- the drive's tag removed by hand, a transient tagging answer --
    used to make unpublish BUILD a bucket, OAC and distribution, delete keys that were
    never in it, and report success, while the original object stayed public and the
    successful report took the local record with it."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    with pytest.raises(pd.PublishError):
        asyncio.run(pd.PersonalDriveProvider().unpublish(external_id="abc~alpha"))
    assert spy.created_buckets == [], "a withdrawal must not provision anything"
    assert spy.created_distributions == []


def test_a_sharing_change_never_builds_a_replacement_drive(monkeypatch, payload):
    """Same rule: only a first publish may create infrastructure."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    with pytest.raises(pd.PublishError):
        asyncio.run(
            pd.PersonalDriveProvider().update_sharing(
                external_id="abc~alpha", visibility="private", shared_with=[]
            )
        )
    assert spy.created_buckets == []
    assert spy.created_distributions == []


def test_a_push_never_builds_a_replacement_drive(monkeypatch, payload):
    """A push reports the missing drive rather than writing into a fresh one, where the
    bytes would be served from nowhere the publication's link points at."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    out = asyncio.run(
        pd.PersonalDriveProvider().push_version(
            external_id="abc~alpha", file_path=payload, expected_token=""
        )
    )
    assert out.error
    assert spy.created_buckets == []
    assert spy.created_distributions == []


def test_taking_private_recopies_the_served_copy_even_when_both_keys_exist(monkeypatch, payload):
    """push_version heads public/<key> first, so whenever both prefixes hold a copy a push
    lands on public/<key> -- which is the SOURCE when taking something private. Skipping
    the copy and deleting it would destroy the pushed bytes, keep the stale private copy,
    and leave the record's token matching neither, so every later push conflicts forever."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PRIVATE_PREFIX}abc", f"{pd.PUBLIC_PREFIX}abc"})
    spy.install(monkeypatch)
    asyncio.run(
        pd.PersonalDriveProvider().update_sharing(
            external_id="abc~alpha", visibility="private", shared_with=[]
        )
    )
    copies = [a for a in spy.checked if a[:2] == ["s3api", "copy-object"]]
    assert copies, "the served copy must be carried across before it is deleted"
    assert copies[0][copies[0].index("--key") + 1] == f"{pd.PRIVATE_PREFIX}abc"
    assert copies[0][copies[0].index("--copy-source") + 1].endswith(f"{pd.PUBLIC_PREFIX}abc")


def test_a_surviving_stale_source_never_overwrites_a_newer_destination(monkeypatch, payload):
    """Both keys present means an earlier move landed and only its cleanup failed. The
    destination is authoritative and may be strictly newer -- a push writes to whichever
    prefix is live -- so copying again would overwrite the pushed version with pre-push
    bytes and report success. Clean up the source instead."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PRIVATE_PREFIX}abc", f"{pd.PUBLIC_PREFIX}abc"})
    spy.install(monkeypatch)
    asyncio.run(
        pd.PersonalDriveProvider().update_sharing(
            external_id="abc~alpha", visibility="public", shared_with=[]
        )
    )
    assert [a for a in spy.checked if a[:2] == ["s3api", "copy-object"]] == []
    deleted = [a[a.index("--key") + 1] for a in spy.checked if a[:2] == ["s3api", "delete-object"]]
    assert deleted == [f"{pd.PRIVATE_PREFIX}abc"], "the stale source is what must go"
    assert spy.invalidated == [DIST]


def test_the_drive_never_claims_the_site_name_a_user_would_destroy(monkeypatch, payload):
    """The engine hard-codes kirocrew:site on the create call and it can only be removed
    afterwards, so a window exists where the drive is listed as a deploy site. Named
    "default" that window meant `destroy default` -- the likeliest destroy anyone runs --
    deleting the distribution every published URL depends on."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert spy.created_site_ids == [pd.SITE_TAG_VALUE]
    assert "default" not in spy.created_site_ids


def test_a_failed_purge_does_not_abort_a_completed_move_to_public(monkeypatch, payload):
    """The object is already served from the new prefix, so raising here would abort
    before the caller records the new visibility -- record private, content served."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PRIVATE_PREFIX}abc"})
    spy.install(monkeypatch)
    monkeypatch.setattr(
        pd.engine, "invalidate", lambda d, p: (_ for _ in ()).throw(pd.engine.AWSError("throttled"))
    )
    asyncio.run(
        pd.PersonalDriveProvider().update_sharing(
            external_id="abc~alpha", visibility="public", shared_with=[]
        )
    )


def test_a_failed_purge_does_abort_a_move_to_private(monkeypatch, payload):
    """The mirror case: the purge is what evicts the copy the edge is still serving, so a
    failure means the withdrawal did not happen and must be reported."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc"})
    spy.install(monkeypatch)
    monkeypatch.setattr(
        pd.engine, "invalidate", lambda d, p: (_ for _ in ()).throw(pd.engine.AWSError("throttled"))
    )
    with pytest.raises(pd.PublishError):
        asyncio.run(
            pd.PersonalDriveProvider().update_sharing(
                external_id="abc~alpha", visibility="private", shared_with=[]
            )
        )


def test_a_publish_makes_no_cloudfront_tag_calls_at_all(monkeypatch, payload):
    """The distribution is born with its final tag set, so neither tag-resource nor
    untag-resource is ever issued. That is what removes the drive's need for
    cloudfront:UntagResource, which the shipped least-privilege policy does not grant."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert [a for a in spy.checked if a[:2] == ["cloudfront", "untag-resource"]] == []
    assert [a for a in spy.checked if a[:2] == ["cloudfront", "tag-resource"]] == []


def test_going_public_caches_the_domain_so_the_seam_can_derive_the_link(monkeypatch, payload):
    """The seam derives the public link from view_url_for right after this call, and on a
    fresh process this is the only call that resolved the domain."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PRIVATE_PREFIX}abc"})
    spy.install(monkeypatch)
    p = pd.PersonalDriveProvider()
    asyncio.run(p.update_sharing(external_id="abc~alpha", visibility="public", shared_with=[]))
    assert p.view_url_for("abc~alpha") == f"https://{DOMAIN}/{pd.PUBLIC_PREFIX}abc"


def test_a_named_account_never_borrows_another_accounts_domain(monkeypatch):
    """An uncached named account must resolve to no link rather than to a link into the
    wrong AWS account's drive, which reads as working and is not."""
    _stub_profiles(monkeypatch, ["alpha", "beta"])
    p = pd.PersonalDriveProvider()
    p._domains["alpha"] = DOMAIN
    assert p.view_url_for("abc~beta") == ""


def test_a_failed_cleanup_does_not_abort_a_completed_move_to_public(monkeypatch, payload):
    """The copy already landed in the served prefix, so the object IS public. Raising
    here would leave the record saying private about content that is being served."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PRIVATE_PREFIX}abc"})
    spy.install(monkeypatch)
    inner = pd.engine._checked

    def _boom(args, profile, action="", timeout=0):
        if args[:2] == ["s3api", "delete-object"]:
            raise pd.engine.AWSError("Throttling: rate exceeded")
        return inner(args, profile, action=action)

    monkeypatch.setattr(pd.engine, "_checked", _boom)
    asyncio.run(
        pd.PersonalDriveProvider().update_sharing(
            external_id="abc~alpha", visibility="public", shared_with=[]
        )
    )
    assert spy.invalidated == [DIST], "the purge must still run"


def test_a_failed_delete_does_abort_a_move_to_private(monkeypatch, payload):
    """The mirror case: taking something private, the source IS the served copy, so a
    failed delete means the withdrawal did not happen and must be reported."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc"})
    spy.install(monkeypatch)
    inner = pd.engine._checked

    def _boom(args, profile, action="", timeout=0):
        if args[:2] == ["s3api", "delete-object"]:
            raise pd.engine.AWSError("Throttling: rate exceeded")
        return inner(args, profile, action=action)

    monkeypatch.setattr(pd.engine, "_checked", _boom)
    with pytest.raises(pd.PublishError):
        asyncio.run(
            pd.PersonalDriveProvider().update_sharing(
                external_id="abc~alpha", visibility="private", shared_with=[]
            )
        )


def test_push_version_types_the_bytes_it_was_handed_not_the_stale_remote_header(
    monkeypatch, payload
):
    """An artifact re-saved as HTML after being published as text arrives here with an
    .html suffix while the remote object still advertises text/plain. Reusing the remote
    header would serve markup as source text under nosniff."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc"}, content_type="text/plain")
    spy.install(monkeypatch)
    out = asyncio.run(
        pd.PersonalDriveProvider().push_version(
            external_id="abc~alpha", file_path=payload, expected_token=""
        )
    )
    assert not out.error
    assert spy.arg("put-object", "--content-type") == "text/html"


def test_push_version_keeps_the_stored_type_for_a_suffix_it_cannot_read(monkeypatch, tmp_path):
    """The fallback is the remote header, so an unrecognised suffix is no worse than
    before rather than being forced to octet-stream."""
    _stub_profiles(monkeypatch, ["alpha"])
    blob = tmp_path / "artifact.bin"
    blob.write_bytes(b"\x00\x01")
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc"}, content_type="image/png")
    spy.install(monkeypatch)
    out = asyncio.run(
        pd.PersonalDriveProvider().push_version(
            external_id="abc~alpha", file_path=str(blob), expected_token=""
        )
    )
    assert not out.error
    assert spy.arg("put-object", "--content-type") == "image/png"


# ── drive creation ───────────────────────────────────────────────────────────


def test_first_publish_builds_the_pooled_drive(monkeypatch, payload):
    """One bucket, one OAC, one distribution -- and the prefix-scoped policy written
    AFTER the distribution, because the policy pins that distribution's ARN."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert len(spy.created_buckets) == 1
    assert spy.created_buckets[0].startswith(pd.BUCKET_PREFIX)
    assert spy.created_distributions == spy.created_buckets
    allow = [s for s in spy.bucket_policy()["Statement"] if s["Effect"] == "Allow"][0]
    assert allow["Condition"]["StringEquals"]["AWS:SourceArn"].endswith("E999")


def test_drive_creation_is_not_repeated_when_it_exists(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=True)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert spy.created_buckets == []
    assert spy.created_distributions == []


def test_creation_rechecks_discovery_under_the_lock(monkeypatch, payload):
    """A concurrent first publish may have built the drive while this call waited;
    creating a second one would leave two tagged drives for discovery to reject."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    calls = {"n": 0}
    real_find = pd.PersonalDriveProvider._find_drive

    def _find(self, profile, region):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # pre-lock: nothing yet
        spy.drive = True  # a rival finished while we waited
        return real_find(self, profile, region)

    monkeypatch.setattr(pd.PersonalDriveProvider, "_find_drive", _find)
    _publish(pd.PersonalDriveProvider(), payload)
    assert calls["n"] >= 2, "discovery must be re-checked after taking the lock"
    assert spy.created_buckets == [], "the rival's drive must be reused, not duplicated"


def test_bucket_creation_failure_is_reported(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    monkeypatch.setattr(
        pd.engine, "run_aws", lambda args, profile: (1, "", "AccessDenied: s3:CreateBucket")
    )
    with pytest.raises(PublishError) as exc:
        _publish(pd.PersonalDriveProvider(), payload)
    assert "storage bucket" in str(exc.value)


# ── bucket policy ────────────────────────────────────────────────────────────


def test_bucket_policy_serves_only_the_public_prefix():
    """The bucket also holds private objects, so a whole-bucket grant would serve
    them."""
    policy = json.loads(
        pd.PersonalDriveProvider()._prefix_scoped_bucket_policy(
            BUCKET, "arn:aws:cloudfront::1:distribution/E1"
        )
    )
    allow = [s for s in policy["Statement"] if s["Effect"] == "Allow"]
    deny = [s for s in policy["Statement"] if s["Effect"] == "Deny"]
    assert len(allow) == 1
    assert allow[0]["Resource"] == f"arn:aws:s3:::{BUCKET}/public/*"
    assert allow[0]["Condition"]["StringEquals"]["AWS:SourceArn"].endswith("E1")
    assert deny and deny[0]["NotResource"] == f"arn:aws:s3:::{BUCKET}/public/*"


def test_bucket_policy_never_grants_the_private_prefix():
    policy = json.loads(pd.PersonalDriveProvider()._prefix_scoped_bucket_policy("b", "arn:x"))
    for stmt in policy["Statement"]:
        if stmt["Effect"] == "Allow":
            assert pd.PRIVATE_PREFIX not in stmt["Resource"]


# ── sharing changes and removal ──────────────────────────────────────────────


def test_making_public_moves_the_object_into_the_served_prefix(monkeypatch):
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PRIVATE_PREFIX}abc"})
    spy.install(monkeypatch)
    asyncio.run(
        pd.PersonalDriveProvider().update_sharing(
            external_id="abc~alpha", visibility="public", shared_with=[]
        )
    )
    assert spy.keys_copied_to() == [f"{pd.PUBLIC_PREFIX}abc"]
    assert spy.keys_deleted() == [f"{pd.PRIVATE_PREFIX}abc"]
    assert spy.invalidated == [DIST]


def test_sharing_change_to_the_current_state_is_a_no_op(monkeypatch):
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc"})
    spy.install(monkeypatch)
    asyncio.run(
        pd.PersonalDriveProvider().update_sharing(
            external_id="abc~alpha", visibility="public", shared_with=[]
        )
    )
    assert spy.keys_copied_to() == []
    assert spy.keys_deleted() == []


def test_sharing_change_on_a_missing_object_says_so(monkeypatch):
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy(existing_keys=set()).install(monkeypatch)
    with pytest.raises(PublishError) as exc:
        asyncio.run(
            pd.PersonalDriveProvider().update_sharing(
                external_id="gone~alpha", visibility="public", shared_with=[]
            )
        )
    assert "no longer in the drive" in str(exc.value)


def test_sharing_change_to_a_grant_list_is_refused(monkeypatch):
    _stub_profiles(monkeypatch, ["alpha"])
    _EngineSpy().install(monkeypatch)
    with pytest.raises(CapabilityNotSupportedError):
        asyncio.run(
            pd.PersonalDriveProvider().update_sharing(
                external_id="abc~alpha", visibility="shared", shared_with=["someone"]
            )
        )


def test_unpublish_covers_both_prefixes(monkeypatch):
    """An artifact may be in either prefix depending on its visibility, and a
    half-removed artifact would leave a live public copy behind."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc", f"{pd.PRIVATE_PREFIX}abc"})
    spy.install(monkeypatch)
    asyncio.run(pd.PersonalDriveProvider().unpublish(external_id="abc~alpha"))
    assert set(spy.keys_deleted()) == {
        f"{pd.PUBLIC_PREFIX}abc",
        f"{pd.PRIVATE_PREFIX}abc",
    }
    assert spy.invalidated == [DIST]


def test_a_failing_delete_still_purges_and_still_tries_the_other_key(monkeypatch, payload):
    """The record is cleared whether or not this call succeeded, so nothing skipped here
    is ever retried. A throttle on the private key must not leave the public copy alive
    in the edge cache with no local handle left to purge it."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    inner = pd.engine._checked

    def _boom(args, profile, action="", timeout=0):
        if args[:2] == ["s3api", "delete-object"] and pd.PRIVATE_PREFIX in args[-1]:
            inner(args, profile, action=action)  # keep it in the recorded call log
            raise pd.engine.AWSError("Throttling: rate exceeded")
        return inner(args, profile, action=action)

    monkeypatch.setattr(pd.engine, "_checked", _boom)
    with pytest.raises(pd.PublishError):
        asyncio.run(pd.PersonalDriveProvider().unpublish(external_id="abc~alpha"))
    deleted = [a[a.index("--key") + 1] for a in spy.checked if a[:2] == ["s3api", "delete-object"]]
    assert deleted == [f"{pd.PUBLIC_PREFIX}abc", f"{pd.PRIVATE_PREFIX}abc"]
    assert spy.invalidated == [DIST], "the purge must run even though a delete failed"


def test_unpublish_without_an_account_reports_the_remedy(monkeypatch):
    _stub_profiles(monkeypatch, [])
    with pytest.raises(pd.PublishUnavailableError):
        asyncio.run(pd.PersonalDriveProvider().unpublish(external_id="abc~alpha"))


# ── optional read-back facets stay unimplemented on purpose ─────────────────


def test_read_back_facets_report_unavailable():
    """The drive cannot read sharing state or content back, so it inherits the
    seam's None defaults rather than pretending."""
    p = pd.PersonalDriveProvider()
    assert asyncio.run(p.fetch_state(external_id="abc")) is None
    assert asyncio.run(p.fetch_content(external_id="abc")) is None


def test_discovery_is_declared_off():
    m = pd.PersonalDriveProvider().discovery_model()
    assert not m.list_mine and not m.list_public and not m.pull_by_id


def test_installable_so_an_account_with_no_drive_is_still_offered():
    assert pd.PersonalDriveProvider().installable() is True


def test_ensure_ready_tracks_availability(monkeypatch):
    _stub_profiles(monkeypatch, [])
    assert asyncio.run(pd.PersonalDriveProvider().ensure_ready()) is False
    _stub_profiles(monkeypatch, ["alpha"])
    assert asyncio.run(pd.PersonalDriveProvider().ensure_ready()) is True


def test_registered_names_survives_an_unreadable_registry(monkeypatch):
    def _boom():
        raise OSError("nope")

    monkeypatch.setattr(pd.profiles, "load_registry", _boom)
    assert pd.PersonalDriveProvider()._registered_names() == []


def test_engine_failure_is_wrapped_not_leaked(monkeypatch, payload):
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    real = pd.engine._checked

    def _checked(args, profile, action="", timeout=0):
        if args[:2] == ["s3api", "put-object"]:
            raise pd.engine.AWSError("throttled")
        return real(args, profile, action=action, timeout=timeout)

    monkeypatch.setattr(pd.engine, "_checked", _checked)
    with pytest.raises(PublishError) as exc:
        _publish(pd.PersonalDriveProvider(), payload)
    assert "rejected this publish" in str(exc.value)


def test_a_retried_take_private_still_purges_the_cache(monkeypatch):
    """Regression: if a previous attempt moved the object and then failed on the
    invalidation, the edge cache still holds the public copy. Returning early because
    the object is 'already private' would make the retry a no-op and leave those bytes
    served while the record says private."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PRIVATE_PREFIX}abc"})
    spy.install(monkeypatch)
    asyncio.run(
        pd.PersonalDriveProvider().update_sharing(
            external_id="abc~alpha", visibility="private", shared_with=[]
        )
    )
    assert spy.keys_copied_to() == [], "nothing to move -- it is already private"
    assert spy.invalidated == [DIST], "but the purge must still be retried"


# ── what the public edition registers ────────────────────────────────────────


def test_registration_makes_the_drive_the_default_destination(monkeypatch):
    _stub_profiles(monkeypatch, ["alpha", "beta"], default="alpha")
    pd.register_public_edition_providers()
    assert isinstance(pp.get_provider(), pd.PersonalDriveProvider)


def test_registration_exposes_exactly_one_destination(monkeypatch):
    """No per-account keys: they cannot pass the publish gate, so listing them would
    put unusable rows in the picker."""
    _stub_profiles(monkeypatch, ["alpha", "beta"], default="alpha")
    pd.register_public_edition_providers()
    ids = {p.name for p in pp.list_providers()}
    assert ids == {pd.DEFAULT_PROVIDER}


def test_provider_list_has_no_duplicate_rows(monkeypatch):
    """``list_providers`` does not dedupe, so registering one provider under two keys
    would render the same destination twice in the picker."""
    _stub_profiles(monkeypatch, ["alpha", "beta"], default="alpha")
    pd.register_public_edition_providers()
    listed = pp.list_providers()
    assert len(listed) == len({p.name for p in listed})
    assert len(listed) == len({p.display_name for p in listed})


def test_registry_key_and_provider_name_agree(monkeypatch):
    """The seam documents ``name`` as the registry key; a mismatch would break any
    caller that round-trips a listed provider's name back through get_provider."""
    _stub_profiles(monkeypatch, ["alpha"], default="alpha")
    pd.register_public_edition_providers()
    for key in list(pp._FACTORIES):
        assert pp.get_provider(key).name == key


def test_the_bucket_policy_lands_before_the_drive_becomes_discoverable(monkeypatch, payload):
    """The drive tag is the only thing _find_drive keys on, so it must not go on until
    the bucket policy is installed. Reversed, ONE PutBucketPolicy failure leaves a drive
    that discovery reports as complete and that serves 403 for the life of the account:
    the next publish finds bucket + distribution, hands out a link, and nothing ever
    goes back to install the policy."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    verbs = [tuple(a[:2]) for a in spy.checked]
    policy = verbs.index(("s3api", "put-bucket-policy"))
    assert policy >= 0, "the served prefix must be granted before a link is handed out"


def test_the_drive_serves_every_artifact_from_an_opaque_origin(monkeypatch, payload):
    """Every artifact on the drive shares ONE domain, so without this they share a browser
    origin and one published document can read what another wrote to localStorage. The
    documents are mutually untrusted authored content, so the CSP sandbox directive gives
    each an opaque origin -- scripts run, storage is unreachable. It has to be a HEADER:
    the sandbox directive is ignored in a <meta> CSP by specification."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    created = [a for a in spy.checked if a[:2] == ["cloudfront", "create-response-headers-policy"]]
    assert created, "the drive must define its own response headers"
    cfg = json.loads(created[0][created[0].index("--response-headers-policy-config") + 1])
    csp = cfg["SecurityHeadersConfig"]["ContentSecurityPolicy"]
    assert csp["ContentSecurityPolicy"].startswith("sandbox ")
    assert csp["Override"] is True
    assert spy.policy_ids == [POLICY_NEW], "the distribution must actually carry it"


def test_a_second_drive_reuses_the_existing_headers_policy(monkeypatch, payload):
    """The policy is account-global, so a second drive or a retried create must find it
    rather than fail on the name collision."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False, existing_policy=True)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert [
        a for a in spy.checked if a[:2] == ["cloudfront", "create-response-headers-policy"]
    ] == []
    assert spy.policy_ids == [POLICY_EXISTING]


def test_a_first_publish_uploads_before_it_reports_the_rollout(monkeypatch, payload):
    """A distribution is always InProgress in the minutes after it is created, so
    asserting first made the very first publish on a fresh account fail with nothing
    written -- leaving a tagged distribution over an empty origin, the dangling-origin
    shape this destination exists to avoid. The object goes up, then the state is
    reported, so the same link works the moment the rollout finishes.

    And the report is a NOTICE, not a raise. The orchestration records the publication
    only when publish returns, so raising after the upload stranded a public object with
    no withdrawal handle -- while the message claimed "nothing was lost"."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False, status="InProgress")
    spy.install(monkeypatch)
    res = _publish(pd.PersonalDriveProvider(), payload)
    assert spy.put_keys(), "the origin must not be left empty"
    assert any(k.startswith(pd.PUBLIC_PREFIX) for k in spy.put_keys()), spy.put_keys()
    assert "still rolling out" in res.notice
    assert res.external_id, "the handle for the uploaded object must come back"


def test_a_failed_purge_after_a_push_still_returns_the_new_token(monkeypatch, payload):
    """The bytes are already uploaded, so reporting failure would leave the caller holding
    the previous token while the object carries the new digest -- and a stored digest
    matching neither the token nor the next payload is a conflict, so every later push
    would conflict forever with nothing able to reconcile it."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(existing_keys={f"{pd.PUBLIC_PREFIX}abc"})
    spy.install(monkeypatch)
    monkeypatch.setattr(
        pd.engine, "invalidate", lambda d, p: (_ for _ in ()).throw(pd.engine.AWSError("throttled"))
    )
    out = asyncio.run(
        pd.PersonalDriveProvider().push_version(
            external_id="abc~alpha", file_path=payload, expected_token=""
        )
    )
    assert not out.error, "a failed purge must not discard the token"
    assert out.concurrency_token


def test_a_denied_policy_reassert_blocks_a_link_but_not_a_withdrawal(monkeypatch, payload):
    """The two objections to this call are both real and they point opposite ways, so the
    rule is the one the serving assertion already uses. Handing out a link while the bucket
    is unconfirmed promises a URL that may 403; raising on a withdrawal instead turns an
    optional permission into one every call needs, so a profile without
    s3:PutBucketPolicy would lose the ability to withdraw from a drive that works."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    inner = pd.engine._checked

    def _boom(args, profile, action="", timeout=0):
        if args[:2] == ["s3api", "put-bucket-policy"]:
            raise pd.engine.AWSError("AccessDenied: s3:PutBucketPolicy")
        return inner(args, profile, action=action)

    monkeypatch.setattr(pd.engine, "_checked", _boom)
    with pytest.raises(pd.PublishError):
        _publish(pd.PersonalDriveProvider(), payload)
    asyncio.run(pd.PersonalDriveProvider().unpublish(external_id="abc~alpha"))
    assert spy.invalidated == [DIST], "the withdrawal must still complete"


def test_a_reused_drive_reinstalls_the_bucket_policy(monkeypatch, payload):
    """Both resources are tagged, so discovery finds the drive the moment it exists. If a
    first publish lost PutBucketPolicy to a denial or a throttle, the reuse path would
    otherwise hand out links to a bucket CloudFront cannot read -- 403 for the life of the
    account, reported as success. Re-installing makes that state self-healing, and also
    repairs a policy deleted by hand."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy()
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    policies = [a for a in spy.checked if a[:2] == ["s3api", "put-bucket-policy"]]
    assert policies, "a publish onto an existing drive must re-assert the policy"
    assert pd.PUBLIC_PREFIX in policies[0][policies[0].index("--policy") + 1]


def test_the_distribution_is_born_with_the_drive_tag(monkeypatch, payload):
    """Without the drive's own tag the distribution is invisible to this module's
    discovery, and because a missing distribution reads as 'no drive yet', every publish
    would build another one."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert spy.created_tags, "the distribution must be created with an explicit tag set"
    keys = {t["Key"] for t in spy.created_tags[0]}
    assert pd.TAG_DRIVE in keys


def test_the_distribution_is_never_born_with_the_deploy_site_tag(monkeypatch, payload):
    """Carrying kirocrew:site the pooled distribution shows on the deploy site surface,
    where a destroy deletes the drive every artifact lives on. Supplying the final tags on
    create is what removes the window a tag-then-untag left open, and with it the
    cloudfront:UntagResource permission the shipped least-privilege policy never grants."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    keys = {t["Key"] for t in spy.created_tags[0]}
    assert pd.engine.TAG_SITE not in keys
    assert [a for a in spy.checked if a[:2] == ["cloudfront", "untag-resource"]] == []
    assert [a for a in spy.checked if a[:2] == ["cloudfront", "tag-resource"]] == []


def test_a_second_publish_reuses_the_distribution(monkeypatch, payload):
    """The regression the re-tag prevents: discovery must find the distribution it just
    created, or each publish leaks another one and none of them is ever serving."""
    _stub_profiles(monkeypatch, ["alpha"])
    spy = _EngineSpy(drive=False)
    spy.install(monkeypatch)
    _publish(pd.PersonalDriveProvider(), payload)
    assert len(spy.created_distributions) == 1
    spy.drive = True  # the drive now exists and carries the drive tag
    _publish(pd.PersonalDriveProvider(), payload)
    assert len(spy.created_distributions) == 1, "a second publish must not build another"


def test_registration_survives_an_unreadable_registry(monkeypatch):
    """Registration runs at boot; a broken registry must not take the gateway down."""

    def _boom():
        raise OSError("registry unreadable")

    monkeypatch.setattr(pd.profiles, "load_registry", _boom)
    pd.register_public_edition_providers()
    assert set(pp._FACTORIES) == {pd.DEFAULT_PROVIDER}


def test_the_artifact_store_is_never_imported_at_module_scope():
    """Ratchet, not decoration. ``kiro_crew.artifacts`` resolves config while it is
    still importing, which installs the platform context, which registers this provider,
    which imports this module. A module-scope ``from kiro_crew.artifacts import ...``
    therefore ran against a half-initialised module and every publish died on a
    circular-import ImportError -- it reds the sandbox shard, not this file, so only a
    check like this one stops it coming back as a tidy-up."""
    import ast

    source = Path(pd.__file__).with_suffix(".py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders = [
        node.module
        for node in tree.body  # module scope ONLY -- nested imports are the fix
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("kiro_crew.artifacts")
    ]
    assert offenders == [], f"module-scope import of {offenders} recreates the cycle"


def test_public_edition_registry_wires_the_drive(monkeypatch):
    """The platform seam is the only place the edition names a concrete provider."""
    from kiro_crew.platform.defaults import DefaultPublishRegistry

    _stub_profiles(monkeypatch, ["alpha"])
    DefaultPublishRegistry().register_publish_providers()
    assert isinstance(pp.get_provider(), pd.PersonalDriveProvider)


# ── view_url ─────────────────────────────────────────────────────────────────


def test_view_url_is_empty_until_the_domain_is_known():
    """The seam treats an empty url as 'published, no link' rather than failure."""
    assert pd.PersonalDriveProvider().view_url_for("abc~alpha") == ""


def test_view_url_uses_the_served_prefix_and_the_random_half_only():
    p = pd.PersonalDriveProvider()
    p._domains["alpha"] = "d9.cloudfront.net"
    assert p.view_url_for("abc~alpha") == "https://d9.cloudfront.net/public/abc"


def test_external_id_round_trips():
    ident = pd.make_external_id("alpha")
    key_part, bound = pd.split_external_id(ident)
    assert bound == "alpha"
    assert key_part and pd._ID_SEP not in key_part


def test_a_profile_containing_the_separator_cannot_corrupt_the_key():
    """The random half comes FIRST precisely so the key survives any profile name."""
    ident = pd.make_external_id("we~ird")
    key_part, bound = pd.split_external_id(ident)
    assert bound == "we~ird"
    assert pd._ID_SEP not in key_part
