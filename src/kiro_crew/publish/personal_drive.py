"""The personal cloud drive: a publish destination in the operator's own AWS account.

One private bucket plus ONE CloudFront distribution, shared by every artifact this
provider publishes. Each artifact is a single object under a served key prefix, so a
publish after the first is an upload plus an invalidation -- seconds, no control-plane
wait, and no new billable resource per artifact.

Why pooled and not one distribution per artifact
------------------------------------------------
A distribution per artifact costs a cold create (minutes of propagation, during which
the URL the user was just handed does not resolve), consumes an account distribution
quota slot per artifact, and presents a brand-new distribution whose origin is a
still-empty bucket -- a shape security scanners flag as a dangling origin and some
auto-remediate by DISABLING the distribution, which kills the link permanently.

Why the drive carries its OWN tag, not the deploy engine's site tag
-------------------------------------------------------------------
The engine's ``list_sites`` groups resources by ``kirocrew:site`` and SKIPS anything
without it, and ``find_site_by_tag`` resolves a site by that same tag. Tagging the
pooled drive as a site would therefore make it appear in the deploy surface as an
ordinary one-off site, where recall empties it and destroy deletes it -- one action on
an unfamiliar row would wipe every artifact every user of this drive ever published.
The drive tags ``kirocrew:publish-drive`` instead, so the deploy surface cannot see it
and cannot name it. The ambiguity and ownership hardenings that tag-based discovery
needs are implemented here (:meth:`_find_drive`) rather than inherited.

Tagging the DISTRIBUTION takes a second step, and skipping it is not cosmetic.
``engine.create_distribution`` hard-codes ``kirocrew:site=<id>`` on the resource it
creates, so the distribution comes into existence carrying the very tag the drive
avoids. Left that way it is both destroyable from the deploy surface AND invisible to
this module's own discovery -- and because :meth:`_ensure_drive` treats a missing
distribution as "no drive yet", every publish would build another one. So the
distribution is CREATED carrying the drive tag and not the site tag, by handing
``engine.create_distribution`` the final tag set rather than correcting it afterwards.

Account identity travels with the artifact
------------------------------------------
The seam persists a publication's provider as a REGISTRY KEY and re-resolves it later
to unpublish or push. A destination whose key means "whatever account is default right
now" is therefore unsafe: change the default and a later unpublish runs against a
different account, reporting success while the original object stays served. So the
resolved account is bound into the ``external_id`` at publish time -- the one per-
artifact value the seam round-trips -- and every later operation uses that binding
rather than re-resolving the default. The public URL is built from the id's random half
only, so an account label never appears in a link.

Sharing
-------
``visibility="public"`` puts the object under the served prefix; ``"private"`` keeps it
under an unserved one, so a private artifact has no reachable URL at all rather than a
URL that merely lacks a link. ``"shared"`` -- a per-principal grant list -- is not
implemented and is refused explicitly rather than silently downgraded to public.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import threading
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from kiro_crew.deploy import engine, profiles
from kiro_crew.publish_provider import (
    DEFAULT_PROVIDER,
    Capability,
    CapabilityNotSupportedError,
    DiscoveryModel,
    KindSupport,
    PublishError,
    PublishProvider,
    PublishResult,
    PublishUnavailableError,
    PushResult,
    SharingModel,
    SyncModel,
    register_provider,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: The drive's OWN tag key -- deliberately NOT ``engine.TAG_SITE`` (see module docstring).
TAG_DRIVE = "kirocrew:publish-drive"
TAG_DRIVE_VALUE = "default"

# What the drive hands engine.create_distribution as its site id. It is only used for the
# OAC name now that the drive supplies its own tag set, so the kirocrew:site tag is never
# written on the drive's distribution at all -- but a self-describing value is still the
# right one to pass, because the OAC it names is visible in the account's CloudFront
# console and "default" would say nothing about what owns it.
SITE_TAG_VALUE = "kirocrew-personal-drive"

#: Opaque bucket name, and the pattern discovery requires. A distinct prefix from the
#: deploy engine's so the two families are never mistaken for one another.
# The drive's bucket namespace, nested INSIDE the deploy engine's own prefix rather than
# beside it. Sharing the outer prefix is what puts the bucket inside the grant the shipped
# least-privilege policy already writes (`iam.py` grants `kirocrew-web-*`), so the drive
# needs no new bucket-level permission. It does NOT put the bucket into the deploy
# surface's lifecycle: `list_sites` filters on the managed TAG and then skips any resource
# whose site tag is empty, and the drive carries the drive tag instead -- so `recall` and
# `destroy` cannot see it by name or by tag. The extra segment keeps it its own namespace
# for this module's own discovery.
BUCKET_PREFIX = f"{engine.BUCKET_PREFIX}pub-"
_BUCKET_RE = re.compile(rf"^{re.escape(BUCKET_PREFIX)}[0-9a-f]{{12}}$")

#: Objects under this prefix are the only ones the distribution may read.
PUBLIC_PREFIX = "public/"

#: A published-then-made-private object moves here: still stored, no longer served.
PRIVATE_PREFIX = "private/"

#: ``external_id`` is ``<random>~<profile>``. The RANDOM half comes first so a profile
#: name containing the separator cannot corrupt the object key, and the key/URL use only
#: that half -- an account label never reaches a public link.
_ID_SEP = "~"
_EXTERNAL_ID_BYTES = 16

#: Object metadata key holding the content digest, so a push can detect that the remote
#: object changed under it instead of overwriting blindly.
_SHA_META = "sha256"

#: ``Cache-Control: max-age`` written on every uploaded object, which BOUNDS how long a
#: withdrawn artifact can still be served from an edge cache.
#:
#: Withdrawal deletes the object and submits an invalidation, but an invalidation is
#: asynchronous and best-effort: it is submitted, not awaited, because awaiting it would
#: block a user-facing withdrawal for minutes and a timeout would force the caller to
#: report failure after the object is already gone -- the lossy shape this module is
#: careful to avoid. So the invalidation is the FAST path, and this header is the
#: guarantee. Without it, objects inherit the distribution's cache policy default
#: (``Managed-CachingOptimized``, 24 hours), so an edge that had served a public artifact
#: could keep serving it for a day after the user withdrew it and was told it was gone.
#: Five minutes still absorbs a burst of readers on one link while capping that exposure.
_EDGE_MAX_AGE_SECONDS = 300

#: Serializes drive CREATION within this process. Two first publishes racing would each
#: see no drive and each build one, leaving two tagged drives for discovery to reject.
#: This closes the same-process race (a double-clicked publish, two artifacts published
#: together); a race between two machines sharing one account is not covered and shows
#: up as the actionable ambiguity error from :meth:`_find_drive`.
_CREATE_LOCK = threading.Lock()

#: Account-global name of the response-headers policy that opaque-origins published
#: artifacts. Account-global because a policy name is unique per account, which is
#: exactly why the find-or-create below must VERIFY a same-named policy before reusing it
#: (see :meth:`_sandbox_headers_policy`).
_SANDBOX_POLICY_NAME = "kirocrew-publish-drive-sandbox"

#: THE sandbox CSP directive -- one source of truth. It is spelled ONCE here so the create
#: path and the reuse-verification path cannot drift: if a second literal existed in the
#: verify path, a future edit to one copy would let the verifier accept (or a create
#: emit) a policy whose CSP no longer matches, silently defeating the isolation. Delivered
#: as a HEADER (a ``<meta>`` CSP ignores ``sandbox`` by spec), it gives each served
#: document an OPAQUE origin so origin-keyed storage (localStorage/IndexedDB/cookies) is
#: simply unreachable and cannot be shared between mutually-untrusted published documents.
_SANDBOX_CSP = "sandbox allow-scripts allow-popups"


def _sandbox_policy_config() -> dict[str, Any]:
    """The response-headers-policy config this module installs -- one source of truth.

    Both the create path and the reuse-verification path derive the required CSP from the
    SAME :data:`_SANDBOX_CSP` constant via this builder, so there is no second literal to
    drift out of sync with the one the create path writes.
    """
    return {
        "Name": _SANDBOX_POLICY_NAME,
        "Comment": "Opaque-origin sandbox for pooled artifact publishing",
        "SecurityHeadersConfig": {
            "ContentSecurityPolicy": {
                "ContentSecurityPolicy": _SANDBOX_CSP,
                "Override": True,
            },
            "ContentTypeOptions": {"Override": True},
            "StrictTransportSecurity": {
                "AccessControlMaxAgeSec": 31536000,
                "IncludeSubdomains": True,
                "Override": True,
            },
            "ReferrerPolicy": {
                "ReferrerPolicy": "same-origin",
                "Override": True,
            },
        },
    }


# One lock per published artifact, so two mutations of the SAME object cannot interleave.
# Every mutating path is read-then-write against S3 (head to decide, then copy/put/delete),
# and the seam does not serialize per artifact -- its only lock covers cloning -- so a push
# landing between a move's copy and its delete used to have its bytes deleted with the
# source, leaving the record naming a version the drive no longer held and, since the
# stored digest then matched nothing, every later push in permanent conflict.
#
# Keyed on the artifact's own id at this destination, so unrelated artifacts never contend.
# Entries are never evicted: one small lock per artifact touched in this process is cheaper
# than the bugs that come from freeing a lock somebody may be about to take.
_KEY_LOCKS: dict[str, threading.Lock] = {}
_KEY_LOCKS_GUARD = threading.Lock()


def _key_lock(key_part: str) -> threading.Lock:
    """The lock guarding one artifact's remote state."""
    with _KEY_LOCKS_GUARD:
        return _KEY_LOCKS.setdefault(key_part, threading.Lock())


def _serialized(key_part: str, work: Callable[[], _T]) -> _T:
    """Run one artifact's remote mutation under that artifact's own lock.

    Called on the worker thread, never on the event loop, so a blocking lock is the right
    primitive here and cannot stall the gateway.
    """
    with _key_lock(key_part):
        return work()


def _no_profile_message() -> str:
    return (
        "No AWS account is registered yet, so there is nowhere to put the drive. "
        "Register a profile on the Artifact Deploy page first; the drive is then "
        "created in that account on the first publish."
    )


def _unknown_profile_message(requested: str, known: list[str]) -> str:
    if known:
        return (
            f"AWS profile {requested!r} is not registered. "
            f"Registered profiles: {', '.join(sorted(known))}."
        )
    return _no_profile_message()


def make_external_id(profile: str) -> str:
    """A publish handle binding the resolved account to an unguessable key."""
    return f"{secrets.token_hex(_EXTERNAL_ID_BYTES)}{_ID_SEP}{profile}"


def split_external_id(external_id: str) -> tuple[str, str]:
    """Return ``(key_part, bound_profile)``; the profile is ``''`` when unbound.

    An id minted before account binding existed, or one hand-written by a caller, has
    no separator and yields an empty profile -- the caller then falls back to the
    registry default, which is the pre-binding behaviour.
    """
    key_part, _, bound = (external_id or "").partition(_ID_SEP)
    return key_part, bound


class PersonalDriveProvider(PublishProvider):
    """Publish an artifact to the operator's own S3 + CloudFront drive."""

    name = DEFAULT_PROVIDER
    # Names the DESTINATION, not the infrastructure behind it. The picker already
    # carries a row reading "Publish to public web (your AWS)" for the per-artifact
    # deploy path, so repeating "your AWS" here made the two rows read as variants of
    # one thing. Which account it lands in belongs in the unconfigured hint and the
    # setup surface, where it is actionable. "drive" alone read as PRIVATE storage, so
    # beside that sibling row a newcomer could not tell which choice makes content
    # world-readable -- "Public web" keeps the public promise legible.
    display_name = "Public web (shared drive)"
    install_hint = _no_profile_message()

    def __init__(self) -> None:
        """One destination, one registry key.

        There is deliberately no per-account variant: a provider id is validated at the
        HTTP boundary against ``^[a-z0-9-]{1,32}$``, so an account-qualified key could be
        listed in the picker but never submitted -- a row that 400s on click is worse
        than no row. Which account a NEW publish uses is the profile registry's default;
        which account an EXISTING publication belongs to is bound into its
        ``external_id``, because the default can move and a publication must not move
        with it.
        """
        #: Distribution domain, cached per bound profile. A pooled distribution's domain
        #: never changes for the life of that drive, and resolving it costs an AWS call.
        self._domains: dict[str, str] = {}

    # ── account resolution ────────────────────────────────────────────────

    def _registered_names(self) -> list[str]:
        try:
            registry = profiles.load_registry()
        except Exception:  # pragma: no cover - registry unreadable
            return []
        return [str(p.get("name", "")) for p in registry.get("profiles", []) if p.get("name")]

    def _resolve_profile(self, requested: str = "") -> tuple[str, str]:
        """The (profile name, region) to run under, or raise with the fixing action.

        ``requested`` wins over this instance's pinning: it carries the account an
        EXISTING publication was bound to and must be honoured; empty means a NEW
        publish, which follows the registry default.

        ``profiles.resolve_profile`` is the registry's own contract: an empty request
        resolves to the registry default and an unregistered name resolves to ``None``,
        so a publish can only ever run under a registered account.
        """
        wanted = requested
        resolved = profiles.resolve_profile(wanted)
        if resolved is None:
            if wanted:
                raise PublishUnavailableError(
                    _unknown_profile_message(wanted, self._registered_names())
                )
            raise PublishUnavailableError(_no_profile_message())
        name, region = resolved
        return name, region or engine.DEFAULT_REGION

    def _profile_for(self, external_id: str) -> tuple[str, str]:
        """Resolve the account an EXISTING publication belongs to."""
        _key, bound = split_external_id(external_id)
        return self._resolve_profile(bound)

    # ── availability ──────────────────────────────────────────────────────

    def available(self) -> bool:
        """True when an account is registered. Deliberately does NOT reach AWS.

        The seam calls this to decide whether to offer the destination at all, so it
        must stay cheap; whether the drive's infrastructure exists yet is a separate
        question answered by the first publish, which creates it.
        """
        try:
            self._resolve_profile()
        except PublishError:
            return False
        return True

    def installable(self) -> bool:
        """The drive builds itself on first publish, so an account with no drive yet
        is still a usable destination."""
        return True

    async def ensure_ready(self) -> bool:
        return self.available()

    # ── discovery (blocking; always called via asyncio.to_thread) ─────────

    def _find_drive(self, profile: str, region: str) -> Optional[dict[str, str]]:
        """Resolve this account's drive by tag, or ``None`` when it does not exist yet.

        Discovery is a TRUST decision -- what this returns is what gets written to and
        deleted from -- so it is hardened three ways, mirroring the deploy engine's own
        reasoning rather than borrowing its tag:

        1. The filter is the drive's OWN tag, which nothing else in the product writes,
           passed as a SINGLE filter. Requiring a second tag in the same call would make
           the result depend on how repeated ``--tag-filters`` flags combine, which is
           not worth betting discovery on when one private tag already identifies us.
        2. The bucket name must match this module's own naming scheme, so a bucket that
           carries the tag by accident is still rejected.
        3. More than one match is AMBIGUOUS and raises, naming every candidate, instead
           of pairing an arbitrary bucket with an unrelated distribution.

        The two resource types are discovered through DIFFERENT APIs, because one API does
        not cover both. ``resourcegroupstaggingapi`` returns only resources "located in the
        specified AWS Region", and CloudFront's own documentation states that Tag Editor
        and Resource Groups are not supported for CloudFront, listing only the CloudFront
        API's own tag operations for programmatic tagging. So a single regional call finds
        the bucket (S3 is regional and supported) and silently misses the distribution --
        and a missing distribution is not a harmless gap here: it makes discovery report a
        half-built drive, which sends a HEALTHY drive down the partial-create recovery path,
        builds a second distribution, and re-points the bucket policy at it so every link
        already handed out answers 403. The default region is not us-east-1, so that was
        the normal path rather than an edge case.
        """
        bucket = self._find_bucket(profile, region)
        dist_id, dist_arn = self._find_distribution(profile, region)
        if not bucket:
            return None
        return {
            "bucket": bucket,
            "distribution_id": dist_id,
            "distribution_arn": dist_arn,
        }

    def _tagged_resources(self, profile: str, region: str) -> list[str]:
        """ARNs carrying the drive's tag, as reported for ``region``."""
        out = engine._checked(
            [
                "resourcegroupstaggingapi",
                "get-resources",
                "--tag-filters",
                f"Key={TAG_DRIVE},Values={TAG_DRIVE_VALUE}",
                "--region",
                region,
                "--output",
                "json",
            ],
            profile,
            action="tag:GetResources",
        )
        try:
            data = json.loads(out or "{}")
        except json.JSONDecodeError:  # pragma: no cover - malformed API output
            return []
        return [str(m.get("ResourceARN", "")) for m in data.get("ResourceTagMappingList", [])]

    def _find_bucket(self, profile: str, region: str) -> str:
        """The drive's bucket, by tag, in the drive's own region."""
        buckets = []
        for arn in self._tagged_resources(profile, region):
            if arn.startswith("arn:aws:s3:::"):
                candidate = arn.split(":::", 1)[1]
                if _BUCKET_RE.match(candidate):
                    buckets.append(candidate)
        if len(buckets) > 1:
            raise PublishError(
                "This account has more than one personal drive bucket, so it is not "
                "clear which one to use and nothing was touched. Buckets: "
                f"{', '.join(sorted(buckets))}. Remove the extra one in the AWS "
                "console, then publish again."
            )
        return buckets[0] if buckets else ""

    def _find_distribution(self, profile: str, region: str) -> tuple[str, str]:
        """The drive's distribution, as (id, arn), or ("", "").

        Tried in two ways because the cheap one is not reliable for CloudFront. The tag
        query is kept as a FAST PATH -- one call, and it does answer in us-east-1 -- but a
        miss falls through to CloudFront's own tag API, which is the path CloudFront
        documents. Without that fallback a drive is reported half-built on every account
        whose region is not us-east-1.
        """
        arns = [a for a in self._tagged_resources(profile, "us-east-1") if ":distribution/" in a]
        if region != "us-east-1":
            arns += [a for a in self._tagged_resources(profile, region) if ":distribution/" in a]
        found = {a: a.rsplit("/", 1)[-1] for a in arns}
        if not found:
            found = self._distributions_by_own_tags(profile)
        if len(found) > 1:
            raise PublishError(
                "This account has more than one personal drive distribution, so it is "
                "not clear which one serves the drive and nothing was touched. "
                f"Distributions: {', '.join(sorted(found.values()))}. Remove the extra "
                "one in the AWS console, then publish again."
            )
        if not found:
            return "", ""
        arn, dist_id = next(iter(found.items()))
        return dist_id, arn

    def _distributions_by_own_tags(self, profile: str) -> dict[str, str]:
        """Drive distributions found through CloudFront's OWN tag API, as {arn: id}.

        Reached only when the tag query found nothing. Enumerating and reading tags per
        distribution costs one call plus one per distribution, which is why it is the
        fallback and not the primary -- but it is the only path CloudFront documents, so
        it is what makes discovery correct outside us-east-1.
        """
        try:
            out = engine._checked(
                ["cloudfront", "list-distributions", "--output", "json"],
                profile,
                action="cloudfront:ListDistributions",
            )
            items = (json.loads(out or "{}").get("DistributionList") or {}).get("Items") or []
        except (engine.AWSError, json.JSONDecodeError):
            # Discovery that cannot see is reported as "nothing found" by the caller, which
            # refuses rather than guessing. Raising here would also break withdrawal.
            return {}
        found: dict[str, str] = {}
        for item in items:
            arn = str(item.get("ARN", ""))
            if not arn:
                continue
            try:
                tags_out = engine._checked(
                    ["cloudfront", "list-tags-for-resource", "--resource", arn, "--output", "json"],
                    profile,
                    action="cloudfront:ListTagsForResource",
                )
                tags = (json.loads(tags_out or "{}").get("Tags") or {}).get("Items") or []
            except (engine.AWSError, json.JSONDecodeError):
                continue
            for tag in tags:
                if tag.get("Key") == TAG_DRIVE and tag.get("Value") == TAG_DRIVE_VALUE:
                    found[arn] = str(item.get("Id", "")) or arn.rsplit("/", 1)[-1]
                    break
        return found

    # ── infrastructure (blocking) ─────────────────────────────────────────

    def _prefix_scoped_bucket_policy(self, bucket: str, distribution_arn: str) -> str:
        """Grant CloudFront read on the served prefix ONLY.

        The engine's shared policy grants ``<bucket>/*``, which is right for a bucket
        that holds one site's content and nothing else. This bucket also holds private
        objects, so a whole-bucket grant would serve them. The Deny is belt-and-braces:
        the Allow already omits them, and an explicit Deny keeps that true if the Allow
        is ever widened.
        """
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AllowCloudFrontServedPrefixOnly",
                    "Effect": "Allow",
                    "Principal": {"Service": "cloudfront.amazonaws.com"},
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{bucket}/{PUBLIC_PREFIX}*",
                    "Condition": {"StringEquals": {"AWS:SourceArn": distribution_arn}},
                },
                {
                    "Sid": "DenyCloudFrontEverythingElse",
                    "Effect": "Deny",
                    "Principal": {"Service": "cloudfront.amazonaws.com"},
                    "Action": "s3:GetObject",
                    "NotResource": f"arn:aws:s3:::{bucket}/{PUBLIC_PREFIX}*",
                },
            ],
        }
        return json.dumps(policy)

    def _put_bucket_policy(self, bucket: str, distribution_arn: str, profile: str) -> None:
        """Install the prefix-scoped read policy. Idempotent -- PutBucketPolicy replaces.

        Called on creation AND every time an existing drive is resolved, because without
        it a single failure here is permanent and silent. Both resources are tagged, so
        discovery finds the drive the moment it exists; if the policy did not land, the
        reuse path would hand out links to a bucket CloudFront cannot read and every one
        of them would 403 for the life of the account while the publish reported success.
        Re-installing costs one call per publish and makes that state self-healing -- it
        also covers a policy deleted or edited by hand, which nothing else would repair.
        """
        engine._checked(
            [
                "s3api",
                "put-bucket-policy",
                "--bucket",
                bucket,
                "--policy",
                self._prefix_scoped_bucket_policy(bucket, distribution_arn),
            ],
            profile,
            action="s3:PutBucketPolicy",
        )

    def _reassert_bucket_policy(
        self, found: dict[str, str], profile: str, *, link_promised: bool
    ) -> None:
        """Re-install the read policy on an EXISTING drive.

        Both resources are tagged, so discovery finds the drive the moment it exists; if a
        first publish lost ``PutBucketPolicy`` to a denial or a throttle, the reuse path
        would otherwise hand out links to a bucket CloudFront cannot read -- 403 for the
        life of the account, reported as success. Re-installing makes that self-healing,
        and also repairs a policy deleted or edited by hand.

        Whether a failure is fatal depends on ONE question: does this call end by giving
        the user a URL? If it does, it must not, while it cannot confirm the bucket is
        readable. If it does not -- a push, a withdrawal, a take-private -- raising would
        turn an OPTIONAL permission into one every call requires, so a profile without
        ``s3:PutBucketPolicy`` would lose the ability to withdraw from a drive that works.
        That question is deliberately NOT ``require_serving``: a publish uploads before it
        checks the rollout, so it resolves the drive without asserting the network yet
        while still being a call that promises a link. Creation asserts the policy
        unconditionally -- a drive that never got one is a failed publish, correctly.
        """
        if not found.get("distribution_arn"):
            return
        try:
            self._put_bucket_policy(found["bucket"], found["distribution_arn"], profile)
        except Exception as exc:
            if link_promised:
                raise PublishError(
                    "The drive's read permission could not be confirmed, so no link was "
                    f"handed out rather than one that may not resolve: {exc}"
                ) from exc
            logger.warning(
                "personal drive: could not re-assert the read policy on %s; if it is "
                "missing, published links will not resolve until it is restored: %s",
                found.get("bucket", ""),
                exc,
            )

    def _ensure_drive(
        self, profile: str, region: str, *, require_serving: bool, link_promised: bool = False
    ) -> tuple[str, str, str]:
        """Find or create the drive. Returns (bucket, distribution id, domain).

        ``require_serving`` gates the delivery-network health assertion, and the
        distinction is load-bearing rather than a convenience. A path that HANDS OUT a
        link must refuse when the network cannot serve it, or it returns a URL that
        resolves to nothing. A path that REMOVES content or takes it private must NOT:
        those act on S3 objects, and the delivery network's rollout state cannot affect
        whether an S3 write succeeds, so refusing there would block a withdrawal for a
        reason unrelated to it -- and a withdrawal the user cannot complete is how content
        stays served after they asked for it to stop.
        """
        existing = self._find_drive(profile, region)
        if existing and existing.get("bucket") and existing.get("distribution_id"):
            bucket = existing["bucket"]
            dist_id = existing["distribution_id"]
            if existing.get("distribution_arn"):
                self._reassert_bucket_policy(existing, profile, link_promised=link_promised)
            status, domain = engine.distribution_status(dist_id, profile)
            if require_serving:
                self._assert_serving(dist_id, status, profile)
            return bucket, dist_id, domain
        return self._create_drive(profile, region, existing, require_serving=require_serving)

    def _require_drive(
        self, profile: str, region: str, *, require_serving: bool
    ) -> tuple[str, str, str]:
        """Resolve an EXISTING drive. Never creates one.

        Only a first publish may build infrastructure. Every other path acts on a
        publication that already exists, so a drive that cannot be found is a condition
        to report, not something to provision. Routing them through the create-capable
        path meant a discovery miss -- the drive's tag removed by hand, a transient
        answer from the tagging API -- made ``unpublish`` BUILD a fresh bucket, OAC and
        distribution, delete keys that were never in it, invalidate it, and report
        success, while the original object stayed public. With the withdrawal reported as
        successful the local record went too. A withdrawal that provisions infrastructure
        and leaves the content served is the worst outcome this module can produce.
        """
        found = self._find_drive(profile, region)
        if not found or not found.get("bucket") or not found.get("distribution_id"):
            raise PublishError(
                "This account's personal drive could not be found, so nothing was "
                "changed. If it was deleted or its tag was altered, any copy it was "
                "serving may still be public -- check the drive's bucket in the AWS "
                "console for this account."
            )
        if found.get("distribution_arn"):
            # Here the two questions coincide: the only caller that asserts the network is
            # make-public, which is also the only one that ends by handing out a link.
            self._reassert_bucket_policy(found, profile, link_promised=require_serving)
        dist_id = found["distribution_id"]
        status, domain = engine.distribution_status(dist_id, profile)
        if require_serving:
            self._assert_serving(dist_id, status, profile)
        return found["bucket"], dist_id, domain

    def _create_drive(
        self,
        profile: str,
        region: str,
        existing: Optional[dict[str, str]],
        *,
        require_serving: bool,
    ) -> tuple[str, str, str]:
        """Build the drive under a lock, re-checking discovery once it is held."""
        with _CREATE_LOCK:
            # Re-check inside the lock: a concurrent first publish may have built the
            # drive while this call waited, and creating a second one would leave two
            # tagged drives for discovery to reject.
            recheck = self._find_drive(profile, region)
            if recheck and recheck.get("bucket") and recheck.get("distribution_id"):
                bucket = recheck["bucket"]
                dist_id = recheck["distribution_id"]
                status, domain = engine.distribution_status(dist_id, profile)
                if require_serving:
                    self._assert_serving(dist_id, status, profile)
                return bucket, dist_id, domain

            # Partial-create recovery: a prior run tagged the bucket but died before the
            # distribution existed. Reuse it rather than orphaning it behind a new one.
            bucket = (recheck or existing or {}).get("bucket", "")
            if not bucket:
                bucket = self._create_bucket(profile, region)

            engine._harden_bucket(
                bucket,
                profile,
                f"TagSet=[{{Key={engine.TAG_MANAGED},Value=true}},"
                f"{{Key={TAG_DRIVE},Value={TAG_DRIVE_VALUE}}}]",
            )
            oac_id = engine.create_oac(engine._oac_name(SITE_TAG_VALUE), profile)
            # Born with the final tag set: the drive's own tag, and NOT the deploy
            # surface's site tag. Adding ours after the fact and then removing theirs
            # meant two extra calls, a window in which the resource was listed as a
            # deploy site and destroyable there, and a cloudfront:UntagResource
            # permission the drive would otherwise never need -- one the shipped
            # least-privilege policy does not grant, so on such a profile the removal
            # could never succeed and the window never closed.
            dist = engine.create_distribution(
                bucket,
                region,
                oac_id,
                SITE_TAG_VALUE,
                profile,
                tags=[
                    {"Key": engine.TAG_MANAGED, "Value": "true"},
                    {"Key": TAG_DRIVE, "Value": TAG_DRIVE_VALUE},
                ],
                response_headers_policy_id=self._sandbox_headers_policy(profile),
            )
            # The policy needs the distribution ARN (it pins AWS:SourceArn), so it cannot
            # precede the distribution. It therefore cannot precede discoverability
            # either, now that the distribution is born carrying the drive tag -- which is
            # why the reuse path re-installs it rather than assuming this call succeeded.
            self._put_bucket_policy(bucket, dist["arn"], profile)
            status, domain = engine.distribution_status(dist["id"], profile)
            if require_serving:
                self._assert_serving(dist["id"], status, profile)
            return bucket, dist["id"], domain

    def _sandbox_headers_policy(self, profile: str) -> str:
        """Find or create the response-headers policy that isolates published artifacts.

        Every artifact on this drive is served from ONE domain, so without this they share
        a browser origin: localStorage, IndexedDB and cookies are keyed by origin, which
        means one published document could read what another wrote. The documents are
        mutually untrusted -- they are authored content, and the drive's owner is not the
        only author of what ends up in them -- so sharing an origin is the one cost of
        pooling that cannot be argued away.

        ``Content-Security-Policy: sandbox allow-scripts allow-popups`` makes the browser
        give each document an OPAQUE origin: scripts still run, links still open, and
        storage is simply not reachable, so there is nothing to share. This is the same
        posture the dashboard's own viewer already applies -- it renders artifact HTML in
        an iframe without ``allow-same-origin`` -- so no artifact loses a capability it
        had; published copies were the one surface where the sandbox was missing.

        The ``sandbox`` directive is deliberately delivered as a HEADER. It is ignored in a
        ``<meta>`` CSP by specification, so injecting it into the document is not an
        option, and a header on a CloudFront distribution comes only from a policy.

        Find-or-create, keyed on a fixed name: the policy is account-global, so a second
        drive or a retried create must reuse it rather than fail on the name collision.

        A name match is necessary but NOT sufficient to reuse. This product runs agents
        that hold the user's own AWS credentials, so a policy of this fixed name can be
        pre-created WITHOUT the sandbox CSP (or with a weaker one). Reusing it on the name
        alone would hand its id to ``create_distribution`` and silently drop the
        opaque-origin sandbox this method's docstring calls the one cost of pooling that
        cannot be argued away -- published documents would then share a real browser
        origin and one could read another's origin-keyed storage. Separately, and with no
        attacker involved: if a same-named policy's config were never compared, any future
        change to :data:`_SANDBOX_CSP` would never reach an account that already has the
        policy -- the name still matches, so a stale policy would be reused forever. So a
        name match reuses ONLY when the policy's CSP equals :data:`_SANDBOX_CSP` exactly
        with ``Override`` true; anything else FAILS CLOSED, never silently attaching the
        mismatched policy and never silently creating a duplicate under a taken name.
        """
        name = _SANDBOX_POLICY_NAME
        out = engine._checked(
            [
                "cloudfront",
                "list-response-headers-policies",
                "--type",
                "custom",
                "--output",
                "json",
            ],
            profile,
            action="cloudfront:ListResponseHeadersPolicies",
        )
        try:
            items = json.loads(out or "{}").get("ResponseHeadersPolicyList", {}).get("Items", [])
        except json.JSONDecodeError:  # pragma: no cover - malformed API output
            items = []
        for item in items:
            cfg = (item or {}).get("ResponseHeadersPolicy", {})
            policy_config = cfg.get("ResponseHeadersPolicyConfig", {})
            if policy_config.get("Name") != name:
                continue
            # The list response ALREADY carries each item's full ResponseHeadersPolicyConfig,
            # so verifying the CSP here costs ZERO extra API calls.
            policy_id = str(cfg.get("Id", ""))
            csp = policy_config.get("SecurityHeadersConfig", {}).get("ContentSecurityPolicy", {})
            if csp.get("ContentSecurityPolicy") == _SANDBOX_CSP and csp.get("Override") is True:
                return policy_id
            raise PublishError(
                f"A CloudFront response-headers policy named {name!r} (id {policy_id}) "
                f"already exists in this account but does not carry the required "
                f"opaque-origin sandbox (Content-Security-Policy {_SANDBOX_CSP!r} with "
                f"Override enabled). Reusing it would let published documents share a "
                f"browser origin. Policy names are account-global, so Kiro Crew cannot "
                f"create a correct one under this name automatically: inspect the existing "
                f"policy in the CloudFront console and delete or rename it, then publish "
                f"again."
            )
        config = _sandbox_policy_config()
        created = engine._checked(
            [
                "cloudfront",
                "create-response-headers-policy",
                "--response-headers-policy-config",
                json.dumps(config),
                "--output",
                "json",
            ],
            profile,
            action="cloudfront:CreateResponseHeadersPolicy",
        )
        return str(json.loads(created)["ResponseHeadersPolicy"]["Id"])

    def _create_bucket(self, profile: str, region: str) -> str:
        """Allocate the drive's bucket, retrying past a globally-taken random name."""
        for _ in range(5):
            candidate = f"{BUCKET_PREFIX}{secrets.token_hex(6)}"
            rc, _out, err = engine.run_aws(
                ["s3api", "create-bucket", "--bucket", candidate, "--region", region]
                + (
                    []
                    if region == "us-east-1"
                    else ["--create-bucket-configuration", f"LocationConstraint={region}"]
                ),
                profile,
            )
            if rc == 0:
                return candidate
            if "BucketAlreadyExists" in err or "BucketAlreadyOwnedByYou" in err:
                continue
            raise PublishError(
                "Could not create the drive's storage bucket"
                + (
                    f" (missing IAM statement {sid})"
                    if (sid := engine.map_access_denied(err))
                    else ""
                )
                + f": {engine._trimmed_stderr(err)}"
            )
        raise PublishError("Could not allocate a unique bucket name for the drive.")

    def _serving_notice(self, dist_id: str, status: str, profile: str) -> str:
        """Describe the two states that make a link dead, or "" when it will resolve.

        Both are states a user hits in practice and neither is visible from the URL
        alone, so a publish that returned a link without mentioning them would look
        like a product that hands out broken links.

        Returned rather than raised so each caller can choose: before anything has been
        written, an abort is right; AFTER content is already uploaded it is not, because
        the caller stores the publication only on success and an abort would leave the
        object with no withdrawal handle.
        """
        if status and status != "Deployed":
            return (
                "The drive's delivery network is still rolling out this change. "
                "Published content becomes reachable when it "
                "finishes -- usually a few minutes. Nothing was lost; try again "
                "shortly and the same link will work."
            )
        if not self._distribution_enabled(dist_id, profile):
            return (
                "The drive's delivery network is DISABLED, so published links will "
                "not resolve. This is not a state Kiro Crew sets -- an account "
                "security control or an administrator disabled it. Re-enable the "
                "distribution in the AWS console before publishing."
            )
        return ""

    def _assert_serving(self, dist_id: str, status: str, profile: str) -> None:
        """Raising form of :meth:`_serving_notice`, for callers that have written nothing."""
        notice = self._serving_notice(dist_id, status, profile)
        if notice:
            raise PublishError(notice)

    def _distribution_enabled(self, dist_id: str, profile: str) -> bool:
        try:
            out = engine._checked(
                ["cloudfront", "get-distribution-config", "--id", dist_id, "--output", "json"],
                profile,
                action="cloudfront:GetDistributionConfig",
            )
            return bool(json.loads(out).get("DistributionConfig", {}).get("Enabled", True))
        except Exception:  # pragma: no cover - treat unreadable as no objection
            return True

    # ── object plane ──────────────────────────────────────────────────────

    @staticmethod
    def _is_public(visibility: str) -> bool:
        return (visibility or "").strip().lower() == "public"

    @staticmethod
    def _check_visibility(visibility: str) -> None:
        v = (visibility or "").strip().lower()
        if v in ("public", "private", ""):
            return
        if v == "shared":
            # Explicit refusal, not a silent downgrade: quietly publishing a
            # "share with these people" request as world-readable would be the
            # worst possible failure mode for this destination.
            raise CapabilityNotSupportedError(Capability.SHARING)
        raise PublishError(f"Unsupported visibility {visibility!r} for the personal drive.")

    def _read_payload(self, file_path: str) -> bytes:
        # Deferred, and it must stay deferred. ``kiro_crew.artifacts`` resolves config
        # while it is still importing, which installs the platform context, which calls
        # this edition's register_publish_providers, which imports this module. At module
        # scope this import would therefore run against a half-initialised ``artifacts``
        # whose MAX_CONTENT_BYTES is not bound yet, and every publish would die on a
        # circular-import ImportError. Importing here keeps one source of truth for the
        # limit without closing that cycle.
        from kiro_crew.artifacts import MAX_CONTENT_BYTES

        data = Path(file_path).read_bytes()
        if len(data) > MAX_CONTENT_BYTES:
            raise PublishError(
                f"This artifact is {len(data)} bytes, over the "
                f"{MAX_CONTENT_BYTES}-byte limit for a single published object."
            )
        return data

    _NOT_FOUND_MARKERS = ("Not Found", "NoSuchKey", "NoSuchBucket", "404")

    def _head(self, profile: str, bucket: str, key: str) -> Optional[dict[str, Any]]:
        """Return the object's metadata, ``None`` only for a CONFIRMED absence.

        Collapsing every failure to "absent" is what makes this dangerous: the callers use
        the answer to DECIDE, so a throttled or unauthorized read taken as "already gone"
        makes a sharing change conclude the object was moved already and return success
        without moving anything, and makes a push write to the wrong prefix. Anything that
        is not a definite not-found is therefore raised.
        """
        rc, out, err = engine.run_aws(
            ["s3api", "head-object", "--bucket", bucket, "--key", key, "--output", "json"],
            profile,
        )
        if rc == 0:
            try:
                return json.loads(out or "{}")
            except json.JSONDecodeError:  # pragma: no cover
                return {}
        if any(marker in err for marker in self._NOT_FOUND_MARKERS):
            return None
        raise PublishError(
            "Could not check whether this artifact is still in the drive, so nothing "
            f"was changed: {engine._trimmed_stderr(err)}"
        )

    def _put(
        self,
        *,
        profile: str,
        bucket: str,
        key: str,
        path: str,
        content_type: str,
        digest: str,
    ) -> None:
        """Upload one object with its content type and content digest declared.

        Content type is not optional here. Without it S3 stores
        ``binary/octet-stream``, and the delivery network's security headers include
        ``X-Content-Type-Options: nosniff``, so a browser refuses to treat an HTML
        page as HTML and downloads it instead of rendering it.

        The digest is stored as object metadata so a later push can tell that the
        object changed out of band, which is the whole content of the concurrency
        guarantee this provider declares in :meth:`sync_model`.
        """
        engine._checked(
            [
                "s3api",
                "put-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--body",
                path,
                "--content-type",
                content_type or "application/octet-stream",
                "--cache-control",
                f"max-age={_EDGE_MAX_AGE_SECONDS}",
                "--metadata",
                f"{_SHA_META}={digest}",
            ],
            profile,
            action="s3:PutObject",
            timeout=300,
        )

    # The suffixes the seam actually writes when it stages an artifact for upload. Keyed
    # by EXTENSION, not by artifact kind, so this answers a stable local question ("what
    # does this file extension mean") instead of duplicating the seam's kind vocabulary.
    # Deliberately not mimetypes.guess_type: that reads the host's /etc/mime.types, and a
    # header served under nosniff must not vary by machine.
    _SUFFIX_TYPES: dict[str, str] = {
        ".html": "text/html",
        ".md": "text/markdown",
        ".svg": "image/svg+xml",
        ".txt": "text/plain",
    }

    @classmethod
    def _content_type_for(cls, file_path: str, stored: str) -> str:
        """The type for the bytes being pushed NOW, falling back to the stored header.

        A push re-uploads whatever the seam staged, and it stages with the extension of
        the artifact's CURRENT kind -- so an artifact re-saved as HTML after being
        published as text arrives here as ``.html`` while the remote object still
        advertises ``text/plain``. Reusing the remote's header would serve the new bytes
        under the old type, and with ``nosniff`` the browser renders markup as source
        text. The argument is the truth for this push; the remote header is a fact about
        the past, kept only for a suffix this map does not recognise.
        """
        return cls._SUFFIX_TYPES.get(Path(file_path).suffix.lower(), stored)

    @staticmethod
    def _stored_digest(head: dict[str, Any]) -> str:
        meta = head.get("Metadata") or {}
        if isinstance(meta, dict):
            return str(meta.get(_SHA_META, "") or "")
        return ""

    # ── PublishProvider interface ─────────────────────────────────────────

    def view_url_for(self, external_id: str) -> str:
        """Stable link for an id, built from its random half only.

        Empty when this process has not resolved the drive's domain yet -- the seam
        treats an empty url as "published, no browsable link" rather than as failure,
        and the next publish or push fills the cache in.
        """
        key_part, bound = split_external_id(external_id)
        if not key_part:
            return ""
        domain = self._domains.get(bound, "")
        if not domain and not bound and len(self._domains) == 1:
            # Only an id carrying NO account half may borrow the sole cached domain. A
            # named account that simply is not cached yet must resolve to nothing: handing
            # it another account's domain would publish a link into the wrong AWS
            # account's drive, which reads as a working link and is not one.
            domain = next(iter(self._domains.values()), "")
        if not domain:
            return ""
        return f"https://{domain}/{PUBLIC_PREFIX}{key_part}"

    async def publish(
        self,
        *,
        file_path: str,
        content_type: str,
        title: str,
        summary: str,
        tags: list[str],
        visibility: str,
        shared_with: list[str],
    ) -> PublishResult:
        self._check_visibility(visibility)
        if shared_with:
            raise CapabilityNotSupportedError(Capability.SHARING)
        public = self._is_public(visibility)

        def _work() -> tuple[str, str, str, str, str]:
            profile, region = self._resolve_profile()
            payload = self._read_payload(file_path)
            digest = hashlib.sha256(payload).hexdigest()
            external_id = make_external_id(profile)
            key_part, _ = split_external_id(external_id)
            # Creating the drive is required either way, and the serving assertion is
            # deliberately NOT made here. A distribution is always InProgress in the
            # minutes after it is created, so asserting before the upload made the very
            # first publish on a fresh account fail with nothing written -- leaving a
            # tagged distribution over an empty origin, which is precisely the
            # dangling-origin shape this destination exists to avoid. Upload first, then
            # report the rollout state below.
            bucket, dist_id, domain = self._ensure_drive(
                profile, region, require_serving=False, link_promised=public
            )
            prefix = PUBLIC_PREFIX if public else PRIVATE_PREFIX
            self._put(
                profile=profile,
                bucket=bucket,
                key=f"{prefix}{key_part}",
                path=file_path,
                content_type=content_type,
                digest=digest,
            )
            # No invalidation here. The key is freshly minted for this publish, so no
            # edge has ever served it and there is nothing cached to purge. Issuing one
            # anyway would add the one failure mode this call otherwise does not have:
            # the object is already uploaded at that point, so a throttled invalidation
            # would surface as a failed publish while leaving the object behind with no
            # publication record pointing at it.
            #
            # NOW report the rollout state, and only when a link is being handed out: the
            # object is in place, so a distribution that is still deploying will serve it
            # the moment it finishes, and the message can honestly say the same link will
            # work shortly. Re-reading the status here rather than reusing the one from
            # drive resolution also picks up a rollout that finished during the upload.
            #
            # REPORTED, never raised. The object is already uploaded at this point and the
            # caller records the publication only when this returns, so raising here left
            # a public object with no withdrawal handle -- while the message it raised
            # claimed "nothing was lost". The notice rides back on the result instead.
            notice = ""
            if public:
                # The PROBE is inside the guard too, not just its verdict. Making the
                # verdict non-fatal was not enough: this read reaches CloudFront, so a
                # throttle or a permissions gap raises here, the wrapper below turns it
                # into a PublishError, and the caller -- which records the publication
                # only when this returns -- leaves the object public with no withdrawal
                # handle. Not knowing the rollout state is a thing to say, not a reason to
                # discard a handle for content that is already uploaded.
                try:
                    status, _domain = engine.distribution_status(dist_id, profile)
                    notice = self._serving_notice(dist_id, status, profile)
                except engine.AWSError as exc:
                    logger.warning(
                        "publish: could not read the drive's rollout state for %s: %s",
                        dist_id,
                        exc,
                    )
                    notice = (
                        "The publish completed, but the drive's delivery network did not "
                        "report whether it is serving yet, so the link may not resolve for "
                        "a few minutes. Nothing was lost."
                    )
            return external_id, profile, domain, digest, notice

        try:
            external_id, profile, domain, digest, notice = await asyncio.to_thread(_work)
        except PublishError:
            raise
        except engine.AWSError as exc:
            raise PublishError(f"The drive rejected this publish: {exc}") from exc

        if domain:
            self._domains[profile] = domain
        return PublishResult(
            external_id=external_id,
            view_url=self.view_url_for(external_id) if public else "",
            version_number=1,
            concurrency_token=digest,
            owner="",
            notice=notice,
        )

    async def push_version(
        self, *, external_id: str, file_path: str, expected_token: str
    ) -> PushResult:
        """Re-upload the artifact's bytes. Best-effort by contract: never raises."""
        key_part, _ = split_external_id(external_id)

        def _work() -> tuple[str, str, str, bool]:
            profile, region = self._profile_for(external_id)
            payload = self._read_payload(file_path)
            digest = hashlib.sha256(payload).hexdigest()
            # A push mutates content in place; it does not hand out a new link, and
            # refusing mid-rollout would strand the artifact on stale bytes.
            bucket, dist_id, domain = self._require_drive(profile, region, require_serving=False)
            key = f"{PUBLIC_PREFIX}{key_part}"
            head = self._head(profile, bucket, key)
            if head is None:
                # A private artifact is pushed the same way; it just is not served, so
                # there is nothing to invalidate.
                key = f"{PRIVATE_PREFIX}{key_part}"
                head = self._head(profile, bucket, key)
                if head is None:
                    raise PublishError(
                        "This artifact is no longer in the drive, so there was "
                        "nothing to update. Publish it again to recreate it."
                    )
            # The declared concurrency is token-guarded, so honour the token: a remote
            # digest that matches neither what we last pushed nor what we are about to
            # push means the object changed out of band, and overwriting it would
            # destroy that change while reporting success. A MISSING digest counts as
            # a conflict too, and the empty-string check is deliberately absent: every
            # object this module writes carries the metadata, so its absence means the
            # object was replaced by something that is not us (an upload from the S3
            # console, or `aws s3 cp`, neither of which preserves user metadata).
            # Re-adding an `and stored` guard here would fail open on exactly the
            # signal this check depends on.
            stored = self._stored_digest(head)
            if expected_token and stored != expected_token and stored != digest:
                return domain, digest, profile, True
            self._put(
                profile=profile,
                bucket=bucket,
                key=key,
                path=file_path,
                content_type=self._content_type_for(
                    file_path, str(head.get("ContentType") or "application/octet-stream")
                ),
                digest=digest,
            )
            if key.startswith(PUBLIC_PREFIX):
                # The bytes are already uploaded, so this is the one call left that can
                # fail after the remote changed. Reporting failure here would leave the
                # caller holding the PREVIOUS token while the object carries the new
                # digest -- and since a stored digest matching neither the token nor the
                # next payload is a conflict, every later push would conflict forever with
                # nothing able to reconcile it. A stale edge copy expiring on its own is
                # the smaller harm, so the purge warns and the new token is returned.
                try:
                    engine.invalidate(dist_id, profile)
                except Exception as exc:
                    logger.warning(
                        "personal drive: %s was updated but its cached copy could not be "
                        "purged; the old bytes may be served until they expire: %s",
                        key,
                        exc,
                    )
            return domain, digest, profile, False

        try:
            domain, digest, profile, conflict = await asyncio.to_thread(
                _serialized, key_part, _work
            )
        except Exception as exc:  # contract: report, never raise
            return PushResult(error=str(exc))
        if domain:
            self._domains[profile] = domain
        if conflict:
            return PushResult(
                conflict=True,
                error=(
                    "The copy in the drive changed since Kiro Crew last pushed it, so "
                    "it was left alone rather than overwritten."
                ),
            )
        return PushResult(version_number=0, concurrency_token=digest)

    async def update_sharing(
        self, *, external_id: str, visibility: str, shared_with: list[str]
    ) -> None:
        """Move the object between the served and unserved prefix.

        Making something private removes it from the served prefix, so the public link
        stops resolving -- but edge caches can still answer for already-cached content
        until the invalidation lands, and a copy someone already downloaded cannot be
        recalled at all.
        """
        self._check_visibility(visibility)
        if shared_with:
            raise CapabilityNotSupportedError(Capability.SHARING)
        key_part, _ = split_external_id(external_id)
        want_public = self._is_public(visibility)

        def _work() -> None:
            profile, region = self._profile_for(external_id)
            # Taking something PRIVATE must not be blocked by an unhealthy network:
            # that path withdraws public content, and refusing it would leave the
            # content served while the caller believes it was withdrawn.
            bucket, dist_id, domain = self._require_drive(
                profile, region, require_serving=want_public
            )
            # Cache the domain under the BOUND profile. The seam derives the public link
            # from view_url_for immediately after this call, and on a fresh process this
            # is the only call that resolved the domain -- dropping it left the link
            # empty, or, with exactly one other account cached, pointing into THAT
            # account's drive.
            if domain:
                self._domains[profile] = domain
            src = f"{PRIVATE_PREFIX if want_public else PUBLIC_PREFIX}{key_part}"
            dst = f"{PUBLIC_PREFIX if want_public else PRIVATE_PREFIX}{key_part}"
            dst_present = self._head(profile, bucket, dst) is not None
            if self._head(profile, bucket, src) is None:
                if dst_present:
                    # Already moved -- but do NOT return without purging. If a previous
                    # attempt moved the object and then failed on the invalidation, the
                    # edge cache still holds the public copy; returning here would make
                    # the retry a no-op and leave those bytes served while the record
                    # says private. The purge is the part that has to be idempotent.
                    engine.invalidate(dist_id, profile)
                    return
                raise PublishError(
                    "This artifact is no longer in the drive, so its sharing could "
                    "not be changed."
                )
            # Copy unless the destination already holds the authoritative bytes. Which
            # side is authoritative is decided by ONE fact: push_version heads
            # public/<key> first and writes to whichever prefix exists, so whenever both
            # do, a push lands on public/<key>. Going public that is dst, so a second
            # copy would overwrite the pushed version with the pre-push source. Going
            # private that is src, so skipping the copy and then deleting src would
            # DESTROY the pushed bytes, keep a stale private copy, and -- because the
            # record's token then matches neither -- leave every later push in permanent
            # conflict with nothing able to heal it.
            if not dst_present or not want_public:
                engine._checked(
                    [
                        "s3api",
                        "copy-object",
                        "--bucket",
                        bucket,
                        "--key",
                        dst,
                        "--copy-source",
                        f"{bucket}/{src}",
                        "--metadata-directive",
                        "COPY",
                    ],
                    profile,
                    action="s3:PutObject",
                )
            # The source delete is the WHOLE POINT when taking something private -- src is
            # the served copy, and leaving it is the failure. On the way OUT to public it
            # is only cleanup: the copy already landed in the served prefix, so the object
            # is public the moment it succeeds, and the leftover private copy is not
            # served by the prefix-scoped policy. Letting a throttle on that cleanup raise
            # would abort before the caller records the new visibility, leaving the record
            # saying "private" about content that is being served -- the inversion that
            # actually harms the user, and the reverse of what a failure here looks like.
            try:
                engine._checked(
                    ["s3api", "delete-object", "--bucket", bucket, "--key", src],
                    profile,
                    action="s3:DeleteObject",
                )
            except Exception as exc:
                if not want_public:
                    raise
                logger.warning(
                    "personal drive: the private copy of %s could not be cleaned up "
                    "after it was made public: %s",
                    key_part,
                    exc,
                )
            # Same asymmetry as the delete above, for the same reason. Taking something
            # private, the purge is load-bearing: it evicts the copy the edge is still
            # serving, so a failure means the withdrawal did not happen. Going public it
            # is cleanup -- the object is already served from the new prefix, and letting
            # a failure raise here would abort before the caller records the new
            # visibility, leaving the record saying "private" about served content.
            try:
                engine.invalidate(dist_id, profile)
            except Exception as exc:
                if not want_public:
                    raise
                logger.warning(
                    "personal drive: the cache purge after publishing %s failed; a stale "
                    "copy may be served until it expires: %s",
                    key_part,
                    exc,
                )

        try:
            await asyncio.to_thread(_serialized, key_part, _work)
        except PublishError:
            raise
        except engine.AWSError as exc:
            raise PublishError(f"The drive rejected this sharing change: {exc}") from exc

    async def unpublish(self, *, external_id: str) -> None:
        key_part, _ = split_external_id(external_id)

        def _work() -> None:
            profile, region = self._profile_for(external_id)
            # A removal is not gated on the delivery network being healthy, because the
            # two have nothing to do with each other: the delete targets S3, and refusing
            # it while CloudFront is mid-rollout would block a withdrawal for a reason
            # that cannot affect whether it succeeds.
            bucket, dist_id, _domain = self._require_drive(profile, region, require_serving=False)
            # No existence probe before the delete, deliberately. DeleteObject is
            # idempotent on a key that is not there, so a probe cannot change what this
            # call does -- it only adds one more request that can fail on a throttle, an
            # expired token or a scoped-down role, turning a withdrawal that would have
            # worked into one that reports failure. Delete both prefixes unconditionally
            # and let S3 shrug at the one that is absent.
            #
            # By the same argument neither delete may abort the other, and neither may
            # skip the purge below: whatever is skipped here is never retried by anyone,
            # so a throttle on the private key must not leave the public copy alive in
            # the edge cache with no local handle left to purge it.
            failure: Exception | None = None
            for key in (f"{PUBLIC_PREFIX}{key_part}", f"{PRIVATE_PREFIX}{key_part}"):
                try:
                    engine._checked(
                        ["s3api", "delete-object", "--bucket", bucket, "--key", key],
                        profile,
                        action="s3:DeleteObject",
                    )
                except Exception as exc:  # reported below, after the purge has run
                    failure = failure or exc
            # Always purge: edge caches may hold the public copy, and this is the last
            # call that will ever run for this artifact.
            engine.invalidate(dist_id, profile)
            if failure is not None:
                raise failure

        try:
            await asyncio.to_thread(_serialized, key_part, _work)
        except PublishError:
            raise
        except engine.AWSError as exc:
            raise PublishError(f"The drive rejected this removal: {exc}") from exc

    # ── capability negotiation ────────────────────────────────────────────

    def capabilities(self) -> set[Capability]:
        """``SHARING`` means "this destination can change an artifact's visibility".

        It does NOT mean "this destination has a per-principal grant list" -- that is
        what :meth:`sharing_model` declares, and it declares ``supports_shared=False``,
        so no grant-list control is ever offered and :meth:`update_sharing` refuses a
        non-empty ``shared_with`` outright.

        Withholding it is not the conservative choice, it is a silent hole: the
        orchestration gates its visibility reconciliation on this capability, so a
        re-publish of an already-public artifact requesting private would push new bytes
        into the SERVED prefix and skip the move entirely -- content still public, local
        record still saying public.
        """
        return {Capability.SHARING}

    def kind_support(self, kind: str) -> KindSupport:
        """A blob store serves any bytes the seam renders for it."""
        return KindSupport.NATIVE

    def sharing_model(self) -> SharingModel:
        return SharingModel(
            supports_private=True,
            supports_shared=False,
            supports_public=True,
            principal_kind="none",
            supports_roles=False,
            supports_expiration=False,
            programmable=True,
        )

    def sync_model(self) -> SyncModel:
        """Kiro Crew owns the content; the drive is a token-guarded mirror of it."""
        return SyncModel(authority="kirocrew", concurrency="token")

    def discovery_model(self) -> DiscoveryModel:
        """Nothing to browse: a link is unguessable by construction and the drive
        keeps no per-artifact listing of its own."""
        return DiscoveryModel(
            list_mine=False,
            list_shared_with_me=False,
            list_public=False,
            full_text_search=False,
            pull_by_id=False,
        )


def register_public_edition_providers() -> None:
    """Register the drive as the public edition's default publish destination.

    One key, ``DEFAULT_PROVIDER``: ``publish_sync`` resolves the unnamed destination
    through it, so registering it is what makes the drive the edition's DEFAULT
    publisher rather than merely an available one.

    There is deliberately no per-account key. A provider id is validated at the one HTTP
    boundary that accepts one against ``^[a-z0-9-]{1,32}$``, so an account-qualified key
    would be listed by ``list_providers`` and rejected on submit -- a picker row that
    400s when clicked is worse than no row at all. Choosing an account per publish needs
    a real publish argument on the shared seam, which is its own change. Until then a
    new publish uses the profile registry's default, and an existing publication keeps
    the account bound into its ``external_id``.

    There is also no bare ``personal-drive`` alias: ``list_providers`` returns one
    instance per registered key without deduping, so a second key for the same provider
    would render as a duplicate row.
    """
    register_provider(DEFAULT_PROVIDER, PersonalDriveProvider)
