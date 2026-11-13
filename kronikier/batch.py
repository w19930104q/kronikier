"""Resumable batch manifest for ``--targets-file`` runs.

A batch over dozens of domains must survive two realities of unattended
runs:

1. **A single target can blow up.** A network drop, a malformed CDX
   response, or any unexpected exception used to kill the whole process
   — every later target never ran and nothing recorded which targets
   were already done.
2. **The process can be stopped (Ctrl+C / OOM / reboot).** On the next
   invocation the run has to pick up where it stopped instead of
   re-fetching targets that already finished.

This module owns the on-disk *manifest* that makes both possible. The
manifest is a single JSON file next to the targets file
(``<targets>.manifest.json``, overridable via ``--state-file``), written
atomically after every state transition, so a crash at any moment leaves
either the previous complete document or the next complete document —
never a truncated one.

States per target, recorded in the input file's order:

    waiting      — parsed, never attempted (or reset by --restart)
    running      — scan in progress; a manifest left in this state marks
                   a target whose process died mid-scan, so the next run
                   retries it
    completed    — scan finished and at least one snapshot was fetched
    no_snapshots — scan finished cleanly but the archive holds nothing
                   fetchable for the target (terminal success — nothing
                   would change on a retry)
    failed       — scan raised or the archive backend failed before any
                   snapshot could be fetched; ``error_category`` says why

Terminal successes (``completed`` / ``no_snapshots``) are skipped on
resume; everything else is retried. Nothing here ever touches the
per-target scan outputs (CSV / text / JSON) — those stay exactly as the
single-target CLI produces them.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from kronikier.cli import Target

from kronikier.pipeline import ScanResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State vocabulary
# ---------------------------------------------------------------------------

#: States that count as "done, don't redo this target on the next run".
TERMINAL_SUCCESS_STATES = frozenset(("completed", "no_snapshots"))
#: Every legal state value, in the lifecycle order used for summaries.
STATES = ("waiting", "running", "completed", "no_snapshots", "failed")

#: Error category labels. ``ok`` is the placeholder on non-failed entries.
#:
#: - ``network``        — transport-level failure (connection reset, DNS,
#:                        timeout, HTTP 5xx from CDX/availability API).
#: - ``unexpected``     — anything else that escaped the scan's own
#:                        error handling.
CATEGORY_OK = "ok"
CATEGORY_NETWORK = "network"
CATEGORY_UNEXPECTED = "unexpected"

MANIFEST_VERSION = 1
MANIFEST_SUFFIX = ".manifest.json"


class BatchStateError(Exception):
    """Raised when an existing manifest cannot be read or parsed."""


def default_manifest_path(targets_file: Path) -> Path:
    """Where the manifest lives unless the user overrides it.

    Kept next to the targets file (rather than in CWD) so rerunning the
    same ``--targets-file`` from another directory still resumes.
    """
    return targets_file.with_name(targets_file.name + MANIFEST_SUFFIX)


def _utc_now() -> str:
    """ISO-8601 UTC timestamp, second precision, explicit ``Z`` suffix."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# Entry + outcome classification
# ---------------------------------------------------------------------------


@dataclass
class BatchEntry:
    """One row of the manifest — one merged scan target."""

    domain: str
    #: Extra probe paths carried by the targets-file URL entries. Stored so
    #: a resumed run can prove the on-disk state still matches the input.
    extra_paths: list[str] = field(default_factory=list)
    #: Set for ``--single-url``-derived targets (defensive — batch mode
    #: currently only comes from --targets-file).
    single_url: str | None = None
    state: str = "waiting"
    error_category: str = CATEGORY_OK
    error_message: str | None = None
    contacts: int = 0
    snapshots_fetched: int = 0
    snapshots_considered: int = 0
    attempts: int = 0
    csv_path: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None

    def is_terminal_success(self) -> bool:
        return self.state in TERMINAL_SUCCESS_STATES


def classify_failure(exc: BaseException) -> str:
    """Map an escaped exception onto a manifest error category.

    Everything ``requests`` raises (connection errors, timeouts, HTTP
    errors) is a *network* problem worth retrying later; anything else is
    an *unexpected* local failure.
    """
    if isinstance(exc, requests.RequestException):
        return CATEGORY_NETWORK
    return CATEGORY_UNEXPECTED


def classify_result(result: ScanResult) -> tuple[str, str, str | None]:
    """Classify a returned :class:`ScanResult` into ``(state, category, message)``.

    The scanner swallows most archive problems itself and reports them in
    ``result.errors``; this function turns that into the batch-level
    state vocabulary:

    - at least one snapshot fetched        → ``completed`` (contacts or
      not — a clean zero-contact scan is still a finished scan);
    - nothing fetched, no backend failure  → ``no_snapshots``;
    - nothing fetched, backend did fail    → ``failed`` / ``network``,
      the kind of outcome a later rerun may fix.

    An interrupt is never classified here — the batch runner handles
    :attr:`ScanResult.interrupted` separately so the entry stays
    retryable rather than being mistaken for a terminal state.
    """
    if result.snapshots_fetched > 0:
        return "completed", CATEGORY_OK, None

    backend_errors = [e for e in result.errors if "CDX query failed" in e]
    if backend_errors:
        return "failed", CATEGORY_NETWORK, backend_errors[0]
    return "no_snapshots", CATEGORY_OK, None


# ---------------------------------------------------------------------------
# Manifest container
# ---------------------------------------------------------------------------


class BatchManifest:
    """Ordered, atomically persisted collection of :class:`BatchEntry` rows.

    Construction never performs I/O on its own — use :meth:`load` to
    hydrate from an existing file and :meth:`sync_targets` to align the
    rows with a freshly parsed targets file.
    """

    def __init__(self, path: Path, targets_file: Path):
        self.path = Path(path)
        self.targets_file = Path(targets_file)
        self.entries: list[BatchEntry] = []
        self.created_at: str | None = None

    # ----- (de)serialisation ------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Path,
        targets: list[Target],
        *,
        targets_file: Path,
        restart: bool = False,
    ) -> "BatchManifest":
        """Open the manifest, hydrating from disk when a compatible one exists.

        ``restart=True`` ignores any on-disk state and starts every target
        back at ``waiting``. A corrupt or incompatible file raises
        :class:`BatchStateError` — callers advise the analyst to use
        ``--restart`` rather than silently discarding history.
        """
        manifest = cls(path, targets_file)
        manifest.created_at = _utc_now()

        if path.exists() and not restart:
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                raise BatchStateError(
                    f"cannot read batch manifest {path}: {e}. "
                    f"Re-run with --restart to start the batch over."
                ) from e
            if not isinstance(doc, dict) or doc.get("version") != MANIFEST_VERSION:
                raise BatchStateError(
                    f"batch manifest {path} has an unknown or missing version. "
                    f"Re-run with --restart to start the batch over."
                )
            manifest.created_at = doc.get("created_at") or manifest.created_at
            raw_entries = doc.get("entries")
            if not isinstance(raw_entries, list):
                raise BatchStateError(
                    f"batch manifest {path} is malformed (no entries list). "
                    f"Re-run with --restart to start the batch over."
                )
            known = {
                e["domain"]: BatchEntry(
                    **{
                        k: v
                        for k, v in e.items()
                        if k in BatchEntry.__dataclass_fields__
                    }
                )
                for e in raw_entries
                if isinstance(e, dict) and "domain" in e
            }
            manifest.entries = [
                known[t.domain] if t.domain in known else manifest._make_entry(t)
                for t in targets
            ]
            # Input order is the targets file's order; rows whose domain
            # vanished from the input are dropped by the rebuild above.
        else:
            manifest.entries = [manifest._make_entry(t) for t in targets]

        manifest.sync_targets(targets)
        # Guarantee a complete, parseable manifest exists as soon as a batch
        # opens — even if the process dies before the first scan starts.
        manifest.flush()
        return manifest

    @staticmethod
    def _make_entry(target: Target) -> BatchEntry:
        return BatchEntry(
            domain=target.domain,
            extra_paths=list(target.extra_paths),
            single_url=target.single_url,
        )

    def sync_targets(self, targets: list[Target]) -> None:
        """Align rows with a freshly parsed targets file.

        - New domains are inserted at their input position as ``waiting``.
        - Domains removed from the input drop their rows.
        - A domain whose URL-path hints changed resets to ``waiting``:
          the on-disk state described a different scan, so it can't be
          reused.
        """
        by_domain = {e.domain: e for e in self.entries}
        rebuilt: list[BatchEntry] = []
        for target in targets:
            entry = by_domain.get(target.domain)
            if entry is None:
                rebuilt.append(self._make_entry(target))
                continue
            if (
                list(entry.extra_paths) != list(target.extra_paths)
                or entry.single_url != target.single_url
            ):
                log.info(
                    "target %s changed since the manifest was written; "
                    "resetting to waiting",
                    target.domain,
                )
                rebuilt.append(self._make_entry(target))
            else:
                rebuilt.append(entry)
        self.entries = rebuilt

    # ----- Persistence ------------------------------------------------------

    def to_doc(self) -> dict:
        return {
            "version": MANIFEST_VERSION,
            "saved_at": _utc_now(),
            "created_at": self.created_at,
            "targets_file": str(self.targets_file),
            "entries": [asdict(e) for e in self.entries],
        }

    def flush(self) -> None:
        """Atomically write the whole manifest to :attr:`path`.

        Write to a temp file in the same directory, fsync it, then rename
        over the target. POSIX rename is atomic, so a concurrent reader
        (or a crash mid-write) can only ever see a complete document.
        """
        doc = self.to_doc()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".kronikier-manifest-", suffix=".tmp", dir=str(self.path.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
        except OSError:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    # ----- Lookups ----------------------------------------------------------

    def get(self, domain: str) -> BatchEntry | None:
        for e in self.entries:
            if e.domain == domain:
                return e
        return None

    def pending(self) -> list[BatchEntry]:
        """Rows that still need a scan attempt, in input order."""
        return [e for e in self.entries if not e.is_terminal_success()]

    def counts(self) -> dict[str, int]:
        out = {state: 0 for state in STATES}
        for e in self.entries:
            out[e.state] = out.get(e.state, 0) + 1
        return out

    # ----- State transitions ------------------------------------------------

    def mark_running(self, domain: str) -> BatchEntry:
        entry = self._require(domain)
        entry.state = "running"
        entry.attempts += 1
        entry.error_category = CATEGORY_OK
        entry.error_message = None
        now = _utc_now()
        if entry.started_at is None:
            entry.started_at = now
        entry.updated_at = now
        self.flush()
        return entry

    def mark_result(
        self,
        domain: str,
        result: ScanResult,
        *,
        csv_path: Path | None,
    ) -> BatchEntry:
        """Record a finished, non-interrupted scan outcome and persist it."""
        entry = self._require(domain)
        state, category, message = classify_result(result)
        entry.state = state
        entry.error_category = category
        entry.error_message = message
        entry.contacts = len(result.by_value())
        entry.snapshots_fetched = result.snapshots_fetched
        entry.snapshots_considered = result.snapshots_considered
        entry.csv_path = str(csv_path) if csv_path else None
        now = _utc_now()
        entry.updated_at = now
        entry.completed_at = now
        self.flush()
        return entry

    def mark_exception(self, domain: str, exc: BaseException) -> BatchEntry:
        """Record an escaped exception as a retryable failure and persist it."""
        entry = self._require(domain)
        entry.state = "failed"
        entry.error_category = classify_failure(exc)
        entry.error_message = f"{type(exc).__name__}: {exc}"[:500]
        entry.updated_at = _utc_now()
        self.flush()
        return entry

    def mark_interrupted(self, domain: str, result: ScanResult | None) -> BatchEntry:
        """Record a Ctrl+C stop.

        The entry goes back to ``waiting`` so the next run retries the
        target; whatever partial counters the scan managed to return are
        kept for forensics but never mistaken for a terminal state.
        """
        entry = self._require(domain)
        entry.state = "waiting"
        entry.error_category = CATEGORY_OK
        entry.error_message = "interrupted before completion"
        if result is not None:
            entry.contacts = len(result.by_value())
            entry.snapshots_fetched = result.snapshots_fetched
            entry.snapshots_considered = result.snapshots_considered
        entry.updated_at = _utc_now()
        self.flush()
        return entry

    def _require(self, domain: str) -> BatchEntry:
        entry = self.get(domain)
        if entry is None:  # pragma: no cover — caller bug, not user data
            raise KeyError(f"unknown batch target: {domain!r}")
        return entry
