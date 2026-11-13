"""Unit tests for the resumable batch manifest.

Covers the state machine itself (transitions, ordering, atomic
persistence, resume rules, input-change resets, corrupt-manifest
handling) — CLI-level wiring with mocked archive responses lives in
``test_batch_cli.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from kronikier.batch import (
    CATEGORY_NETWORK,
    CATEGORY_OK,
    CATEGORY_UNEXPECTED,
    MANIFEST_VERSION,
    BatchEntry,
    BatchManifest,
    BatchStateError,
    classify_failure,
    classify_result,
    default_manifest_path,
)
from kronikier.cli import Target
from kronikier.pipeline import ScanResult


def _targets(*specs: str) -> list[Target]:
    out: list[Target] = []
    for spec in specs:
        if "/" in spec:
            host, path = spec.split("/", 1)
            out.append(Target(domain=host, extra_paths=("/" + path,)))
        else:
            out.append(Target(domain=spec))
    return out


def _result(
    domain: str, *, fetched=0, considered=0, contacts=0, errors=None, interrupted=False
) -> ScanResult:
    from kronikier.extractors import Contact
    from kronikier.pipeline import ContactSighting

    sightings = [
        ContactSighting(
            contact=Contact(kind="email", value=f"info@{domain}", raw=f"info@{domain}"),
            snapshot_url=f"https://web.archive.org/web/20200101000000/http://{domain}/",
            timestamp="20200101000000",
            source_url=f"http://{domain}/",
        )
        for _ in range(contacts)
    ]
    return ScanResult(
        domain=domain,
        snapshots_considered=considered,
        snapshots_fetched=fetched,
        sightings=sightings,
        errors=list(errors or []),
        interrupted=interrupted,
    )


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    targets_file = tmp_path / "targets.txt"
    manifest_file = tmp_path / "targets.txt.manifest.json"
    targets_file.write_text("a.example\nb.example\nc.example\n", encoding="utf-8")
    return targets_file, manifest_file


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


class TestClassifyResult:
    def test_fetched_snapshot_is_completed_even_without_contacts(self):
        state, cat, msg = classify_result(
            _result("x.example", fetched=3, considered=5, contacts=0)
        )
        assert state == "completed"
        assert cat == CATEGORY_OK
        assert msg is None

    def test_zero_fetched_no_errors_is_no_snapshots(self):
        state, cat, _ = classify_result(_result("dead.example"))
        assert state == "no_snapshots"
        assert cat == CATEGORY_OK

    def test_cdx_backend_failure_with_nothing_fetched_is_failed_network(self):
        state, cat, msg = classify_result(
            _result(
                "broken.example",
                errors=[
                    "CDX query failed: HTTPError 503 — try a longer `--cdx-timeout`"
                ],
            )
        )
        assert state == "failed"
        assert cat == CATEGORY_NETWORK
        assert "CDX query failed" in msg

    def test_cdx_error_but_snapshot_fetched_still_completes(self):
        # Partial backend trouble must not poison an otherwise successful scan.
        state, cat, _ = classify_result(
            _result("x.example", fetched=1, errors=["CDX query failed: boom"])
        )
        assert state == "completed"
        assert cat == CATEGORY_OK

    def test_per_snapshot_fetch_errors_alone_are_no_snapshots(self):
        # Fetcher-level errors live in result.errors too, but without the
        # CDX marker and with zero fetched pages the archive simply had
        # nothing usable — a retryable backend failure would carry the
        # CDX marker; plain fetch failures here mean no snapshots reached us.
        state, cat, _ = classify_result(
            _result("x.example", errors=["http://x.example/ @ 2020: HTTP 404"])
        )
        assert state == "no_snapshots"
        assert cat == CATEGORY_OK


class TestClassifyFailure:
    def test_request_exception_is_network(self):
        assert classify_failure(requests.ConnectionError("reset")) == CATEGORY_NETWORK
        assert classify_failure(requests.Timeout("slow")) == CATEGORY_NETWORK

    def test_anything_else_is_unexpected(self):
        assert classify_failure(ValueError("nope")) == CATEGORY_UNEXPECTED
        assert classify_failure(RuntimeError("boom")) == CATEGORY_UNEXPECTED


def test_default_manifest_path_sits_next_to_targets_file(tmp_path):
    tf = tmp_path / "lists" / "targets.txt"
    assert default_manifest_path(tf) == tmp_path / "lists" / "targets.txt.manifest.json"


# ---------------------------------------------------------------------------
# Manifest lifecycle
# ---------------------------------------------------------------------------


class TestManifestLifecycle:
    def test_fresh_manifest_creates_waiting_rows_in_input_order(self, paths):
        targets_file, manifest_file = paths
        m = BatchManifest.load(
            manifest_file,
            _targets("a.example", "b.example", "c.example"),
            targets_file=targets_file,
        )
        assert [e.domain for e in m.entries] == ["a.example", "b.example", "c.example"]
        assert all(e.state == "waiting" for e in m.entries)
        assert all(e.attempts == 0 for e in m.entries)
        assert manifest_file.exists()
        doc = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert doc["version"] == MANIFEST_VERSION
        assert [e["domain"] for e in doc["entries"]] == [
            "a.example",
            "b.example",
            "c.example",
        ]

    def test_running_then_completed_transitions_persist(self, paths):
        targets_file, manifest_file = paths
        m = BatchManifest.load(
            manifest_file, _targets("a.example"), targets_file=targets_file
        )
        m.mark_running("a.example")
        assert m.get("a.example").state == "running"
        assert m.get("a.example").attempts == 1

        m.mark_result(
            "a.example", _result("a.example", fetched=2, contacts=1), csv_path=None
        )
        entry = m.get("a.example")
        assert entry.state == "completed"
        assert entry.contacts == 1
        assert entry.snapshots_fetched == 2
        assert entry.completed_at is not None
        assert entry.error_category == CATEGORY_OK

        # Reload from disk — state must survive the process boundary.
        m2 = BatchManifest.load(
            manifest_file, _targets("a.example"), targets_file=targets_file
        )
        assert m2.get("a.example").state == "completed"
        assert m2.get("a.example").attempts == 1
        assert m2.get("a.example").contacts == 1

    def test_no_snapshots_is_terminal_success(self, paths):
        targets_file, manifest_file = paths
        m = BatchManifest.load(
            manifest_file, _targets("dead.example"), targets_file=targets_file
        )
        m.mark_running("dead.example")
        m.mark_result("dead.example", _result("dead.example"), csv_path=None)
        assert m.get("dead.example").state == "no_snapshots"
        assert m.get("dead.example").is_terminal_success()
        assert m.pending() == []

    def test_exception_marks_failed_with_category_and_keeps_batch_runnable(self, paths):
        targets_file, manifest_file = paths
        m = BatchManifest.load(
            manifest_file,
            _targets("a.example", "b.example"),
            targets_file=targets_file,
        )
        m.mark_running("a.example")
        entry = m.mark_exception("a.example", requests.ConnectionError("network down"))
        assert entry.state == "failed"
        assert entry.error_category == CATEGORY_NETWORK
        assert "ConnectionError" in entry.error_message
        # Failed stays retryable, i.e. not a terminal success.
        assert not entry.is_terminal_success()
        assert [e.domain for e in m.pending()] == ["a.example", "b.example"]

    def test_interrupt_resets_to_waiting_so_resume_retries(self, paths):
        targets_file, manifest_file = paths
        m = BatchManifest.load(
            manifest_file, _targets("a.example"), targets_file=targets_file
        )
        m.mark_running("a.example")
        partial = _result("a.example", fetched=1, contacts=1, interrupted=True)
        entry = m.mark_interrupted("a.example", partial)
        assert entry.state == "waiting"
        assert entry.error_message == "interrupted before completion"
        # Partial forensic counters are kept.
        assert entry.contacts == 1
        assert entry.snapshots_fetched == 1
        assert entry.attempts == 1
        assert not entry.is_terminal_success()

    def test_interrupt_without_result_also_resets(self, paths):
        targets_file, manifest_file = paths
        m = BatchManifest.load(
            manifest_file, _targets("a.example"), targets_file=targets_file
        )
        m.mark_running("a.example")
        entry = m.mark_interrupted("a.example", None)
        assert entry.state == "waiting"

    def test_attempts_increment_on_each_run(self, paths):
        targets_file, manifest_file = paths
        for _ in range(3):
            m = BatchManifest.load(
                manifest_file, _targets("flaky.example"), targets_file=targets_file
            )
            m.mark_running("flaky.example")
            m.mark_exception("flaky.example", requests.Timeout("slow"))
        again = BatchManifest.load(
            manifest_file, _targets("flaky.example"), targets_file=targets_file
        )
        assert again.get("flaky.example").attempts == 3


# ---------------------------------------------------------------------------
# Resume semantics
# ---------------------------------------------------------------------------


class TestResume:
    def test_terminal_successes_are_skipped_pending_retried(self, paths):
        targets_file, manifest_file = paths
        targets = _targets("a.example", "b.example", "c.example")

        m = BatchManifest.load(manifest_file, targets, targets_file=targets_file)
        m.mark_running("a.example")
        m.mark_result(
            "a.example", _result("a.example", fetched=1, contacts=1), csv_path=None
        )
        m.mark_running("b.example")
        m.mark_exception("b.example", requests.ConnectionError("boom"))
        # c stays waiting.

        m2 = BatchManifest.load(manifest_file, targets, targets_file=targets_file)
        pending = [e.domain for e in m2.pending()]
        assert pending == ["b.example", "c.example"]
        assert m2.get("a.example").state == "completed"
        # Input order is preserved across serialization.
        assert [e.domain for e in m2.entries] == ["a.example", "b.example", "c.example"]

    def test_stale_running_entry_from_dead_process_is_retried(self, paths):
        targets_file, manifest_file = paths
        targets = _targets("a.example")
        m = BatchManifest.load(manifest_file, targets, targets_file=targets_file)
        m.mark_running("a.example")
        # Simulate a hard crash: process gone, manifest left at "running".
        m2 = BatchManifest.load(manifest_file, targets, targets_file=targets_file)
        assert m2.get("a.example").state == "running"
        assert not m2.get("a.example").is_terminal_success()
        assert [e.domain for e in m2.pending()] == ["a.example"]

    def test_new_target_appended_in_input_position(self, paths):
        targets_file, manifest_file = paths
        old = _targets("a.example", "c.example")
        m = BatchManifest.load(manifest_file, old, targets_file=targets_file)
        m.mark_running("a.example")
        m.mark_result("a.example", _result("a.example", fetched=1), csv_path=None)

        new = _targets("a.example", "b.example", "c.example")
        m2 = BatchManifest.load(manifest_file, new, targets_file=targets_file)
        assert [e.domain for e in m2.entries] == ["a.example", "b.example", "c.example"]
        assert m2.get("a.example").state == "completed"  # history kept
        assert m2.get("b.example").state == "waiting"
        assert m2.get("c.example").state == "waiting"

    def test_removed_target_drops_its_row(self, paths):
        targets_file, manifest_file = paths
        m = BatchManifest.load(
            manifest_file,
            _targets("a.example", "b.example"),
            targets_file=targets_file,
        )
        m.mark_running("a.example")
        m.mark_result("a.example", _result("a.example", fetched=1), csv_path=None)

        m2 = BatchManifest.load(
            manifest_file,
            _targets("b.example"),
            targets_file=targets_file,
        )
        assert [e.domain for e in m2.entries] == ["b.example"]

    def test_changed_extra_paths_reset_entry_to_waiting(self, paths):
        targets_file, manifest_file = paths
        m = BatchManifest.load(
            manifest_file,
            _targets("a.example/contact"),
            targets_file=targets_file,
        )
        m.mark_running("a.example")
        m.mark_result("a.example", _result("a.example", fetched=1), csv_path=None)
        assert m.get("a.example").state == "completed"

        # Same domain, different probe path hints → the old state described
        # a different scan; it must not be reused.
        m2 = BatchManifest.load(
            manifest_file,
            _targets("a.example/team"),
            targets_file=targets_file,
        )
        entry = m2.get("a.example")
        assert entry.state == "waiting"
        assert entry.extra_paths == ["/team"]

    def test_restart_flag_wipes_history(self, paths):
        targets_file, manifest_file = paths
        targets = _targets("a.example", "b.example")
        m = BatchManifest.load(manifest_file, targets, targets_file=targets_file)
        m.mark_running("a.example")
        m.mark_result("a.example", _result("a.example", fetched=2), csv_path=None)
        m.mark_running("b.example")
        m.mark_exception("b.example", RuntimeError("boom"))

        m2 = BatchManifest.load(
            manifest_file,
            targets,
            targets_file=targets_file,
            restart=True,
        )
        assert all(e.state == "waiting" for e in m2.entries)
        assert all(e.attempts == 0 for e in m2.entries)
        assert len(m2.pending()) == 2


# ---------------------------------------------------------------------------
# Persistence guarantees
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_flush_is_atomic_no_tmp_left_behind(self, paths):
        targets_file, manifest_file = paths
        m = BatchManifest.load(
            manifest_file,
            _targets("a.example"),
            targets_file=targets_file,
        )
        m.mark_running("a.example")
        m.mark_result("a.example", _result("a.example", fetched=1), csv_path=None)
        leftovers = list(manifest_file.parent.glob(".kronikier-manifest-*.tmp"))
        assert leftovers == []
        # Document is complete, parseable JSON at every observed point.
        doc = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert doc["entries"][0]["state"] == "completed"

    def test_corrupt_manifest_raises_recoverable_error(self, paths):
        targets_file, manifest_file = paths
        manifest_file.write_text("{not json", encoding="utf-8")
        with pytest.raises(BatchStateError, match="--restart"):
            BatchManifest.load(
                manifest_file,
                _targets("a.example"),
                targets_file=targets_file,
            )

    def test_unknown_version_raises(self, paths):
        targets_file, manifest_file = paths
        manifest_file.write_text(
            json.dumps({"version": 999, "entries": []}),
            encoding="utf-8",
        )
        with pytest.raises(BatchStateError, match="version"):
            BatchManifest.load(
                manifest_file,
                _targets("a.example"),
                targets_file=targets_file,
            )

    def test_flush_to_unwritable_location_propagates_oserror(self, tmp_path):
        targets_file = tmp_path / "t.txt"
        targets_file.touch()
        bad_dir = tmp_path / "no-such-dir"
        manifest = BatchManifest(bad_dir / "x.json", targets_file)
        manifest.entries = [BatchEntry(domain="a.example")]
        # Parent creation succeeds (mkdir -p); to force a real failure make
        # the parent a regular file.
        bad_dir.write_text("x", encoding="utf-8")
        with pytest.raises(OSError):
            manifest.flush()

    def test_counts_reports_every_state(self, paths):
        targets_file, manifest_file = paths
        m = BatchManifest.load(
            manifest_file,
            _targets("a.example", "b.example", "c.example", "d.example", "e.example"),
            targets_file=targets_file,
        )
        m.mark_running("a.example")
        m.mark_result("a.example", _result("a.example", fetched=1), csv_path=None)
        m.mark_running("b.example")
        m.mark_result("b.example", _result("b.example"), csv_path=None)  # no snapshots
        m.mark_running("c.example")
        m.mark_exception("c.example", RuntimeError("x"))
        # d waiting, e: leave running
        m.mark_running("e.example")
        counts = m.counts()
        assert counts["completed"] == 1
        assert counts["no_snapshots"] == 1
        assert counts["failed"] == 1
        assert counts["waiting"] == 1
        assert counts["running"] == 1
