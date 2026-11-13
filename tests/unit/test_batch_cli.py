"""CLI-level regression for resumable ``--targets-file`` batches.

The archive is fully local here: every CDX / availability / playback call
is served by ``responses`` with the committed HTML fixtures, so the tests
need no network and stay deterministic.

Coverage demanded by the batch-resilience feature:

- mixed success (contacts) + zero-hit (no snapshots) + an escaped
  exception in the same batch, with the failing target NOT blocking the
  targets behind it;
- Ctrl+C mid-batch: partial outputs kept, manifest atomically flushed,
  exit code 130, and a second invocation that skips finished targets and
  completes the rest without re-fetching them;
- exit-code split: 0 = whole batch reached a terminal success,
  1 = run finished but some targets failed, 130 = interrupted;
- no cross-target result bleed and no duplicate scans on resume;
- ``--restart`` and ``--state-file`` semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
import responses

from kronikier import cli
from kronikier.calibration import CALIBRATION_VERSION, Calibration
from kronikier.cdx import CDX_ENDPOINT
from kronikier.extractors import Contact
from kronikier.pipeline import ContactSighting, ScanResult, scan_domain

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _calibration() -> Calibration:
    return Calibration(
        version=CALIBRATION_VERSION,
        avg_latency_s=0.42,
        sample_count=8,
        last_calibrated_at="2026-01-01T00:00:00+00:00",
        samples_p50=0.4,
        samples_p95=0.5,
        user_agent="test",
    )


def _cdx_body(rows: list[list[str]]) -> str:
    header = [
        "urlkey",
        "timestamp",
        "original",
        "mimetype",
        "statuscode",
        "digest",
        "length",
    ]
    return json.dumps([header, *rows])


def _synthetic_result(
    domain: str, *, interrupted: bool = False, with_contact: bool = False
) -> ScanResult:
    sightings = []
    if with_contact:
        sightings.append(
            ContactSighting(
                contact=Contact(
                    kind="email", value=f"partial@{domain}", raw=f"partial@{domain}"
                ),
                snapshot_url=f"https://web.archive.org/web/20200101000000/http://{domain}/",
                timestamp="20200101000000",
                source_url=f"http://{domain}/",
            )
        )
    return ScanResult(
        domain=domain,
        snapshots_considered=2,
        snapshots_fetched=1 if with_contact else 0,
        sightings=sightings,
        errors=[],
        interrupted=interrupted,
    )


# ---------------------------------------------------------------------------
# Local archive mock: CDX preflight + main query + availability + playback
# ---------------------------------------------------------------------------

CONTACT_DOMAIN = "romashka-llc.ru"
EMPTY_DOMAIN = "archived-empty.example"
FAIL_DOMAIN = "explodes.example"
INTERRUPT_DOMAIN = "ctrlc.example"


def _register_archive_mock() -> None:
    """Serve two real (fixture-backed) domains entirely from local data."""
    contact_url = f"http://{CONTACT_DOMAIN}/contacts.html"
    contact_ts = "20070815120000"

    def cdx_cb(request):
        url = request.url
        if "showNumPages" in url:
            return (200, {}, "1")  # small site → planner takes precise count
        if "fl=urlkey" in url:
            if EMPTY_DOMAIN in url:
                # Header row only → zero captures → no_captures short-circuit.
                return (200, {}, json.dumps([["urlkey"]]))
            return (200, {}, json.dumps([["urlkey"], ["x"]]))
        # Main CDX scan query.
        if EMPTY_DOMAIN in url:
            raise AssertionError(f"empty domain must not run a main CDX scan: {url}")
        return (
            200,
            {},
            _cdx_body(
                [
                    [
                        f"ru,{CONTACT_DOMAIN})/contacts.html",
                        contact_ts,
                        contact_url,
                        "text/html",
                        "200",
                        "X2",
                        "20",
                    ],
                ]
            ),
        )

    responses.add_callback(responses.GET, CDX_ENDPOINT, callback=cdx_cb)
    # Probe is disabled in the runs below, but register a harmless response
    # anyway so an accidental probe call can't surface a ConnectionError.
    responses.add(
        responses.GET,
        "https://archive.org/wayback/available",
        json={"archived_snapshots": {}},
        status=200,
    )
    responses.add(
        responses.GET,
        f"https://web.archive.org/web/{contact_ts}id_/{contact_url}",
        body=(FIXTURES / "contacts_ru_2007.html").read_text(encoding="utf-8"),
        status=200,
        content_type="text/html; charset=utf-8",
    )


def _real_scan(domain: str, **kwargs) -> ScanResult:
    """Invoke the genuine pipeline (its HTTP is mocked by ``responses``)."""
    return scan_domain(
        domain,
        timeout_seconds=kwargs.get("timeout_seconds", 120.0),
        force_all=kwargs.get("force_all", False),
        calibration=kwargs.get("calibration"),
        probe_well_known=False,
        rate_limit_per_sec=0,
        default_phone_regions=("RU",),
        no_escalate=True,
        session=kwargs.get("session"),
        cache=kwargs.get("cache"),
    )


def _write_targets(tmp_path: Path) -> Path:
    tf = tmp_path / "targets.txt"
    tf.write_text(
        f"{CONTACT_DOMAIN}\n{EMPTY_DOMAIN}\n{FAIL_DOMAIN}\n{INTERRUPT_DOMAIN}\n",
        encoding="utf-8",
    )
    return tf


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _states(doc: dict) -> dict[str, str]:
    return {e["domain"]: e["state"] for e in doc["entries"]}


# ---------------------------------------------------------------------------
# Mixed success + zero-hit + exception + Ctrl+C, then resume
# ---------------------------------------------------------------------------


@responses.activate
def test_batch_mixed_outcomes_then_resume_after_ctrl_c(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "ensure_calibration", lambda **kw: _calibration())
    _register_archive_mock()
    targets_file = _write_targets(tmp_path)
    manifest_path = tmp_path / "state" / "batch.json"  # custom --state-file dir

    calls: dict[str, int] = {}

    def fake_scan(domain, **kwargs):
        calls[domain] = calls.get(domain, 0) + 1
        if domain in (CONTACT_DOMAIN, EMPTY_DOMAIN):
            return _real_scan(domain, **kwargs)
        if domain == FAIL_DOMAIN:
            raise requests.ConnectionError("simulated network drop")
        if domain == INTERRUPT_DOMAIN:
            # First run: Ctrl+C lands during this target's scan.
            return _synthetic_result(domain, interrupted=True, with_contact=True)
        raise AssertionError(f"unexpected domain: {domain}")

    monkeypatch.setattr(cli, "scan_domain", fake_scan)
    monkeypatch.chdir(tmp_path)

    # ----- Run 1: interrupts on the 4th target -----
    rc1 = cli.main(
        [
            "--targets-file",
            str(targets_file),
            "--state-file",
            str(manifest_path),
            "--no-progress",
            "--no-cache",
        ]
    )
    assert rc1 == 130

    assert manifest_path.exists(), "manifest must be atomically flushed on Ctrl+C"
    doc = _load_manifest(manifest_path)
    states = _states(doc)
    assert states == {
        CONTACT_DOMAIN: "completed",
        EMPTY_DOMAIN: "no_snapshots",
        FAIL_DOMAIN: "failed",
        INTERRUPT_DOMAIN: "waiting",  # interrupted → retryable, not terminal
    }
    # Error category recorded on the failed row.
    failed_row = next(e for e in doc["entries"] if e["domain"] == FAIL_DOMAIN)
    assert failed_row["error_category"] == "network"
    assert "ConnectionError" in failed_row["error_message"]
    # Interrupted row keeps partial forensic counters but stays retryable.
    interrupted_row = next(e for e in doc["entries"] if e["domain"] == INTERRUPT_DOMAIN)
    assert interrupted_row["contacts"] == 1
    assert interrupted_row["error_message"] == "interrupted before completion"
    # Manifest order follows the input file.
    assert [e["domain"] for e in doc["entries"]] == [
        CONTACT_DOMAIN,
        EMPTY_DOMAIN,
        FAIL_DOMAIN,
        INTERRUPT_DOMAIN,
    ]
    # The failure did NOT block the target behind it (the interrupt target ran).
    assert calls[INTERRUPT_DOMAIN] == 1
    # First three targets were each attempted exactly once in run 1.
    assert calls[CONTACT_DOMAIN] == 1
    assert calls[EMPTY_DOMAIN] == 1
    assert calls[FAIL_DOMAIN] == 1

    # Partial CSV from the interrupted target was still written.
    partial_csvs = list(tmp_path.glob(f"{INTERRUPT_DOMAIN}_*.csv"))
    assert partial_csvs and f"partial@{INTERRUPT_DOMAIN}" in partial_csvs[0].read_text()
    # Successful target's CSV uses the unchanged per-domain format.
    contact_csvs = list(tmp_path.glob(f"{CONTACT_DOMAIN}_*.csv"))
    assert contact_csvs
    header = contact_csvs[0].read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.startswith("kind,value,value_human,value_raw,first_seen,last_seen")
    # The manifest must point at the artefact actually written — not a
    # re-derived timestamped name that doesn't exist on disk.
    contact_row = next(e for e in doc["entries"] if e["domain"] == CONTACT_DOMAIN)
    assert contact_row["csv_path"] is not None
    assert Path(contact_row["csv_path"]).exists()
    assert Path(contact_row["csv_path"]).read_text(
        encoding="utf-8-sig"
    ) == contact_csvs[0].read_text(encoding="utf-8-sig")

    # ----- Run 2: resume — finished targets skipped, rest retried -----
    def resume_scan(domain, **kwargs):
        calls[domain] = calls.get(domain, 0) + 1
        if domain == FAIL_DOMAIN:
            # Transient failure has cleared up on the rerun.
            return _synthetic_result(domain, with_contact=True)
        if domain == INTERRUPT_DOMAIN:
            return _synthetic_result(domain, with_contact=True)
        raise AssertionError(f"resume must not rescan a finished target: {domain}")

    monkeypatch.setattr(cli, "scan_domain", resume_scan)
    capsys.readouterr()  # drain run-1 output so the assertions see run-2 only

    rc2 = cli.main(
        [
            "--targets-file",
            str(targets_file),
            "--state-file",
            str(manifest_path),
            "--no-progress",
            "--no-cache",
        ]
    )
    assert rc2 == 0  # whole batch now at terminal success

    doc2 = _load_manifest(manifest_path)
    states2 = _states(doc2)
    assert all(s in ("completed", "no_snapshots") for s in states2.values())
    assert states2[EMPTY_DOMAIN] == "no_snapshots"
    assert states2[FAIL_DOMAIN] == "completed"
    assert states2[INTERRUPT_DOMAIN] == "completed"

    # No duplicate fetching: the two run-1 successes were never rescanned.
    assert calls[CONTACT_DOMAIN] == 1
    assert calls[EMPTY_DOMAIN] == 1
    assert calls[FAIL_DOMAIN] == 2
    assert calls[INTERRUPT_DOMAIN] == 2

    err2 = capsys.readouterr().err
    assert "already finished" in err2
    assert "skip" in err2 and "completed" in err2
    assert "Batch summary" in err2


@responses.activate
def test_batch_exit_code_1_when_a_target_keeps_failing(monkeypatch, tmp_path, capsys):
    """A finished run with any target still failed must exit 1, distinct from
    both full success (0) and interruption (130)."""
    monkeypatch.setattr(cli, "ensure_calibration", lambda **kw: _calibration())
    _register_archive_mock()
    targets_file = _write_targets(tmp_path)
    manifest_path = tmp_path / "batch.manifest.json"

    def fake_scan(domain, **kwargs):
        if domain in (CONTACT_DOMAIN, EMPTY_DOMAIN):
            return _real_scan(domain, **kwargs)
        if domain == FAIL_DOMAIN:
            raise requests.ConnectionError("still down")
        if domain == INTERRUPT_DOMAIN:
            return _synthetic_result(domain, with_contact=True)
        raise AssertionError(domain)

    monkeypatch.setattr(cli, "scan_domain", fake_scan)
    monkeypatch.chdir(tmp_path)

    rc = cli.main(
        [
            "--targets-file",
            str(targets_file),
            "--state-file",
            str(manifest_path),
            "--no-progress",
            "--no-cache",
        ]
    )
    assert rc == 1
    states = _states(_load_manifest(manifest_path))
    assert states[FAIL_DOMAIN] == "failed"
    # Targets behind the persistent failure still completed — no abort.
    assert states[INTERRUPT_DOMAIN] == "completed"
    assert "1 failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# No cross-target result bleed
# ---------------------------------------------------------------------------


@responses.activate
def test_batch_json_outputs_are_isolated_per_target(monkeypatch, tmp_path, capsys):
    """Each target's JSON report may contain only its own contacts — the
    refactor that shares one aggregation list across the loop would fail."""
    monkeypatch.setattr(cli, "ensure_calibration", lambda **kw: _calibration())
    _register_archive_mock()
    targets_file = tmp_path / "targets.txt"
    targets_file.write_text(f"{CONTACT_DOMAIN}\n{EMPTY_DOMAIN}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    rc = cli.main(
        [
            "--targets-file",
            str(targets_file),
            "--no-progress",
            "--no-cache",
            "--no-csv",
            "--json",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    # Two separate JSON documents (one per target), parse them sequentially.
    docs = []
    decoder = json.JSONDecoder()
    rest = out.lstrip()
    while rest:
        obj, end = decoder.raw_decode(rest)
        docs.append(obj)
        rest = rest[end:].lstrip()

    assert len(docs) == 2
    by_domain = {d["domain"]: d for d in docs}
    assert set(by_domain) == {CONTACT_DOMAIN, EMPTY_DOMAIN}
    contact_emails = {c["value"] for c in by_domain[CONTACT_DOMAIN]["contacts"]}
    assert "info@romashka-llc.ru" in contact_emails
    # The empty target's document carries none of the other target's data.
    assert by_domain[EMPTY_DOMAIN]["contacts"] == []
    assert by_domain[EMPTY_DOMAIN]["domain"] == EMPTY_DOMAIN


# ---------------------------------------------------------------------------
# --restart and flag validation
# ---------------------------------------------------------------------------


def test_restart_resets_every_target_and_rescans(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ensure_calibration", lambda **kw: _calibration())
    targets_file = tmp_path / "targets.txt"
    targets_file.write_text("a.example\nb.example\n", encoding="utf-8")
    manifest_path = tmp_path / "m.json"

    # Seed a manifest whose rows are all terminal successes.
    from kronikier.batch import BatchManifest

    manifest = BatchManifest.load(
        manifest_path,
        [cli.Target(domain="a.example"), cli.Target(domain="b.example")],
        targets_file=targets_file,
    )
    manifest.mark_running("a.example")
    manifest.mark_result(
        "a.example", _synthetic_result("a.example", with_contact=True), csv_path=None
    )
    manifest.mark_running("b.example")
    manifest.mark_result(
        "b.example", _synthetic_result("b.example", with_contact=True), csv_path=None
    )

    calls: list[str] = []

    def fake_scan(domain, **kwargs):
        calls.append(domain)
        return _synthetic_result(domain, with_contact=True)

    monkeypatch.setattr(cli, "scan_domain", fake_scan)

    rc = cli.main(
        [
            "--targets-file",
            str(targets_file),
            "--state-file",
            str(manifest_path),
            "--restart",
            "--no-progress",
            "--no-cache",
            "--no-csv",
        ]
    )
    assert rc == 0
    # --restart ignores the terminal states and scans both targets again.
    assert calls == ["a.example", "b.example"]
    doc = _load_manifest(manifest_path)
    assert all(e["attempts"] == 1 for e in doc["entries"])
    assert all(e["state"] == "completed" for e in doc["entries"])


def test_restart_and_state_file_rejected_without_targets_file(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "ensure_calibration", lambda **kw: _calibration())
    rc = cli.main(["a.example", "--restart", "--no-progress", "--no-cache"])
    assert rc == 2
    assert "--restart" in capsys.readouterr().err

    state = tmp_path / "s.json"
    rc = cli.main(
        [
            "a.example",
            "--state-file",
            str(state),
            "--no-progress",
            "--no-cache",
        ]
    )
    assert rc == 2
    assert "--state-file" in capsys.readouterr().err


def test_corrupt_manifest_is_rejected_with_restart_hint(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "ensure_calibration", lambda **kw: _calibration())
    targets_file = tmp_path / "targets.txt"
    targets_file.write_text("a.example\n", encoding="utf-8")
    manifest_path = tmp_path / "targets.txt.manifest.json"
    manifest_path.write_text("{broken json", encoding="utf-8")

    rc = cli.main(
        [
            "--targets-file",
            str(targets_file),
            "--no-progress",
            "--no-cache",
            "--no-csv",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "--restart" in err


def test_default_manifest_location_sits_next_to_targets_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ensure_calibration", lambda **kw: _calibration())
    targets_file = tmp_path / "lists" / "targets.txt"
    targets_file.parent.mkdir(parents=True)
    targets_file.write_text("a.example\n", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "scan_domain",
        lambda domain, **kw: _synthetic_result(domain, with_contact=True),
    )
    # Run from a different CWD to prove the path tracks the targets file.
    monkeypatch.chdir(tmp_path)
    rc = cli.main(
        [
            "--targets-file",
            str(targets_file),
            "--no-progress",
            "--no-cache",
            "--no-csv",
        ]
    )
    assert rc == 0
    assert (targets_file.parent / "targets.txt.manifest.json").exists()
