"""差异追溯：发布证据留存、版本逐分钟对比、不可追因的旧版。

规则对应：
* 每次发布保留当时参与计算的读数、证书与逐分钟结果（evidence）；
* 后续导入/改证不得改写旧版依据；
* 同探头同 UTC 日的两个版本可逐分钟对照，区分读数/证书/两者共同变化；
* 缺少依据的历史版仍可查询，对比时明确无法追因，且不用现值填补；
* 重算失败不留新证据、不替换旧版。
"""
import json

import pytest

from app import db, services
from app.timeutil import parse_day

from conftest import CSV_HEADER, add_cert, import_csv

DAY = "2024-01-01"
BASE = parse_day(DAY) * 1440
DAY_IDX = BASE // 1440
CMP_URL = "/api/publications/compare"


def _seed_and_publish(client, offset=0.0):
    """Three hot-ish minutes + one certificate, then publish. Returns pub id."""
    import_csv(
        client,
        CSV_HEADER
        + f"P-01,{BASE},8.5\nP-01,{BASE + 1},9.0\nP-01,{BASE + 2},7.0\n",
    )
    add_cert(client, "P-01", BASE, BASE + 1440, offset)
    r = client.post("/api/publications", json={"probe_id": "P-01", "day": DAY})
    assert r.status_code == 200
    return r.json()["publication_id"]


def _publish(client):
    r = client.post("/api/publications", json={"probe_id": "P-01", "day": DAY})
    assert r.status_code == 200, r.text
    return r.json()["publication_id"]


def _compare(client, from_id, to_id, probe="P-01"):
    return client.get(
        CMP_URL,
        params={
            "probe_id": probe,
            "day": DAY,
            "from_id": from_id,
            "to_id": to_id,
        },
    )


def _minute(result, minute):
    for row in result["minutes"]:
        if row["minute_utc"] == minute:
            return row
    raise AssertionError(f"minute {minute} not in comparison")


# --------------------------------------------------------------------------- #
# Evidence retention & immutability
# --------------------------------------------------------------------------- #
def test_publish_retains_readings_certificates_and_results(client):
    pub_id = _seed_and_publish(client, offset=0.2)

    r = client.get(f"/api/publications/P-01/{DAY}?include_evidence=true")
    item = r.json()["publications"][0]
    assert item["publication_id"] == pub_id
    assert item["evidence_available"] is True
    ev = item["evidence"]
    # readings and certificates exactly as committed at publish time
    assert ev["readings"] == [
        {"minute_utc": BASE, "temperature_c": 8.5},
        {"minute_utc": BASE + 1, "temperature_c": 9.0},
        {"minute_utc": BASE + 2, "temperature_c": 7.0},
    ]
    assert ev["certificates"] == [
        {"valid_from": BASE, "valid_to": BASE + 1440, "offset_c": 0.2}
    ]
    # per-minute results: raw, offset, calibrated, over-threshold judgment
    minutes = {m["minute_utc"]: m for m in ev["minutes"]}
    assert minutes[BASE]["raw_c"] == 8.5
    assert minutes[BASE]["offset_c"] == 0.2
    assert minutes[BASE]["calibrated_c"] == 8.7
    assert minutes[BASE]["above_threshold"] is True
    assert minutes[BASE + 2]["calibrated_c"] == 7.2
    assert minutes[BASE + 2]["above_threshold"] is False


def test_later_imports_and_cert_changes_never_rewrite_old_evidence(client):
    pub_id = _seed_and_publish(client, offset=0.2)
    before = client.get(
        f"/api/publications/P-01/{DAY}?include_history=true&include_evidence=true"
    ).json()["publications"][0]
    assert before["publication_id"] == pub_id

    # correction + certificate edit + republish
    import_csv(client, CSV_HEADER + f"P-01,{BASE + 2},9.9\n")
    cert_id = client.get("/api/certificates?probe_id=P-01").json()[
        "certificates"
    ][0]["id"]
    client.patch(f"/api/certificates/{cert_id}", json={"offset_c": -0.5})
    new_id = _publish(client)
    assert new_id != pub_id

    history = client.get(
        f"/api/publications/P-01/{DAY}?include_history=true&include_evidence=true"
    ).json()["publications"]
    old = next(h for h in history if h["publication_id"] == pub_id)
    new = next(h for h in history if h["publication_id"] == new_id)
    # old version's basis is byte-identical to what was retained at publish
    assert old["evidence"] == before["evidence"]
    assert old["status"] == "superseded"
    # new version reflects the corrected reading and the edited certificate
    new_minutes = {m["minute_utc"]: m for m in new["evidence"]["minutes"]}
    assert new_minutes[BASE + 2]["raw_c"] == 9.9
    assert new_minutes[BASE + 2]["offset_c"] == -0.5
    assert new["evidence"]["certificates"][0]["offset_c"] == -0.5


def test_list_marks_evidence_availability_without_embedding_it(client):
    _seed_and_publish(client)
    items = client.get("/api/publications?probe_id=P-01").json()["publications"]
    assert items[0]["evidence_available"] is True
    assert "evidence" not in items[0]  # list responses stay light


# --------------------------------------------------------------------------- #
# Minute-by-minute comparison
# --------------------------------------------------------------------------- #
def test_compare_reading_correction(client):
    v1 = _seed_and_publish(client, offset=0.0)
    import_csv(client, CSV_HEADER + f"P-01,{BASE + 2},9.5\n")  # 7.0 -> 9.5
    v2 = _publish(client)

    r = _compare(client, v1, v2)
    assert r.status_code == 200
    body = r.json()
    assert body["traceable"] is True
    assert body["from"]["publication_id"] == v1
    assert body["to"]["publication_id"] == v2

    row = _minute(body, BASE + 2)
    assert row["change_type"] == "reading"
    assert row["from"]["raw_c"] == 7.0 and row["to"]["raw_c"] == 9.5
    assert row["from"]["offset_c"] == 0.0 and row["to"]["offset_c"] == 0.0
    assert row["from"]["above_threshold"] is False
    assert row["to"]["above_threshold"] is True
    assert row["status_changed"] is True
    # untouched minutes are classified as unchanged
    assert _minute(body, BASE)["change_type"] == "none"

    assert body["summary"]["minutes_changed"] == 1
    assert body["summary"]["reading_changes"] == 1
    assert body["summary"]["certificate_changes"] == 0
    assert body["summary"]["both_changes"] == 0
    assert body["changed_periods"] == [
        {
            "change_type": "reading",
            "start_minute": BASE + 2,
            "end_minute": BASE + 3,
            "start_utc": "2024-01-01T00:02Z",
            "end_utc": "2024-01-01T00:03Z",
            "minutes": 1,
            "duration_minutes": 1,
        }
    ]


def test_compare_certificate_change(client):
    v1 = _seed_and_publish(client, offset=0.0)
    cert_id = client.get("/api/certificates?probe_id=P-01").json()[
        "certificates"
    ][0]["id"]
    assert client.patch(
        f"/api/certificates/{cert_id}", json={"offset_c": 0.5}
    ).status_code == 200
    v2 = _publish(client)

    body = _compare(client, v1, v2).json()
    assert body["traceable"] is True
    assert body["summary"]["certificate_changes"] == 3
    assert body["summary"]["reading_changes"] == 0
    row = _minute(body, BASE)
    assert row["change_type"] == "certificate"
    assert row["from"]["raw_c"] == row["to"]["raw_c"] == 8.5
    assert row["from"]["offset_c"] == 0.0 and row["to"]["offset_c"] == 0.5
    assert row["from"]["calibrated_c"] == 8.5 and row["to"]["calibrated_c"] == 9.0
    # all three minutes changed together -> one merged certificate period
    assert [(p["start_minute"], p["end_minute"], p["change_type"])
            for p in body["changed_periods"]] == [
        (BASE, BASE + 3, "certificate")
    ]


def test_compare_reading_and_certificate_change_together(client):
    v1 = _seed_and_publish(client, offset=0.0)
    import_csv(client, CSV_HEADER + f"P-01,{BASE + 1},6.0\n")  # correction
    cert_id = client.get("/api/certificates?probe_id=P-01").json()[
        "certificates"
    ][0]["id"]
    client.patch(f"/api/certificates/{cert_id}", json={"offset_c": 0.3})
    v2 = _publish(client)

    body = _compare(client, v1, v2).json()
    # corrected minute: raw AND offset differ -> both
    assert _minute(body, BASE + 1)["change_type"] == "both"
    # neighbours: only the offset differs
    assert _minute(body, BASE)["change_type"] == "certificate"
    assert _minute(body, BASE + 2)["change_type"] == "certificate"
    assert body["summary"]["both_changes"] == 1
    assert body["summary"]["certificate_changes"] == 2
    # adjacent minutes of different types are separate periods
    assert [(p["change_type"], p["start_minute"], p["end_minute"])
            for p in body["changed_periods"]] == [
        ("certificate", BASE, BASE + 1),
        ("both", BASE + 1, BASE + 2),
        ("certificate", BASE + 2, BASE + 3),
    ]


def test_compare_new_reading_minute_counts_as_reading_change(client):
    v1 = _seed_and_publish(client, offset=0.0)
    import_csv(client, CSV_HEADER + f"P-01,{BASE + 3},9.9\n")  # new minute
    v2 = _publish(client)

    body = _compare(client, v1, v2).json()
    row = _minute(body, BASE + 3)
    assert row["change_type"] == "reading"
    assert row["from"] is None  # no reading in the old version
    assert row["to"]["raw_c"] == 9.9
    assert row["to"]["above_threshold"] is True
    assert row["status_changed"] is True


def test_compare_certificate_gap_appearance(client):
    # v1: certificate covers everything; v2: certificate deleted -> gap
    v1 = _seed_and_publish(client, offset=0.0)
    cert_id = client.get("/api/certificates?probe_id=P-01").json()[
        "certificates"
    ][0]["id"]
    assert client.delete(f"/api/certificates/{cert_id}").status_code == 200
    v2 = _publish(client)

    body = _compare(client, v1, v2).json()
    row = _minute(body, BASE)
    assert row["change_type"] == "certificate"
    assert row["from"]["offset_c"] == 0.0
    assert row["to"]["offset_c"] is None
    assert row["to"]["calibrated_c"] is None
    assert row["to"]["above_threshold"] is None
    assert row["status_changed"] is True


def test_compare_identical_recompute_has_no_changes(client):
    v1 = _seed_and_publish(client, offset=0.0)
    v2 = _publish(client)  # nothing changed in between
    body = _compare(client, v1, v2).json()
    assert body["traceable"] is True
    assert body["summary"]["minutes_changed"] == 0
    assert body["changed_periods"] == []
    assert all(m["change_type"] == "none" for m in body["minutes"])


# --------------------------------------------------------------------------- #
# Legacy versions without retained evidence
# --------------------------------------------------------------------------- #
def _null_out_evidence(publication_id):
    """Simulate a historical version published before evidence retention."""
    conn = db.connect()
    try:
        conn.execute(
            "UPDATE publications SET evidence = NULL WHERE id = ?",
            (publication_id,),
        )
        conn.commit()
    finally:
        conn.close()


def test_legacy_version_without_evidence_stays_queryable(client):
    v1 = _seed_and_publish(client)
    _null_out_evidence(v1)

    # old version still listed / fetchable, flagged as lacking a basis
    history = client.get(
        f"/api/publications/P-01/{DAY}?include_history=true"
    ).json()["publications"]
    legacy = next(h for h in history if h["publication_id"] == v1)
    assert legacy["evidence_available"] is False
    assert legacy["payload"]["segments"]  # payload itself intact


def test_compare_with_legacy_version_is_explicitly_untraceable(client):
    v1 = _seed_and_publish(client)
    _null_out_evidence(v1)
    import_csv(client, CSV_HEADER + f"P-01,{BASE + 2},9.5\n")
    v2 = _publish(client)

    for a, b in ((v1, v2), (v2, v1)):
        r = _compare(client, a, b)
        assert r.status_code == 200
        body = r.json()
        assert body["traceable"] is False
        assert "无法追因" in body["reason"]
        assert f"#{v1}" in body["reason"]
        # no current-value back-fill: no minute rows, no summary
        assert body["minutes"] == []
        assert body["changed_periods"] == []
        assert body["summary"] is None
        assert "readings" not in body and "certificates" not in body


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_compare_rejects_same_version(client):
    v1 = _seed_and_publish(client)
    r = _compare(client, v1, v1)
    assert r.status_code == 400


def test_compare_rejects_versions_of_other_probe_or_day(client):
    v1 = _seed_and_publish(client)
    v2 = _publish(client)
    # version ids are fine, but probe/day selector does not match them
    r = _compare(client, v1, v2, probe="P-OTHER")
    assert r.status_code == 400
    r2 = client.get(
        CMP_URL,
        params={"probe_id": "P-01", "day": "2024-01-02",
                "from_id": v1, "to_id": v2},
    )
    assert r2.status_code == 400


def test_compare_unknown_version_404(client):
    v1 = _seed_and_publish(client)
    r = _compare(client, v1, 9999)
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Failure atomicity: no new evidence, old version kept
# --------------------------------------------------------------------------- #
def test_failed_recompute_leaves_no_new_evidence(service_db, monkeypatch):
    services.import_readings(CSV_HEADER + f"P-01,{BASE},9.0\n")
    services.add_certificate("P-01", BASE, BASE + 1, 0.0)
    services.publish_day("P-01", DAY_IDX)
    services.import_readings(CSV_HEADER + f"P-01,{BASE},9.5\n")  # stale

    conn = db.connect()
    try:
        before = conn.execute(
            "SELECT id, status, evidence FROM publications"
        ).fetchall()
    finally:
        conn.close()
    assert len(before) == 1 and before[0]["evidence"] is not None

    def boom(*_a, **_kw):
        raise services.ComputeError("injected")

    monkeypatch.setattr(services, "compute_day_with_trace", boom)
    with pytest.raises(services.ServiceError):
        services.publish_day("P-01", DAY_IDX)

    conn = db.connect()
    try:
        after = conn.execute(
            "SELECT id, status, evidence FROM publications"
        ).fetchall()
    finally:
        conn.close()
    # exactly the same single row, same evidence, still stale
    assert [r["id"] for r in after] == [before[0]["id"]]
    assert after[0]["status"] == "stale"
    assert after[0]["evidence"] == before[0]["evidence"]


# --------------------------------------------------------------------------- #
# Evidence always derives from the same committed snapshot as the payload
# --------------------------------------------------------------------------- #
def test_evidence_matches_the_same_committed_snapshot(service_db):
    from app.engine import compute_day, Calibration, Reading

    mirror_readings = {BASE: 8.5, BASE + 1: 9.2, BASE + 2: 9.3}
    services.import_readings(
        CSV_HEADER
        + f"P-01,{BASE},8.5\nP-01,{BASE + 1},9.2\nP-01,{BASE + 2},9.3\n"
    )
    services.add_certificate("P-01", BASE, BASE + 3, 0.1)
    services.publish_day("P-01", DAY_IDX)

    # interleave: correction + cert edit + a FAILED import (rolled back)
    mirror_readings[BASE + 1] = 7.0
    services.import_readings(CSV_HEADER + f"P-01,{BASE + 1},7.0\n")
    services.update_certificate(1, offset_c=0.4)
    with pytest.raises(services.ServiceError):
        services.import_readings(CSV_HEADER + f"P-01,{BASE},bad\n")
    services.publish_day("P-01", DAY_IDX)

    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT payload, evidence FROM publications WHERE status = 'serving'"
        ).fetchone()
    finally:
        conn.close()
    evidence = json.loads(row["evidence"])
    payload = json.loads(row["payload"])

    # evidence inputs == mirror of the committed state (failed import absent)
    assert {r["minute_utc"]: r["temperature_c"] for r in evidence["readings"]
            } == mirror_readings
    assert evidence["certificates"] == [
        {"valid_from": BASE, "valid_to": BASE + 3, "offset_c": 0.4}
    ]
    # evidence results == recomputation over the same mirror state
    expected = compute_day(
        "P-01",
        DAY_IDX,
        [Reading(m, t) for m, t in sorted(mirror_readings.items())],
        [Calibration(BASE, BASE + 3, 0.4)],
    )
    assert payload["segments"] == expected["segments"]
    minutes = {m["minute_utc"]: m for m in evidence["minutes"]}
    for seg in expected["segments"]:
        for m in range(seg["start_minute"], seg["end_minute"]):
            assert minutes[m]["above_threshold"] is True
    assert minutes[BASE + 1]["above_threshold"] is False  # 7.0 + 0.4 <= 8
