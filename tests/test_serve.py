"""The model server's HTTP API, through Flask's test client: a corpus track
streamed one fix per request, the retry and ordering rules, and finish
returning the benchmark's routes."""

from pathlib import Path

import pytest

from serve.app import create_app

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ("train", "00000005")


@pytest.fixture(scope="module")
def client():
    if not (ROOT / "results" / "hmm" / "parameters.json").exists():
        pytest.skip("run `python -m hmm` first")
    app = create_app(ROOT / "data" / "corpus", ROOT / "results", ROOT / "src",
                     streaming_workers=1, batch_workers=1)
    return app.test_client()


def test_stream_and_finish(client) -> None:
    row = next(r for r in client.get("/v1/corpus").get_json()
               if (r["split"], r["id"]) == FOLDER)
    fixes = client.get(f"/v1/corpus/{FOLDER[0]}/{FOLDER[1]}/track").get_json()["fixes"]
    opened = client.post("/v1/sessions", json={
        "map_id": row["map_id"], "models": ["hmm", "mcm"],
        "options": {"truth": {"corpus": {"split": FOLDER[0], "id": FOLDER[1]}}}})
    assert opened.status_code == 201
    sid = opened.get_json()["session_id"]

    committed: list[int] = []
    for seq, fix in enumerate(fixes):
        reply = client.post(f"/v1/sessions/{sid}/fixes", json={"seq": seq, "fixes": [fix]})
        assert reply.status_code == 200, reply.get_json()
        update = reply.get_json()["updates"]["hmm"]
        assert update["committed"]["at"] == len(committed)
        committed += update["committed"]["arcs"]
        assert reply.get_json()["updates"]["mcm"] == {"kept": None}
        if seq == 3:
            again = client.post(f"/v1/sessions/{sid}/fixes", json={"seq": seq, "fixes": [fix]})
            assert again.get_json() == reply.get_json()          # retry: same reply
            gap = client.post(f"/v1/sessions/{sid}/fixes", json={"seq": seq + 9, "fixes": [fix]})
            assert gap.status_code == 409 and gap.get_json()["error"]["expected"] == seq + 1
            stale = client.post(f"/v1/sessions/{sid}/fixes", json={"seq": seq + 1, "fixes": [fix]})
            assert stale.status_code == 400                      # t not after the last
            assert stale.get_json()["error"]["code"] == "invalid_fix"

    done = client.post(f"/v1/sessions/{sid}/finish?wait=30").get_json()
    while done["status"] != "done":
        done = client.get(f"/v1/sessions/{sid}?wait=30").get_json()
    for model in ["hmm", "mcm"]:
        want = (ROOT / "results" / model / "routes" / FOLDER[0]
                / f"{FOLDER[1]}.route").read_text().split()
        assert done["results"][model]["route"] == [int(a) for a in want]
        assert "correct_fraction" in done["results"][model]["score"]
    assert done["results"]["hmm"]["route"][:len(committed)] == committed
    late = client.post(f"/v1/sessions/{sid}/fixes", json={"seq": len(fixes), "fixes": [[8, 49, 1e9]]})
    assert late.status_code == 409
