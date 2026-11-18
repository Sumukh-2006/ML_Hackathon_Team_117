# UE24CS352A — Machine Learning Hackathon

**Team Number:** 117
**Date:** 18 November 2026 (Wednesday)
**Course:** UE24CS352A — Machine Learning

---

## Schedule & Timings

| Time | Activity |
|---|---|
| 8:15 – 8:45 AM | Quiz — CSE |
| 9:00 – 9:30 AM | Quiz — AIML |
| 9:30 AM | **Problem Statement Release** (sharp) — work may begin immediately |
| 11:30 AM – 12:40 PM | First round of evaluation |
| 12:45 PM | Lunch break |
| 1:15 PM | Report to venue for Leadership Address |
| 1:30 PM | Leadership Address |
| After address | Return to respective classrooms and continue working |
| 4:30 PM | Evaluation and demonstration |
| May extend past 5:30 PM | Depends on evaluation time per batch |

**Notes**
- Students proceed to their regular classrooms for the hackathon after the quiz.
- Evaluation may run beyond 5:30 PM — plan accordingly.

---

## Contacts

- **Dr. Ashwini M Joshi** — Professor, Dept. of CSE, PES University (ashwinimjoshi@pes.edu)
- **Dr. Divyashree N** — Associate Professor, Dept. of CSE (divyashreen@pes.edu)
- **IQAC CSE RR Campus** — iqaccs.rr@pes.edu

---

## Folder Layout

```
ml_hackathon_team_117/
├── README.md          # This file — schedule, checklist, team info
├── app/               # FastAPI backend + evaluator UI (see app/README.md)
├── data/              # Provided datasets (raw + processed)
├── notebooks/         # Working notebooks / scripts
├── submissions/       # Final submission files, per-round versions
├── notes/             # Problem statement, approach notes, evaluation feedback
├── tools/             # Screenshot helpers for the report
├── train/             # 80 training trips
└── student_test/      # 10 student-testing trips
```

## Demo App (API + UI)

Run it from the repo root:

```bash
pip install -r app/requirements.txt
python -m app.build_payloads          # once, ~60 s: one payload per trip
python -m uvicorn app.server:app --port 8077
# http://127.0.0.1:8077/
```

The provided `train/`, `student_test/` and `data/` folders are **not committed**
(course-provided inputs, ~120 MB). The generated payloads under
`app/web/data/` are committed, so the interface runs on a fresh clone;
`POST /api/match` and `build_payloads` need the raw folders present.

The UI draws the road network, the reference route, the snapped path and the GPS
fixes on one sheet, then scores the whole drive. `POST /api/match` re-runs the
matcher live so an evaluator can see a real response and its wall-clock time;
`GET /api/methodology` states what each number counts and what is N/A. Full
details in [`app/README.md`](app/README.md).

Screenshots for the report land in `notes/ui/` — regenerate with
`python tools/shoot.py` and `python tools/interactions.py` while the server is
running. `python -m app.test_smoke` checks the scoring invariants the sheet
claims.

## Team Roster

| Name | SRN | Role | Contact |
|---|---|---|---|
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

## Pre-Hackathon Checklist

- [ ] Laptop charged + charger packed
- [ ] Environment ready (Python, Jupyter, key libraries: numpy, pandas, scikit-learn, xgboost/lightgbm, matplotlib, seaborn)
- [ ] Offline docs / saved snippets for common preprocessing & model pipelines
- [ ] Dataset handling plan (download vs. provided USB/portal)
- [ ] Backup: cloud notebook access + local copy
- [ ] Team communication channel decided

## Problem Statement

> _To be filled in at 9:30 AM._

**Objective:**

**Dataset:**

**Evaluation metric:**

**Deliverables:**

## Approach Log

| Time | Decision / Finding | Owner |
|---|---|---|
|  |  |  |

## Round 1 Evaluation Feedback (11:30 AM – 12:40 PM)

-

## Final Submission (4:30 PM Demo)

- **Submission file:**
- **Demo plan (2 min pitch):**
  1.
  2.
  3.
- **Known limitations / next steps:**

## Results

-
