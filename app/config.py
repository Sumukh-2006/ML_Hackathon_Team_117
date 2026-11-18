"""Paths and constants for the PathSnap demo app.

One place for every path so the server, the payload builder and the tests agree.
Nothing here is learned from data, so nothing here can leak across partitions.
"""
from __future__ import annotations

import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)          # repo root (ml_hackathon_team_117/)
WEB_DIR = os.path.join(APP_DIR, "web")
DATA_DIR = os.path.join(WEB_DIR, "data")  # generated payloads, committed

TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR = os.path.join(ROOT, "student_test")

# --- matcher knobs (same values as notes/baseline_check.py so numbers agree) ---
CELL = 100.0          # m, spatial-hash cell
RING = 2              # cells searched each way -> exact for anything < 200 m
BBOX_MARGIN = 1000.0  # m; an arc further out cannot be the argmin for any fix
CORRIDOR = 650.0      # m; only draw arcs this close to the drive (payload size)
COORD_DP = 5          # 5 dp of a degree is ~1.1 m, finer than the GPS noise

METHOD = "nearest-arc"
