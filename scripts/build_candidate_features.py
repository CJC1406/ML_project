"""Build candidate rows, aligned labels and features for every training and student-testing trip.

Shared by all five models so preprocessing is identical across the comparison.
    export PATHSNAP_DATA_ROOT="/path/to/Student folder"
    python3 scripts/build_candidate_features.py
Writes data/processed/candidate_features/<split>_<trip>.npz (gitignored; rebuild in ~1 min).
Each file: X (rows x features), y (1 = true arc), obs (GPS index), cand (arc id), true_arc per GPS point,
row_labelled (False where the reference route is >50 m away and the point is excluded).
"""
import os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from pathsnap.features.candidate_features import cache_all

root = os.environ.get("PATHSNAP_DATA_ROOT") or (
    os.path.dirname(os.path.normpath(os.environ["PATHSNAP_TRAIN_ROOT"])) if os.environ.get("PATHSNAP_TRAIN_ROOT") else None)
if not root:
    sys.exit("Set PATHSNAP_DATA_ROOT to the folder that contains train/ and student_test/")
cache_all(root, os.path.join(REPO, "data", "processed", "candidate_features"))
print("done")
