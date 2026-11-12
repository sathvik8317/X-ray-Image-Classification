import json

import numpy as np

from xray_classifier.report import main, pick_gradcam_cases, results_table


def test_pick_gradcam_cases_finds_hits_and_worst_misses():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.7, 0.9, 0.95, 0.6, 0.2])
    cases = dict(pick_gradcam_cases(y_true, y_prob, threshold=0.5))
    assert cases == {
        "Correct: NORMAL": 0,
        "Correct: PNEUMONIA": 3,
        "False positive": 2,  # the more confident of the two false positives
        "False negative": 5,
    }


def test_pick_gradcam_cases_skips_missing_kinds():
    cases = pick_gradcam_cases(np.array([0, 1]), np.array([0.1, 0.9]), threshold=0.5)
    assert [caption for caption, _ in cases] == ["Correct: NORMAL", "Correct: PNEUMONIA"]


def test_results_table_formats_percentages_and_intervals():
    row = {"model": "m", "sensitivity": 0.9, "specificity": 0.8, "accuracy": 0.85,
           "roc_auc": 0.93, "ece": 0.05, "threshold": 0.4,
           "ci95": {"sensitivity": [0.85, 0.95], "roc_auc": [0.9, 0.95]}}  # fmt: skip
    table = results_table([row])
    assert "| m | 90.0 (85.0-95.0) | 80.0 | 85.0 | 0.930 (0.900-0.950) | 0.050 | 0.400 |" in table


def test_cli_writes_table_json_and_figures(data_dir, trained_cnn_path, tmp_path):
    main([str(trained_cnn_path), f"--data-dir={data_dir}", f"--out-dir={tmp_path}",
          "--bootstrap=50"])  # fmt: skip
    rows = json.loads((tmp_path / "results.json").read_text())
    assert rows[0]["model"] == "cnn" and "ci95" in rows[0]
    assert (tmp_path / "results.md").read_text().startswith("| Model |")
    for name in ("roc_comparison", "reliability_comparison", "gradcam_examples"):
        assert (tmp_path / f"{name}.png").exists()
