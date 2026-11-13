import json

import numpy as np
import pytest

from xray_classifier.spectral import frequency_filter, main, spectral_ablation


def _stripes(period: int, size: int = 64) -> np.ndarray:
    x = np.arange(size)
    row = 0.5 + 0.4 * np.sin(2 * np.pi * x / period)
    return np.tile(row, (size, 1))[np.newaxis, :, :, np.newaxis]


def test_lowpass_removes_fine_detail_and_keeps_coarse():
    fine, coarse = _stripes(4), _stripes(32)
    assert frequency_filter(fine, 0.1, "lowpass").std() < 0.05
    assert frequency_filter(coarse, 0.3, "lowpass").std() > 0.2


def test_highpass_removes_coarse_detail_and_keeps_mean():
    coarse = _stripes(32)
    out = frequency_filter(coarse, 0.5, "highpass")
    assert out.std() < 0.1
    assert out.mean() == pytest.approx(coarse.mean(), abs=0.02)


def test_filter_output_shape_dtype_and_range():
    images = np.random.default_rng(0).random((3, 20, 24, 3))
    for mode in ("lowpass", "highpass"):
        out = frequency_filter(images, 0.2, mode)
        assert out.shape == images.shape
        assert out.dtype == np.float32
        assert out.min() >= 0 and out.max() <= 1


def test_filter_rejects_bad_arguments():
    images = np.zeros((1, 8, 8, 1))
    with pytest.raises(ValueError):
        frequency_filter(images, 0.2, "bandpass")
    with pytest.raises(ValueError):
        frequency_filter(images, 0, "lowpass")


def test_spectral_ablation_and_cli(trained_cnn_path, data_dir, tmp_path):
    import keras

    from xray_classifier.data import list_images

    model = keras.models.load_model(trained_cnn_path)
    paths, labels = list_images(data_dir / "test")
    result = spectral_ablation(model, paths, labels, cutoffs=(0.1, 0.5), batch_size=4)
    assert 0 <= result["baseline_auc"] <= 1
    assert len(result["lowpass"]) == len(result["highpass"]) == 2

    main(
        [
            str(trained_cnn_path),
            "--data-dir",
            str(data_dir),
            "--cutoffs",
            "0.1",
            "0.5",
            "--out-dir",
            str(tmp_path),
        ]
    )
    saved = json.loads((tmp_path / "cnn_spectral.json").read_text())
    assert saved["cutoffs"] == [0.1, 0.5]
    assert (tmp_path / "cnn_spectral.png").exists()
    assert (tmp_path / "spectral_examples.png").exists()
