import keras
import numpy as np
import pytest

from xray_classifier.export import with_cam_output
from xray_classifier.gradcam import gradcam
from xray_classifier.models import build_custom_cnn, build_efficientnet_v2_b0


@pytest.fixture(scope="module")
def pooled_model():
    return build_efficientnet_v2_b0(64, weights=None)


def test_cam_output_matches_model_and_gradcam(pooled_model):
    image = np.random.default_rng(0).random((64, 64, 3)).astype("float32")
    prob, cam = with_cam_output(pooled_model)(image[np.newaxis])
    prob, cam = np.asarray(prob), np.asarray(cam)[0, ..., 0]
    np.testing.assert_allclose(prob, pooled_model(image[np.newaxis]), rtol=1e-5)

    # The same post-processing the browser does: sign by predicted class, ReLU, normalize.
    signed = cam if prob[0, 0] >= 0.5 else -cam
    signed = np.maximum(signed, 0)
    signed /= signed.max() + keras.config.epsilon()
    expected = gradcam(pooled_model, image)
    small = keras.ops.convert_to_numpy(
        keras.ops.image.resize(signed[..., np.newaxis], expected.shape)
    )[..., 0]
    np.testing.assert_allclose(small, expected, atol=1e-3)


def test_cam_output_rejects_flatten_models():
    with pytest.raises(ValueError):
        with_cam_output(build_custom_cnn(32))


def test_export_onnx(pooled_model, tmp_path):
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    from xray_classifier.export import export_onnx

    model_path = tmp_path / "m.keras"
    pooled_model.save(model_path)
    meta = export_onnx(model_path, tmp_path / "m.onnx")
    assert meta["input_size"] == 64 and meta["outputs"] == ["prob", "cam"]

    image = np.random.default_rng(1).random((1, 64, 64, 3)).astype("float32")
    session = ort.InferenceSession(str(tmp_path / "m.onnx"))
    outputs = session.run(None, {session.get_inputs()[0].name: image})
    prob = next(o for o in outputs if o.shape == (1, 1))
    np.testing.assert_allclose(prob, pooled_model(image), atol=1e-4)
