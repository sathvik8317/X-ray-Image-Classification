//! Chest X-ray pneumonia classifier for the browser.
//!
//! Runs the ONNX model from `xray-export` with tract and turns its CAM output into a
//! heatmap overlay, all on the client: no server, no upload. The model has two outputs,
//! the pneumonia probability (1, 1) and a raw class activation map (1, h, w, 1); see
//! `src/xray_classifier/export.py` for why that map equals Grad-CAM for this model.
//!
//! Preprocessing copies the Python pipeline (`tf.image.resize`, bilinear with half-pixel
//! centers, no antialiasing, then / 255) so the scores match the Keras model.
//! The `ffi` module exposes a small C ABI for `wasm32-unknown-unknown`, so the page
//! needs no wasm-bindgen: see `www/app.js`.

use tract_onnx::prelude::*;

mod ffi;

/// Side length of the overlay image returned by [`Classifier::classify`].
pub const DISPLAY_SIZE: usize = 384;
/// Heatmap opacity, the same as the Python `gradcam_overlay`.
pub const OVERLAY_ALPHA: f32 = 0.4;

type Model = Arc<TypedRunnableModel>;

pub struct Classifier {
    model: Model,
    input_size: usize,
    threshold: f32,
}

pub struct Prediction {
    /// Pneumonia probability in [0, 1].
    pub prob: f32,
    pub pneumonia: bool,
    /// Raw CAM from the model, row-major, `cam_size` x `cam_size`.
    pub cam: Vec<f32>,
    pub cam_size: usize,
    /// Heatmap in [0, 1] for the predicted class at `DISPLAY_SIZE`, as Grad-CAM shows it.
    pub heatmap: Vec<f32>,
    /// X-ray blended with the `jet` heatmap, RGBA, `DISPLAY_SIZE` x `DISPLAY_SIZE`.
    pub overlay: Vec<u8>,
}

impl Classifier {
    pub fn new(onnx: &[u8], input_size: usize, threshold: f32) -> TractResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(onnx))?
            .with_input_fact(0, f32::fact([1, input_size, input_size, 3]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model, input_size, threshold })
    }

    pub fn classify(&self, image_bytes: &[u8]) -> TractResult<Prediction> {
        let rgb = image::load_from_memory(image_bytes)?.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let pixels: Vec<f32> = rgb.into_raw().into_iter().map(f32::from).collect();

        let n = self.input_size;
        let input: Vec<f32> = resize_bilinear(&pixels, h, w, 3, n, n)
            .into_iter()
            .map(|v| v / 255.0)
            .collect();
        let tensor = tract_ndarray::Array4::from_shape_vec((1, n, n, 3), input)?.into_tensor();
        let outputs = self.model.run(tvec!(tensor.into()))?;

        let (mut prob, mut cam, mut cam_size) = (None, None, 0);
        for out in outputs.iter() {
            let view = out.to_plain_array_view::<f32>()?;
            match view.ndim() {
                2 => prob = view.iter().next().copied(),
                4 => {
                    cam_size = view.shape()[1];
                    cam = Some(view.iter().copied().collect::<Vec<f32>>());
                }
                _ => {}
            }
        }
        let prob = prob.ok_or_else(|| TractError::msg("model has no (1, 1) probability output"))?;
        let cam = cam.ok_or_else(|| TractError::msg("model has no (1, h, w, 1) CAM output"))?;
        let pneumonia = prob >= self.threshold;

        let small = normalized_heatmap(&cam, pneumonia);
        let heatmap = resize_bilinear(&small, cam_size, cam_size, 1, DISPLAY_SIZE, DISPLAY_SIZE);
        let gray: Vec<f32> = pixels.iter().step_by(3).copied().collect();
        let gray = resize_bilinear(&gray, h, w, 1, DISPLAY_SIZE, DISPLAY_SIZE);
        let overlay = blend_overlay(&gray, &heatmap, OVERLAY_ALPHA);
        Ok(Prediction { prob, pneumonia, cam, cam_size, heatmap, overlay })
    }
}

/// Grad-CAM post-processing: point the map at the predicted class, ReLU, scale to [0, 1].
pub fn normalized_heatmap(cam: &[f32], pneumonia: bool) -> Vec<f32> {
    let sign = if pneumonia { 1.0 } else { -1.0 };
    let relu: Vec<f32> = cam.iter().map(|v| (sign * v).max(0.0)).collect();
    let max = relu.iter().copied().fold(0.0f32, f32::max);
    relu.iter().map(|v| v / (max + 1e-7)).collect()
}

/// `tf.image.resize(method="bilinear")`: half-pixel centers, no antialiasing.
/// `src` is row-major (height, width, channels).
pub fn resize_bilinear(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    channels: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let ys = interpolation_weights(in_h, out_h);
    let xs = interpolation_weights(in_w, out_w);
    let mut out = Vec::with_capacity(out_h * out_w * channels);
    for &(y0, y1, fy) in &ys {
        for &(x0, x1, fx) in &xs {
            for c in 0..channels {
                let at = |y: usize, x: usize| src[(y * in_w + x) * channels + c];
                let top = at(y0, x0) + (at(y0, x1) - at(y0, x0)) * fx;
                let bottom = at(y1, x0) + (at(y1, x1) - at(y1, x0)) * fx;
                out.push(top + (bottom - top) * fy);
            }
        }
    }
    out
}

fn interpolation_weights(in_size: usize, out_size: usize) -> Vec<(usize, usize, f32)> {
    let scale = in_size as f32 / out_size as f32;
    (0..out_size)
        .map(|i| {
            let pos = (i as f32 + 0.5) * scale - 0.5;
            let floor = pos.floor();
            let lower = floor.max(0.0) as usize;
            let upper = (pos.ceil().max(0.0) as usize).min(in_size - 1);
            (lower, upper, pos - floor)
        })
        .collect()
}

/// matplotlib's `jet` colormap (its piecewise-linear segment data), x in [0, 1].
pub fn jet(x: f32) -> [f32; 3] {
    const RED: &[(f32, f32)] = &[(0.0, 0.0), (0.35, 0.0), (0.66, 1.0), (0.89, 1.0), (1.0, 0.5)];
    const GREEN: &[(f32, f32)] =
        &[(0.0, 0.0), (0.125, 0.0), (0.375, 1.0), (0.64, 1.0), (0.91, 0.0), (1.0, 0.0)];
    const BLUE: &[(f32, f32)] = &[(0.0, 0.5), (0.11, 1.0), (0.34, 1.0), (0.65, 0.0), (1.0, 0.0)];
    let x = x.clamp(0.0, 1.0);
    let channel = |points: &[(f32, f32)]| {
        let i = points.windows(2).position(|p| x <= p[1].0).unwrap_or(points.len() - 2);
        let ((x0, y0), (x1, y1)) = (points[i], points[i + 1]);
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    };
    [channel(RED), channel(GREEN), channel(BLUE)]
}

/// RGBA bytes: `gray` (0-255) blended with `jet(heatmap)` at opacity `alpha`.
pub fn blend_overlay(gray: &[f32], heatmap: &[f32], alpha: f32) -> Vec<u8> {
    gray.iter()
        .zip(heatmap)
        .flat_map(|(&g, &h)| {
            let base = g / 255.0;
            let [r, gr, b] = jet(h);
            let mix = |c: f32| (((1.0 - alpha) * base + alpha * c).clamp(0.0, 1.0) * 255.0) as u8;
            [mix(r), mix(gr), mix(b), 255]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_identity_keeps_values() {
        let src: Vec<f32> = (0..12).map(|v| v as f32).collect();
        assert_eq!(resize_bilinear(&src, 2, 2, 3, 2, 2), src);
    }

    #[test]
    fn resize_matches_tensorflow_half_pixel_upsampling() {
        // tf.image.resize([[0, 4]] as a 1x2 image, (1, 4)) == [0, 1, 3, 4]
        let out = resize_bilinear(&[0.0, 4.0], 1, 2, 1, 1, 4);
        assert_eq!(out, vec![0.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn resize_downsampling_averages_neighbours() {
        // 4 -> 2 samples at positions 0.5 and 2.5.
        let out = resize_bilinear(&[0.0, 2.0, 4.0, 6.0], 1, 4, 1, 1, 2);
        assert_eq!(out, vec![1.0, 5.0]);
    }

    #[test]
    fn jet_endpoints_and_middle() {
        assert_eq!(jet(0.0), [0.0, 0.0, 0.5]);
        assert_eq!(jet(1.0), [0.5, 0.0, 0.0]);
        let [r, g, b] = jet(0.5);
        assert!(r > 0.4 && r < 0.6 && g == 1.0 && b < 0.5);
    }

    #[test]
    fn heatmap_follows_predicted_class() {
        let cam = [-2.0, 1.0, 0.5, -1.0];
        assert_eq!(normalized_heatmap(&cam, true)[1], 1.0 / (1.0 + 1e-7));
        assert_eq!(normalized_heatmap(&cam, true)[0], 0.0);
        assert!((normalized_heatmap(&cam, false)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn overlay_is_rgba() {
        let out = blend_overlay(&[0.0, 255.0], &[0.0, 1.0], 0.4);
        assert_eq!(out.len(), 8);
        assert_eq!(out[3], 255);
        assert_eq!(&out[..3], &[0, 0, 51]);
    }

    /// Parity with Keras on real test images. Needs `fixtures/` from the Kaggle export
    /// (model.onnx, model_reference.json and the referenced X-rays); skipped otherwise.
    #[test]
    fn matches_keras_reference_outputs() {
        let dir = std::path::Path::new("fixtures");
        let (Ok(onnx), Ok(reference)) =
            (std::fs::read(dir.join("model.onnx")), std::fs::read(dir.join("model_reference.json")))
        else {
            eprintln!("fixtures missing, skipping parity test");
            return;
        };
        let reference: serde_json::Value = serde_json::from_slice(&reference).unwrap();
        let classifier = Classifier::new(&onnx, 224, 0.5).unwrap();
        let files = reference["files"].as_array().unwrap();
        for (i, file) in files.iter().enumerate() {
            let bytes = std::fs::read(dir.join(file.as_str().unwrap())).unwrap();
            let p = classifier.classify(&bytes).unwrap();
            let want = reference["prob"][i].as_f64().unwrap() as f32;
            assert!((p.prob - want).abs() < 0.02, "{file}: prob {} vs keras {want}", p.prob);
            let want_cam: Vec<f32> = reference["cam"][i]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32))
                .collect();
            let scale = want_cam.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let worst = p.cam.iter().zip(&want_cam).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(worst < 0.05 * scale, "{file}: CAM differs by {worst} (scale {scale})");
            println!("{file}: prob {:.4} (keras {want:.4}), CAM max diff {worst:.4}", p.prob);
        }
    }
}
