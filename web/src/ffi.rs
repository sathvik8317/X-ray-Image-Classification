//! C ABI for the browser, without wasm-bindgen.
//!
//! JS calls `alloc` to get a buffer in wasm memory, copies bytes in, then calls `load`
//! (model) or `classify` (image). Results stay in a single-threaded global and are read
//! back with the getters. Every call returns 0 on success; on failure `error_ptr` and
//! `error_len` describe the error as UTF-8.

use std::cell::RefCell;

use crate::{Classifier, Prediction, DISPLAY_SIZE};

#[derive(Default)]
struct State {
    classifier: Option<Classifier>,
    prediction: Option<Prediction>,
    error: String,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// # Safety
/// `ptr` must come from `alloc` with this `len`, and the caller must not use it after.
unsafe fn take(ptr: *mut u8, len: usize) -> Vec<u8> {
    Vec::from_raw_parts(ptr, len, len)
}

fn status<T>(result: Result<T, impl std::fmt::Display>, store: impl FnOnce(&mut State, T)) -> i32 {
    STATE.with_borrow_mut(|state| match result {
        Ok(value) => {
            store(state, value);
            0
        }
        Err(e) => {
            state.error = e.to_string();
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Load the ONNX model; frees the buffer.
///
/// # Safety
/// `ptr`/`len` must be a buffer from `alloc`, filled by the caller.
#[no_mangle]
pub unsafe extern "C" fn load(ptr: *mut u8, len: usize, input_size: u32, threshold: f32) -> i32 {
    let bytes = take(ptr, len);
    let result = Classifier::new(&bytes, input_size as usize, threshold);
    status(result, |s, c| s.classifier = Some(c))
}

/// Classify an encoded JPEG or PNG; frees the buffer.
///
/// # Safety
/// `ptr`/`len` must be a buffer from `alloc`, filled by the caller.
#[no_mangle]
pub unsafe extern "C" fn classify(ptr: *mut u8, len: usize) -> i32 {
    let bytes = take(ptr, len);
    let result = STATE.with_borrow(|s| match &s.classifier {
        Some(c) => c.classify(&bytes).map_err(|e| e.to_string()),
        None => Err("model not loaded".to_string()),
    });
    status(result, |s, p| s.prediction = Some(p))
}

fn with_prediction<T>(f: impl FnOnce(&Prediction) -> T, default: T) -> T {
    STATE.with_borrow(|s| s.prediction.as_ref().map(f).unwrap_or(default))
}

#[no_mangle]
pub extern "C" fn prob() -> f32 {
    with_prediction(|p| p.prob, f32::NAN)
}

#[no_mangle]
pub extern "C" fn display_size() -> u32 {
    DISPLAY_SIZE as u32
}

/// RGBA overlay of the last prediction, `display_size()` squared pixels.
#[no_mangle]
pub extern "C" fn overlay_ptr() -> *const u8 {
    with_prediction(|p| p.overlay.as_ptr(), std::ptr::null())
}

#[no_mangle]
pub extern "C" fn error_ptr() -> *const u8 {
    STATE.with_borrow(|s| s.error.as_ptr())
}

#[no_mangle]
pub extern "C" fn error_len() -> usize {
    STATE.with_borrow(|s| s.error.len())
}

/// getrandom hook for `wasm32-unknown-unknown`. Only random ONNX ops would need it.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[no_mangle]
unsafe extern "Rust" fn __getrandom_v03_custom(
    _dest: *mut u8,
    _len: usize,
) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}
