// Runs the Rust/wasm classifier off the main thread. The wasm module has no imports and a
// tiny C ABI (see web/src/ffi.rs): alloc -> copy bytes in -> load / classify -> getters.
let wasm;

function call(fn, bytes, ...rest) {
  const ptr = wasm.alloc(bytes.length);
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  if (fn(ptr, bytes.length, ...rest) !== 0) {
    const msg = new Uint8Array(wasm.memory.buffer, wasm.error_ptr(), wasm.error_len());
    throw new Error(new TextDecoder().decode(msg));
  }
}

async function init() {
  const [module, model, meta] = await Promise.all([
    WebAssembly.instantiateStreaming(fetch('xray_wasm.wasm'), {}),
    fetch('model/model.onnx').then((r) => r.arrayBuffer()),
    fetch('model/model.json').then((r) => r.json()),
  ]);
  wasm = module.instance.exports;
  const start = performance.now();
  call(wasm.load, new Uint8Array(model), meta.input_size, meta.threshold);
  return { meta, loadMs: performance.now() - start };
}

const ready = init();

self.onmessage = async ({ data }) => {
  try {
    const { meta, loadMs } = await ready;
    if (data.type === 'init') {
      self.postMessage({ type: 'ready', meta, loadMs });
      return;
    }
    const start = performance.now();
    call(wasm.classify, new Uint8Array(data.bytes));
    const size = wasm.display_size();
    const overlay = new Uint8ClampedArray(
      wasm.memory.buffer.slice(wasm.overlay_ptr(), wasm.overlay_ptr() + size * size * 4),
    );
    self.postMessage(
      { type: 'result', prob: wasm.prob(), threshold: meta.threshold, size, overlay,
        ms: performance.now() - start },
      [overlay.buffer],
    );
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err.message || err) });
  }
};
