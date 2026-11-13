// Runs the crate's tests compiled for wasm32-wasip1 under Node's WASI, for machines where
// native test executables can't run (no MSVC linker, or Windows Smart App Control).
//   cargo test --target wasm32-wasip1 --no-run
//   node tools/run-wasi-tests.mjs [test-harness args]
// web/fixtures is mounted as ./fixtures so the parity test can read the model.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WASI } from 'node:wasi';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const deps = join(root, 'target', 'wasm32-wasip1', 'debug', 'deps');
const builds = readdirSync(deps)
  .filter((f) => /^xray_wasm-[0-9a-f]+\.wasm$/.test(f))
  .map((f) => join(deps, f))
  .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
if (!builds.length) {
  console.error('No test build found. Run: cargo test --target wasm32-wasip1 --no-run');
  process.exit(2);
}
const wasi = new WASI({
  version: 'preview1',
  args: ['xray_wasm', '--nocapture', ...process.argv.slice(2)],
  env: { RUST_TEST_THREADS: '1' },
  preopens: { fixtures: join(root, 'fixtures') },
  returnOnExit: true,
});
const module = await WebAssembly.compile(readFileSync(builds[0]));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exitCode = wasi.start(instance);
