// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML's WebGPU kernels on a real GPU from Node, through Dawn (Chrome's WebGPU, the `webgpu` package): every
// result's bits against bankML's scalar reference (`browseml_q1_dot_ref` in browseml.wasm), the simple kernel and the
// fast one (eight lanes per row, many activations, the workgroup size for this vendor), then their speed on the
// model's shapes — one activation (writing) and many (reading a prompt).
//   node testing/gpu.mjs [--dawn DIR_WITH_node_modules/webgpu] [--power high-performance|low-power]
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import * as g from "../q1_0.js";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const dawn = arg("--dawn", process.env.DAWN || ".");
const { create, globals } = createRequire(join(dawn, "package.json"))("webgpu");
Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = await gpu.requestAdapter({ powerPreference: arg("--power", "high-performance") });
if (!adapter) { console.log("no WebGPU adapter"); process.exit(2); }
const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
const info = adapter.info, wg = g.workgroupFor(info);
console.log(`adapter: ${info.vendor} · ${info.architecture} · ${info.description} · subgroups ${info.subgroupMinSize}–${info.subgroupMaxSize} · workgroup ${wg}`);

// bankML's reference, from the browseml.wasm beside the page (read from disk here, as the page fetches it)
const { wasi } = await import("../reference/browseml-wasi.js");
const w = wasi({ files: {}, write: () => {} });
const { instance } = await WebAssembly.instantiate(readFileSync(new URL("../reference/browseml.wasm", import.meta.url)), { ...w.imports, browseml: { piece: () => 1, log: () => {} } });
const x = instance.exports; w.bind(x.memory);
const ref = (W, A, rows, n) => {
  const rb = (n / g.QK1_0) * g.Q1_0_BYTES, out = new Uint32Array(rows);
  const yp = x.browseml_alloc(A.length); new Uint8Array(x.memory.buffer, yp, A.length).set(A);
  const xp = x.browseml_alloc(rb);
  for (let r = 0; r < rows; r++) { new Uint8Array(x.memory.buffer, xp, rb).set(W.subarray(r * rb, (r + 1) * rb)); out[r] = x.browseml_q1_dot_ref(n, xp, rb, yp, A.length); }
  return out;
};

let failed = 0;
const check = (what, ok, more = "") => { console.log(`${ok ? "ok  " : "FAIL"} ${what}${more ? " — " + more : ""}`); failed += !ok; };

// exactness: shapes small and layer-like, one and many activations, −128 included (randomCase puts some in)
for (const [rows, n, cols] of [[64, 128, 1], [257, 1024, 3], [512, 2048, 8], [96, 6144, 5]]) {
  const c0 = g.randomCase(rows, n, rows + n);
  const As = Array.from({ length: cols }, (_, i) => g.randomCase(1, n, 1000 + i).A);
  const want = As.map((A) => ref(c0.W, A, rows, n));
  const fast = await g.gpuMatMul(device, g.packQ1(c0.W, rows, n), g.packQ8(As, n), wg);
  let same = 0;
  for (let c = 0; c < cols; c++) for (let r = 0; r < rows; r++) same += fast.bits[c * rows + r] === want[c][r];
  check(`fast kernel ${rows}×${n} × ${cols}: every result bit-exact`, same === rows * cols, `${same}/${rows * cols}`);
  const simple = await g.gpuDot(device, { W: c0.W, A: As[0], rows, n });
  check(`simple kernel ${rows}×${n}: bit-exact`, simple.bits.every((b, r) => b === want[0][r]));
}
// the other workgroup sizes give the same bits (a vendor's size never changes an answer)
{
  const rows = 300, n = 2048, c0 = g.randomCase(rows, n, 7), As = [c0.A, g.randomCase(1, n, 8).A];
  const runs = await Promise.all([32, 64, 128].map((s) => g.gpuMatMul(device, g.packQ1(c0.W, rows, n), g.packQ8(As, n), s)));
  check("workgroups of 32, 64 and 128: the same bits", runs.every((r) => r.bits.every((b, i) => b === runs[0].bits[i])));
}
// the tiled kernel (four tokens per invocation): the untiled kernel's bits, partial tiles included
for (const cols of [1, 3, 5, 9, 16]) {
  const rows = 257, n = 2048, c0 = g.randomCase(rows, n, 11), As = Array.from({ length: cols }, (_, i) => g.randomCase(1, n, 200 + i).A);
  const Wp = g.packQ1(c0.W, rows, n), Ap = g.packQ8(As, n);
  const a = await g.gpuMatMul(device, Wp, Ap, wg), b = await g.gpuMatMul(device, Wp, Ap, wg, true);
  check(`tiled kernel ${rows}×${n} × ${cols}: the untiled kernel's bits`, a.bits.every((x, i) => x === b.bits[i]));
}

// speed on the model's shapes (Bonsai-1.7B: 2048 wide, 6144 in the FFN), best of 5 after a warm-up
const time = async (rows, n, cols, kernel) => {  // kernel: simple, fast or tiled
  const c0 = g.randomCase(rows, n, 3), As = Array.from({ length: cols }, (_, i) => g.randomCase(1, n, 50 + i).A);
  const Wp = g.packQ1(c0.W, rows, n), Ap = g.packQ8(As, n);
  let best = Infinity;
  for (let t = 0; t < 6; t++) {
    const r = kernel === "simple" ? await g.gpuDot(device, { W: c0.W, A: As[0], rows, n }) : await g.gpuMatMul(device, Wp, Ap, wg, kernel === "tiled");
    if (t) best = Math.min(best, r.ms);
  }
  return best;
};
console.log("\nspeed (ms, submit to result in hand; weights already on the card for the fast kernel):");
for (const [rows, n] of [[2048, 2048], [6144, 2048], [2048, 6144]]) {
  const s = await time(rows, n, 1, "simple"), f1 = await time(rows, n, 1, "fast"), f16 = await time(rows, n, 16, "fast"), f64 = await time(rows, n, 64, "fast");
  const gmac = (cols, ms) => ((rows * n * cols) / (ms / 1000) / 1e9).toFixed(1);
  const t64 = await time(rows, n, 64, "tiled");
  console.log(`  ${rows}×${n}: simple ${s.toFixed(2)} · fast 1 token ${f1.toFixed(2)} (${gmac(1, f1)} GMAC/s) · 16 tokens ${f16.toFixed(2)} (${gmac(16, f16)}) · 64 tokens ${f64.toFixed(2)} (${gmac(64, f64)}) · tiled 64 tokens ${t64.toFixed(2)} (${gmac(64, t64)})`);
}
device.destroy();
console.log(failed ? `\n${failed} check(s) failed` : "\nevery check passed");
process.exit(failed ? 1 : 0);
