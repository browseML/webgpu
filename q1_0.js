// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML on the GPU, exactly: bankML's Q1_0 · q8_0 dot product as a WebGPU (WGSL) kernel, held bit for bit to the
// engine's own scalar reference (`q1_0::vec_dot_ref`, the arithmetic every CPU path matches), and a test that proves it
// on this GPU before any of browseML's work is given to it.
//
// The arithmetic, as the reference does it (bankML/q1_0.rs):
//   per 128 weights: d0 (f16) and 128 sign bits; per 32 activations: d1 (f16) and 32 int8
//   ab[l] = d1·s (k = 0) or fma(d1, s, ab[l]) (k = 1..3)  — d1·s is exact in f32 (11 + 10 bits), so plain multiply
//                                                            then add is the same bits fused or not
//   acc[l] = fma(d0, ab[l], acc[l])                        — NOT exact unfused: WGSL's fma() may round twice, so this
//                                                            is bankML's fma_exact (bankML/gpu/spirv.rs): the product
//                                                            split into two exact halves, TwoSum, and Boldo and
//                                                            Melquiond's round-to-odd (2008) — a correctly rounded FMA
//                                                            from plain f32 operations on any GPU
//   the result: ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7))
// f16 scales are decoded with integer operations (exact, subnormals included), not the hardware's conversion.

export const Q1_0_BYTES = 18, Q8_0_BYTES = 34, QK1_0 = 128, QK8_0 = 32;

export const Q1_DOT_WGSL = /* wgsl */ `
struct P { n: u32, rows: u32, zero: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> A: array<u32>;
@group(0) @binding(2) var<storage, read_write> O: array<u32>;
@group(0) @binding(3) var<uniform> p: P;

fn wb(o: u32) -> u32 { return (W[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }
fn ab8(o: u32) -> u32 { return (A[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }

// f16 bits to f32, exactly
fn f16(h: u32) -> f32 {
  let sgn = (h & 0x8000u) << 16u;
  let e = (h >> 10u) & 0x1fu;
  let m = h & 0x3ffu;
  if (e == 0u) {
    let v = f32(m) * 5.9604644775390625e-8;  // m · 2^-24: exact (m < 2^10, a power-of-two scale)
    return select(v, -v, sgn != 0u);
  }
  if (e == 31u) { return bitcast<f32>(sgn | 0x7f800000u | (m << 13u)); }
  return bitcast<f32>(sgn | ((e + 112u) << 23u) | (m << 13u));
}

// a value the compiler cannot see through: OR with a zero it only learns at run time (the uniform's \`zero\`). WGSL
// cannot mark operations exact (SPIR-V's NoContraction); without this, a driver's float simplifier (Mesa's NIR on
// AMD) may turn (a + b) − a into b and lose TwoSum's error term
fn hide(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) | p.zero); }

// Knuth's TwoSum: s = a + b rounded, and the exact error
fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = hide(a + b);
  let bb = hide(s - a);
  let sb = hide(s - bb);
  return vec2<f32>(s, hide(a - sb) + hide(b - bb));
}

// fma(a, b, c) correctly rounded, from plain operations; a carries at most 12 significant bits (an f16 does)
fn fma_exact(a: f32, b: f32, c: f32) -> f32 {
  let bhi = bitcast<f32>(bitcast<u32>(b) & 0xfffff000u);
  let blo = b - bhi;
  let u = two_sum(a * bhi, a * blo);
  let t = two_sum(c, u.x);
  let v = two_sum(t.y, u.y);
  // round v to odd: when inexact and its last bit is even, one ulp toward what was lost
  var vb = bitcast<u32>(v.x);
  if ((vb & 1u) == 0u && v.y != 0.0) {
    vb = select(vb - 1u, vb + 1u, (v.x > 0.0) == (v.y > 0.0));
  }
  return t.x + bitcast<f32>(vb);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= p.rows) { return; }
  let nb = p.n / 128u;
  var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var i = 0u; i < nb; i++) {
    let xo = (r * nb + i) * 18u;
    let d0 = f16(wb(xo) | (wb(xo + 1u) << 8u));
    var abv = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var k = 0u; k < 4u; k++) {
      let yo = (i * 4u + k) * 34u;
      let d1 = f16(ab8(yo) | (ab8(yo + 1u) << 8u));
      for (var l = 0u; l < 8u; l++) {
        var s: i32 = 0;
        for (var e = 4u * l; e < 4u * l + 4u; e++) {
          let qb = ab8(yo + 2u + e);
          let q = i32(qb) - select(0, 256, qb >= 128u);
          let bit = (wb(xo + 2u + k * 4u + e / 8u) >> (e % 8u)) & 1u;
          // the reference negates as an i8 (wrapping): -(-128) stays -128
          let nq = select(-q, -128, q == -128);
          s += select(nq, q, bit == 1u);
        }
        let prod = d1 * f32(s);
        abv[l] = select(prod + abv[l], prod, k == 0u);
      }
    }
    for (var l = 0u; l < 8u; l++) { acc[l] = fma_exact(d0, abv[l], acc[l]); }
  }
  let h = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
  O[r] = bitcast<u32>(h);
}`;

/**
 * The fast kernel, still exact: bankML's `q1_0_mat_vec8` in WGSL, with a second dimension for many activations (a
 * prompt's tokens). Weights and activations repacked once (`packQ1`, `packQ8`): per weight block its scale as f32 and
 * its 128 bits as four u32; per activation its scales as f32 and its i8 quants four to a word. Eight invocations per
 * (row, activation), one per accumulation lane of the CPU kernel — chains independent there too — then lane 0 adds the
 * eight in the CPU's order. `WG` (the workgroup size, a multiple of 8) follows the vendor: 64 for AMD's wavefronts,
 * 32 for NVIDIA's warps (`workgroupFor`). Output: O[col · rows + row], the f32 bits of `vec_dot_ref`.
 */
export const Q1_MM_WGSL = /* wgsl */ `
override WG: u32 = 64u;
struct P { rows: u32, nb: u32, cols: u32, zero: u32 }
@group(0) @binding(0) var<storage, read> Wd: array<f32>;
@group(0) @binding(1) var<storage, read> Wb: array<u32>;
@group(0) @binding(2) var<storage, read> Ad: array<f32>;
@group(0) @binding(3) var<storage, read> Aq: array<u32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> part: array<f32, WG>;

fn hide(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) | p.zero); }  // see Q1_DOT_WGSL's hide
fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = hide(a + b);
  let bb = hide(s - a);
  let sb = hide(s - bb);
  return vec2<f32>(s, hide(a - sb) + hide(b - bb));
}
fn fma_exact(a: f32, b: f32, c: f32) -> f32 {
  let bhi = bitcast<f32>(bitcast<u32>(b) & 0xfffff000u);
  let blo = b - bhi;
  let u = two_sum(a * bhi, a * blo);
  let t = two_sum(c, u.x);
  let v = two_sum(t.y, u.y);
  var vb = bitcast<u32>(v.x);
  if ((vb & 1u) == 0u && v.y != 0.0) { vb = select(vb - 1u, vb + 1u, (v.x > 0.0) == (v.y > 0.0)); }
  return t.x + bitcast<f32>(vb);
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = gid.x / 8u;
  let l = gid.x % 8u;
  let c = gid.y;
  let live = r < p.rows && c < p.cols;
  var acc = 0.0;
  if (live) {
    let nb = p.nb;
    for (var i = 0u; i < nb; i++) {
      let d0 = Wd[r * nb + i];
      var ab = 0.0;
      for (var k = 0u; k < 4u; k++) {
        let b = i * 4u + k;
        let d1 = Ad[c * nb * 4u + b];
        let bits = (Wb[(r * nb + i) * 4u + k] >> (4u * l)) & 0xfu;
        let qw = Aq[(c * nb * 4u + b) * 8u + l];
        var s: i32 = 0;
        for (var j = 0u; j < 4u; j++) {
          let q = bitcast<i32>(qw << (24u - 8u * j)) >> 24u;
          // the reference negates as an i8 (wrapping): -(-128) stays -128
          let nq = select(-q, -128, q == -128);
          s += select(nq, q, ((bits >> j) & 1u) == 1u);
        }
        let prod = d1 * f32(s);
        ab = select(prod + ab, prod, k == 0u);
      }
      acc = fma_exact(d0, ab, acc);
    }
  }
  part[lid.x] = acc;
  workgroupBarrier();
  if (live && l == 0u) {
    let o = lid.x;
    let h = ((part[o] + part[o + 4u]) + (part[o + 2u] + part[o + 6u])) + ((part[o + 1u] + part[o + 5u]) + (part[o + 3u] + part[o + 7u]));
    O[c * p.rows + r] = h;
  }
}`;

/** The workgroup size for this GPU: 64 for AMD (wavefronts of 64), 32 for NVIDIA (warps of 32), else its subgroup
 *  size rounded up to a multiple of 8 (64 when unknown). */
export function workgroupFor(info = {}) {
  const v = String(info.vendor || '').toLowerCase();
  if (v.includes('nvidia')) return 32;
  if (v.includes('amd') || v.includes('ati')) return 64;
  const sg = Number(info.subgroupMaxSize) || 0;
  return sg >= 8 && sg <= 128 ? Math.ceil(sg / 8) * 8 : 64;
}

const f16f32 = (h) => {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
};
/** Q1_0 rows repacked for the fast kernel: each block's scale as f32 (exact) and its bits as four u32. */
export function packQ1(W, rows, n) {
  const nb = n / QK1_0, d = new Float32Array(rows * nb), b = new Uint32Array(rows * nb * 4);
  const dv = new DataView(W.buffer, W.byteOffset, W.byteLength);
  for (let i = 0; i < rows * nb; i++) {
    const o = i * Q1_0_BYTES;
    d[i] = f16f32(dv.getUint16(o, true));
    for (let k = 0; k < 4; k++) b[i * 4 + k] = dv.getUint32(o + 2 + 4 * k, true);
  }
  return { d, b, rows, n };
}
/** q8_0 activations (one or more, each n values) repacked: scales as f32, quants four to a u32. */
export function packQ8(As, n) {
  const nb32 = n / QK8_0, d = new Float32Array(As.length * nb32), q = new Uint32Array(As.length * nb32 * 8);
  As.forEach((A, c) => {
    const dv = new DataView(A.buffer, A.byteOffset, A.byteLength);
    for (let b = 0; b < nb32; b++) {
      d[c * nb32 + b] = f16f32(dv.getUint16(b * Q8_0_BYTES, true));
      for (let w = 0; w < 8; w++) q[(c * nb32 + b) * 8 + w] = dv.getUint32(b * Q8_0_BYTES + 2 + 4 * w, true);
    }
  });
  return { d, q, cols: As.length, n };
}

/** The fast kernel on `device`: a packed matrix times packed activations; O[col·rows + row] as f32 bits, and the time
 *  from submit to the result in hand. `wg` from `workgroupFor`. Buffers are made and freed here (for tests and
 *  timing); the engine keeps its weights resident instead. */
export async function gpuMatMul(device, Wp, Ap, wg = 64) {
  const U = globalThis.GPUBufferUsage, MAP = globalThis.GPUMapMode;
  const { rows, n } = Wp, cols = Ap.cols, nb = n / QK1_0;
  const buf = (data, usage) => { const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage }); device.queue.writeBuffer(b, 0, data); return b; };
  const bs = [buf(Wp.d, U.STORAGE | U.COPY_DST), buf(Wp.b, U.STORAGE | U.COPY_DST), buf(Ap.d, U.STORAGE | U.COPY_DST), buf(Ap.q, U.STORAGE | U.COPY_DST)];
  const oB = device.createBuffer({ size: rows * cols * 4, usage: U.STORAGE | U.COPY_SRC });
  const pB = buf(new Uint32Array([rows, nb, cols, 0]), U.UNIFORM | U.COPY_DST);
  const rd = device.createBuffer({ size: rows * cols * 4, usage: U.MAP_READ | U.COPY_DST });
  const module = device.createShaderModule({ code: Q1_MM_WGSL });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main', constants: { WG: wg } } });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [...bs, oB, pB].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  await device.queue.onSubmittedWorkDone();  // uploads done: the time below is the product's
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil((rows * 8) / wg), cols); pass.end();
  enc.copyBufferToBuffer(oB, 0, rd, 0, rows * cols * 4);
  const t = performance.now();
  device.queue.submit([enc.finish()]);
  await rd.mapAsync(MAP.READ);
  const ms = performance.now() - t;
  const out = new Uint32Array(rd.getMappedRange().slice(0)); rd.unmap();
  [...bs, oB, pB, rd].forEach((b) => b.destroy());
  return { bits: out, ms };
}

/** A small reproducible generator (mulberry32). */
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
/** A finite f16, sign random, exponent in a range that keeps every product and sum in f32's normal range. */
const f16bits = (r, lo = 3, hi = 17) => ((r() < 0.5 ? 0x8000 : 0) | ((lo + Math.floor(r() * (hi - lo))) << 10) | Math.floor(r() * 1024)) & 0xffff;

/** Random Q1_0 weights (rows × n) and q8_0 activations (n), with int8 values that include −128. */
export function randomCase(rows, n, seed = 1) {
  const r = rng(seed), nb = n / QK1_0;
  const W = new Uint8Array(rows * nb * Q1_0_BYTES), A = new Uint8Array((n / QK8_0) * Q8_0_BYTES);
  for (let b = 0; b < rows * nb; b++) {
    const o = b * Q1_0_BYTES, h = f16bits(r);
    W[o] = h & 255; W[o + 1] = h >> 8;
    for (let j = 2; j < Q1_0_BYTES; j++) W[o + j] = Math.floor(r() * 256);
  }
  for (let b = 0; b < n / QK8_0; b++) {
    const o = b * Q8_0_BYTES, h = f16bits(r);
    A[o] = h & 255; A[o + 1] = h >> 8;
    for (let j = 2; j < Q8_0_BYTES; j++) A[o + j] = r() < 0.02 ? 128 : Math.floor(r() * 256);  // −128 now and then
  }
  return { W, A, rows, n };
}

/** bankML's reference, from its WebAssembly module (browseml.wasm's browseml_q1_dot_ref), one row at a time. */
export async function reference(wasmUrl, wasiUrl, { W, A, rows, n }) {
  const { wasi } = await import(wasiUrl);
  const w = wasi({ files: {}, write: () => {} });
  const { instance } = await WebAssembly.instantiateStreaming(fetch(wasmUrl), { ...w.imports, browseml: { piece: () => 1, log: () => {} } });
  const x = instance.exports; w.bind(x.memory);
  const rowBytes = (n / QK1_0) * Q1_0_BYTES;
  const yp = x.browseml_alloc(A.length); new Uint8Array(x.memory.buffer, yp, A.length).set(A);
  const xp = x.browseml_alloc(rowBytes);
  const out = new Uint32Array(rows);
  for (let r = 0; r < rows; r++) {
    new Uint8Array(x.memory.buffer, xp, rowBytes).set(W.subarray(r * rowBytes, (r + 1) * rowBytes));
    out[r] = x.browseml_q1_dot_ref(n, xp, rowBytes, yp, A.length);
  }
  return out;
}

/** The kernel on this GPU (or a variant of it, for the controls): one row per invocation; each row's f32 bits. */
export async function gpuDot(device, { W, A, rows, n }, code = Q1_DOT_WGSL) {
  const U = globalThis.GPUBufferUsage, MAP = globalThis.GPUMapMode;
  const pad = (b) => { const p = new Uint8Array(Math.ceil(b.length / 4) * 4); p.set(b); return p; };
  const buf = (data, usage) => { const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage }); device.queue.writeBuffer(b, 0, data); return b; };
  const wB = buf(pad(W), U.STORAGE | U.COPY_DST), aB = buf(pad(A), U.STORAGE | U.COPY_DST);
  const oB = device.createBuffer({ size: rows * 4, usage: U.STORAGE | U.COPY_SRC });
  const pB = buf(new Uint32Array([n, rows, 0, 0]), U.UNIFORM | U.COPY_DST);
  const rd = device.createBuffer({ size: rows * 4, usage: U.MAP_READ | U.COPY_DST });
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [wB, aB, oB, pB].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(Math.ceil(rows / 64)); pass.end();
  enc.copyBufferToBuffer(oB, 0, rd, 0, rows * 4);
  const t = performance.now();
  device.queue.submit([enc.finish()]);
  await rd.mapAsync(MAP.READ);
  const ms = performance.now() - t;
  const out = new Uint32Array(rd.getMappedRange().slice(0)); rd.unmap();
  [wB, aB, oB, pB, rd].forEach((b) => b.destroy());
  return { bits: out, ms };
}

/**
 * EXACT: `rows` random rows of `n` weights on this GPU against bankML's reference, bit for bit. The answer to "may
 * this GPU do browseML's work": every row identical, or the rows that differ.
 * @returns {Promise<{rows, n, same, differ: {row, gpu, ref}[], gpuMs, refMs, adapter}>}
 */
export async function exact({ rows = 256, n = 2048, seed = 1, power = 'high-performance', wasmUrl = new URL('./reference/browseml.wasm', import.meta.url).href, wasiUrl = new URL('./reference/browseml-wasi.js', import.meta.url).href } = {}) {
  if (!navigator.gpu) throw new Error('this browser has no WebGPU');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: power });
  if (!adapter) throw new Error('no WebGPU adapter was offered');
  const device = await adapter.requestDevice();
  try {
    const c = randomCase(rows, n, seed);
    const t = performance.now();
    const ref = await reference(wasmUrl, wasiUrl, c);
    const refMs = performance.now() - t;
    const g = await gpuDot(device, c);
    const differ = [];
    for (let r = 0; r < rows; r++) if (g.bits[r] !== ref[r]) differ.push({ row: r, gpu: g.bits[r], ref: ref[r] });
    const info = adapter.info ?? {};
    return { rows, n, same: rows - differ.length, differ: differ.slice(0, 16), gpuMs: g.ms, refMs, adapter: [info.vendor, info.architecture, info.device].filter(Boolean).join(' · ') };
  } finally { device.destroy(); }
}
