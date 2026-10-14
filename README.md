<div align="center">

<img src="logo.png" width="140" alt="browseML">

### browseML on the GPU — exact WebGPU kernels

</div>

browseML runs bankML's engine on the visitor's CPU. This repository is how its work moves to the visitor's **GPU**,
through WebGPU: each kernel written in WGSL to give **the same bits** as bankML's scalar reference, and a test anyone can
run that proves it on their own GPU before a GPU is trusted with an answer.

<table>
<tr>
<td width="33%" valign="top">

**Exact, not close**

Each kernel reproduces the reference's arithmetic: its order of operations, its integer sums, its fused multiply-adds.

</td>
<td width="33%" valign="top">

**Any GPU**

WGSL's `fma()` may round twice. The kernels use bankML's exact FMA (TwoSum and round-to-odd), so a GPU that does
not fuse still gives the reference's bits.

</td>
<td width="33%" valign="top">

**Proved here**

`index.html` runs a kernel on random rows and compares every result's bits with bankML's own reference, plus
controls that must fail.

</td>
</tr>
</table>

## Kernels

| kernel | what | status |
|---|---|---|
| [`q1_0.js`](q1_0.js) `Q1_DOT_WGSL` | Q1_0 weights · q8_0 activations (bankML `q1_0::vec_dot_ref`), one row per invocation | **bit-exact**: 832/832 rows over three shapes (Brave, SwiftShader); controls 110/256 |
| [`q1_0.js`](q1_0.js) `Q1_MM_WGSL` | the fast kernel: eight lanes per row, a token dimension, the vendor's workgroup | **bit-exact** on AMD (Vega 3, Dawn); **bankML's own** as [`gpu/wgsl.rs`](https://github.com/cryptoAGI/bankml/blob/browseml/bankML/gpu/wgsl.rs) |
| [`q1_0.js`](q1_0.js) `Q1_MM4_WGSL` | the fast kernel with a token tile: four tokens per invocation, each weight word decoded once | **bit-exact** (the untiled kernel's bits, partial tiles included); **bankML's own** as `Q1_0_MM4`; reads prompts in browseML 0.6.10 |
| the forward pass | browseML's GPU worker serves bankML's GPU imports; the card computes a verified share of each 1-bit matrix | **in browseML 0.6.7** |
| f16 attention, RoPE, norms | the rest of a layer | next |

## Exact on AMD, and why it needed a fix

On a real AMD GPU the kernels were not exact at first: Mesa's shader compiler simplified TwoSum's `(a + b) − a` and lost
its error term. SPIR-V can mark operations exact (NoContraction; bankML's Vulkan kernels do), WGSL cannot, and Chrome's
compiler does not add it. Each intermediate now passes through an OR with a zero the compiler only learns at run time
(`hide`), so it cannot simplify across it. Checked on an AMD Radeon Vega 3 (RADV) through Dawn, Chrome's WebGPU, in Node
(`tests/gpu.mjs`): every result bit-exact, in every shape.

## The fast kernel

`Q1_MM_WGSL` is bankML's `q1_0_mat_vec8`: eight invocations per row, one per accumulation lane of the CPU kernel, the
lanes added in the CPU's order; a second dimension takes many activations (a prompt's tokens). The workgroup follows the
vendor (`workgroupFor`: 64 for AMD's wavefronts, 32 for NVIDIA's warps); 32, 64 and 128 give the same bits.

| on the Vega 3 | 1 token | 16 tokens | 64 tokens |
|---|---:|---:|---:|
| 2048×2048 | 2.3 ms (simple kernel: 15 ms) | 9.6 ms | 18.7 ms (14.4 GMAC/s) |
| 6144×2048 | 2.7 ms | 18.5 ms | 63.1 ms (12.8 GMAC/s) |

```sh
npm i webgpu   # Dawn for Node, in any directory
DAWN=<that directory> node tests/gpu.mjs
```

**In the engine (browseML 0.6.7, bankML `a32f81c`):** the fast kernel is now bankML's own (`gpu/wgsl.rs`, with
`workgroup_for`). The engine hands the page's GPU worker the shader's text and the workgroup size, and browseML's
test checks its copy is the same text. bankML's GPU worker runs on WebGPU with the same on-card oracle and
calibration as on Vulkan. It also uses the token dimension to read prompts (`begin_many`) and submits Q/K/V and
gate/up together with one wait (`begin_group`). Through browseML's bench on the Vega 3 (3 threads, alternating with
the GPU off), reading went 5.05 → 5.41 and 5.10 → 5.60 tok/s, writing was unchanged, and 12 of 12 answers were
identical to llama.cpp's record. Measured again on a quiet machine (browseML 0.6.8, load 2–5, no browser open):
reading 6.70–7.53 → 7.30–7.74 tok/s and writing 2.88–3.34 → 3.14–3.41 tok/s with the GPU on, 12 of 12 identical.

**Back to the server:** the same two lessons, a token dimension and several submissions in flight, are now in bankML's
Vulkan backend too (`q1_0_mat_mul8`, `Gpu::IN_FLIGHT`), bit-exact on the Vega 3.

**A token tile (browseML 0.6.10, bankML `8f5e930`):** `Q1_MM4_WGSL` is the fast kernel with four tokens per
invocation: each weight word is read and decoded once for all four, and every token's chain is unchanged, so the bits
are the untiled kernel's (checked at 1, 3, 5, 9 and 16 tokens). Past the last token an invocation computes a copy of it
and never stores it, so there is no branch inside the chain. On the Vega 3 (Dawn, interleaved medians over five rounds)
it was 1.14–1.18× the untiled kernel at 64–512 tokens and slower at 2, so bankML uses it only to read prompts of four
or more tokens. Tiles of 8 and 16 lost on short prompts; guarding the unused columns cost more than it saved.

**Changing a kernel:** change it in bankML first (`gpu/wgsl.rs`), then copy it to `q1_0.js` here and to browseML's
`web/browseml-gpu.js`. browseML's `testing/gpu.mjs` fails until the texts match.

## Measured

| where | shapes | result |
|---|---|---|
| `index.html`, headless Brave 152 (SwiftShader), 2026-10-13 | 256 × 2048 · 64 × 6144 · 512 × 128 | **256/256 · 64/64 · 512/512 bit-exact**; both controls differ, as they must |
| `index.html`, headless Brave (SwiftShader), 2026-10-14: the engine's kernels too, five tokens each | the same three shapes | fast **1280/1280 · 320/320 · 2560/2560**, tiled the same; simple 256/256 · 64/64 · 512/512; both controls differ |
| browseML's control panel, **EXACT** (TOOL SUITE) | 256 × 2048 | **256/256 bit-exact**, 258–461 ms |

**Used in browseML today:** the control panel's **EXACT** tool runs the kernel on the visitor's GPU and reports the
verdict; the **GPU** switch (off · low power · high performance) starts bankML with or without the card. With it on,
the card computes its verified share of every answer.

**What the GPU has to beat** (browseML v0.6.2, Ryzen 3 3200U, `browseml-mt-relaxed.wasm`, 4 threads): reading the
prompt at 5.6 tok/s and writing at 3.4 tok/s ([efficiency](https://github.com/browseML/browseML/blob/main/docs/efficiency.md)).
Reading the prompt is where a GPU would help most: many tokens at once reuse each weight row, which suits a GPU's batches.

## Run the test

```sh
git clone https://github.com/browseML/webgpu && cd webgpu
python3 -m http.server 8000      # then open http://localhost:8000 in a WebGPU browser and press Run
```

`reference/` holds bankML's reference as browseML builds it (`browseml.wasm` from
[browseML](https://github.com/browseML/browseML) `0ef80bb`, sha256 `9d658c7fea7cf7ac…`, exporting `browseml_q1_dot_ref`) and
browseML's WASI shim. The arithmetic and the exact-FMA method come from [bankML](https://github.com/cryptoAGI/bankml)
(`bankML/q1_0.rs`, `bankML/gpu/spirv.rs`); llama.cpp's own WebGPU backend (`ggml-webgpu`, MIT) is the reference for
the rest.

Licence: `MIT OR Apache-2.0`.
