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
| f16 attention, RoPE, norms | the rest of a layer | next |
| the forward pass | a GPU worker beside the engine, matmuls handed over through shared memory | next ([TODO](https://github.com/browseML/browseML/blob/main/docs/TODO.md)) |

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
