# Backbones

## Canonical dataset format

Datasets should expose one canonical raw sample format.

- embedding datasets usually return `TensorSpec(shape=(D,))`
- image datasets return `TensorSpec(shape=(H, W, C))`

Backbones adapt from that canonical form internally.

## MLP

`MLPModule` is intended for vector-like tensors whose last dimension is the feature width.

```text
TensorSpec(shape=(50,))
TensorSpec(shape=(None, 50))
TensorSpec(shape=(T, 50))
```

Key config fields:

- `hidden_dims`
- `activation`
- `use_bias`

## CNN

`CNNModule` consumes image-like `TensorSpec(shape=(H, W, C))` values and internally converts them to `NCHW`.

Key config fields:

- `channels`
- `kernel_sizes`
- `strides`
- `paddings`
- `activation`
- `use_bias`
- `transpose`

Use `transpose: true` for explicit upsampling decoders built from transposed convolutions.

## Vision Transformer

`VisionTransformerModule` also consumes `TensorSpec(shape=(H, W, C))` but patchifies the image internally and exposes sequence-shaped latent specs.

Typical config fields:

- `patch_size`
- `hidden_dim`
- `depth`
- `num_heads`
- `mlp_ratio`
- `dropout`

## Auto-inferred decoders

`decoder: null` is supported only when the model can reverse the encoder without changing the runtime decoder input spec.

This is usually fine for:

- basic `AE`
- some shape-preserving quantized models

It is intentionally rejected for models whose decoder space differs from encoder output space, such as hierarchical or latent-shape-changing families.
