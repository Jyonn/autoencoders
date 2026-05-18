# Model Parameter Trees

This section organizes model configuration parameters by inheritance tree rather than by one flat reference table.

Use it to answer two questions quickly:

1. which parameters come from the base family config?
2. which parameters are unique to a specific model?

## Tree layout

- [Deterministic AE family](deterministic.md)
- [Variational family](variational.md)
- [Quantized family](quantized.md)

## Reading the trees

- indented items are inherited from the parent config
- leaf bullets under a concrete model are the extra fields that model adds
- backbone parameters such as `hidden_dims`, `channels`, or `patch_size` live in module configs, not model configs

For a parameter-by-parameter explanation of every field, use the dedicated
[Configuration Reference](../configuration-reference.md).
