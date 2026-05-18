# Training

For field-by-field trainer parameter meanings, see [Configuration Reference](configuration-reference.md#trainer-configs).

## Unified YAML entrypoint

All example runs go through:

```bash
python examples/trainer.py --config <path-to-config.yaml>
```

Each config is split into:

- `dataset`
- `model`
- `encoder`
- `decoder`
- `trainer`

`dataset`, `model`, `encoder`, and `decoder` use `name + config` structure. `trainer` is a flat config block.

## Runtime placeholders

Example configs support runtime overrides such as:

```yaml
max_vectors: ${max_vectors:5000}$
epochs: ${epoch:5}$
learning_rate: ${lr:0.001}$
```

So you can run:

```bash
python examples/trainer.py --config examples/configs/glove/ae.yaml --epoch 10 --lr 0.0005
```

## Trainer optimization options

`TrainingConfig` now supports:

- `optimizer_name`: `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`
- `weight_decay`
- `lr_scheduler_type`: `none`, `constant`, `linear`, `cosine`
- `warmup_epochs`
- `grad_clip_norm`

Example:

```yaml
trainer:
  output_dir: artifacts/glove/ae
  epochs: ${epoch:5}$
  batch_size: 256
  learning_rate: ${lr:0.001}$
  optimizer_name: adamw
  weight_decay: ${wd:0.01}$
  lr_scheduler_type: cosine
  warmup_epochs: ${warmup:1}$
  grad_clip_norm: ${clip:1.0}$
  device: auto
  seed: 42
```

## Family-specific trainers

- `AETrainer`
- `VAETrainer`
- `VQTrainer`
- `FactorVAETrainer`
- `AdversarialAutoencoderTrainer`

The unified entrypoint selects the correct trainer from `model.name`.
