# Getting Started

## Install

```bash
pip install autoencoders
pip install "autoencoders[torch]"
```

For docs and release tooling from source:

```bash
pip install "autoencoders[dev]"
```

## Build a basic AE

```python
import torch

from autoencoders import AutoencoderConfig, AutoencoderModel
from autoencoders.data.base import TensorSpec

model = AutoencoderModel(
    config=AutoencoderConfig(latent_dim=16),
    sample_spec=TensorSpec(shape=(50,)),
    encoder="mlp",
    encoder_config={"hidden_dims": [64, 32], "activation": "relu", "use_bias": True},
    decoder="mlp",
    decoder_config={"hidden_dims": [64, 50], "activation": "relu", "use_bias": True},
)

inputs = torch.randn(32, 50)
outputs = model(inputs)
```

## Inspect the pipeline

```python
for step in model.get_pipeline_trace():
    print(step.name, "->", step.output_spec)
```

## Train from YAML

```bash
python examples/trainer.py --config examples/configs/glove/ae.yaml --epoch 5
python examples/trainer.py --config examples/configs/cifar10/vqvae_vit.yaml --epoch 5
```
