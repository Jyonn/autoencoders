# Configuration Reference

This page explains the meaning of every configuration parameter exposed through the unified YAML entrypoint.

```yaml
dataset:
  name: ...
  config: ...
model:
  name: ...
  config: ...
encoder:
  name: ...
  config: ...
decoder:
  name: ...
  config: ...
trainer:
  ...
```

## Reading This Page

- "Base" sections define fields inherited by many concrete configs.
- Concrete model sections only list fields added on top of the base family.
- Backbone fields belong to `encoder.config` and `decoder.config`.
- Trainer fields belong to the flat `trainer` block.

## Dataset Configs

### BaseDatasetConfig

| Field | Meaning |
| --- | --- |
| `max_vectors` | Optional cap on how many prepared samples are materialized from a downloadable embedding-style dataset. Use it to shorten experiments or smoke tests. |

### GloVeDatasetConfig

| Field | Meaning |
| --- | --- |
| `dim` | Embedding width to extract from the Stanford GloVe archive. Valid values are `50`, `100`, `200`, and `300`. |
| `max_vectors` | Optional cap on how many word vectors to keep from the chosen file. |

### FastTextEnglishDatasetConfig

| Field | Meaning |
| --- | --- |
| `max_vectors` | Optional cap on how many English fastText vectors are loaded from the source file. |

### ConceptNetNumberbatchDatasetConfig

| Field | Meaning |
| --- | --- |
| `max_vectors` | Optional cap on how many Numberbatch vectors are loaded. |

### EncoderBackedTextDatasetConfig

Used by `snli` and `multinli`.

| Field | Meaning |
| --- | --- |
| `encoder` | Text encoder model name used to materialize sentence embeddings, typically a Sentence-Transformers identifier. |
| `encoder_batch_size` | Batch size used while converting raw texts into embeddings during preprocessing. |
| `normalize_embeddings` | Whether to L2-normalize encoder outputs before saving them as dataset samples. |
| `max_vectors` | Optional cap on how many raw text examples are embedded. |

### CLIPBackedDatasetConfig

Used by `flickr30k`.

| Field | Meaning |
| --- | --- |
| `encoder` | CLIP backbone name, such as `ViT-B-32`. |
| `clip_pretrained` | CLIP checkpoint tag paired with the backbone, such as `laion2b_s34b_b79k`. |
| `encoder_batch_size` | Batch size used while extracting CLIP embeddings. |
| `clip_device` | Device override for CLIP preprocessing, such as `cpu`, `cuda`, or `mps`. |
| `normalize_embeddings` | Whether image/text embeddings are normalized to unit length before saving. |
| `clip_modality` | Which modality to materialize: `image`, `text`, or `both`. |
| `max_vectors` | Optional cap on how many records or caption embeddings are materialized. |

### CIFAR10DatasetConfig

| Field | Meaning |
| --- | --- |
| `max_examples` | Optional cap on how many CIFAR-10 images are retained after download and preprocessing. |

## Backbone Configs

### BaseAutoencoderModuleConfig

This is the structural base for built-in backbones. It does not currently add standalone YAML fields beyond what concrete modules define.

### MLPModuleConfig

| Field | Meaning |
| --- | --- |
| `hidden_dims` | Ordered list of layer widths. For an encoder, this is the path from input features to the module output. For an explicit decoder, this is the path from decoder input features back to sample space. |
| `activation` | Nonlinearity inserted after each non-final linear layer. Supported values: `relu`, `gelu`, `silu`, `tanh`. |
| `use_bias` | Whether each linear layer uses a bias term. |
| `dropout` | Dropout probability applied after non-final activations. |
| `norm` | Optional normalization after each non-final linear layer: `none`, `layernorm`, or `batchnorm`. |
| `weight_init` | Initialization strategy for linear weights: `default`, `xavier_uniform`, or `xavier_normal`. |

### CNNModuleConfig

| Field | Meaning |
| --- | --- |
| `channels` | Output channel count for each convolutional stage. |
| `kernel_sizes` | Kernel size per stage. Each value may be one integer or one `(height, width)` pair. |
| `strides` | Stride per stage. Each value may be one integer or one `(height, width)` pair. |
| `paddings` | Padding per stage. Each value may be one integer or one `(height, width)` pair. |
| `activation` | Nonlinearity after each non-final convolution. Supported values: `relu`, `gelu`, `silu`, `tanh`. |
| `use_bias` | Whether convolution layers use bias terms. |
| `transpose` | If `true`, build explicit upsampling layers with `ConvTranspose2d`. Use this for image decoders declared explicitly in YAML. |

### VisionTransformerModuleConfig

| Field | Meaning |
| --- | --- |
| `patch_size` | Patch height and width used to turn images into patch tokens, given as one integer or one `(height, width)` pair. |
| `hidden_dim` | Transformer token width after patch projection. |
| `num_layers` | Number of transformer encoder layers. |
| `num_heads` | Attention heads per transformer layer. `hidden_dim` must be divisible by this value. |
| `mlp_ratio` | Feed-forward expansion ratio inside each transformer block. |
| `dropout` | Dropout probability used inside transformer layers. |
| `use_bias` | Whether patch projection, output projection, and transformer linear layers use bias terms. |

## Model Configs

### BaseAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `latent_dim` | Width of the core latent space when the family uses a single latent width. In deterministic AEs this is the latent width after `project_to_core`; in quantized models it is usually the codebook embedding width. |
| `reconstruction_loss` | Reconstruction objective. Current built-in choices are intended for dense tensors and typically use `mse`. |

### Deterministic AE Family

#### AutoencoderConfig

No extra fields beyond `BaseAutoencoderConfig`.

#### DenoisingAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `noise_type` | Corruption mode applied to inputs before reconstruction. |
| `noise_std` | Standard deviation for additive Gaussian noise when that corruption mode is used. |
| `masking_ratio` | Fraction of features dropped or masked when a masking corruption mode is used. |

#### ContractiveAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `contractive_weight` | Strength of the Jacobian contraction penalty added to the reconstruction objective. |

#### SparseAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `sparsity_weight` | Penalty weight encouraging sparse latent activations. |
| `target_activation` | Desired mean activation level used by the sparsity penalty. |

#### TopKSparseAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `topk` | Number of latent units retained per sample when using top-k sparsification. |

#### KLSparseAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `sparsity_weight` | Weight for the KL sparsity term. |
| `target_activation` | Target average activation used inside the KL sparsity penalty. |

#### WassersteinAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `mmd_weight` | Strength of the MMD regularizer matching latent codes to the chosen prior. |
| `kernel_bandwidths` | Kernel bandwidth list used by the MMD estimator. |

#### AdversarialAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `adversarial_weight` | Strength of the adversarial latent-matching objective. |
| `discriminator_hidden_dims` | Hidden widths for the latent discriminator network. |

### Variational Family

#### BaseVariationalAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `kl_weight` | Multiplier on the KL term after warmup is complete. |
| `free_bits` | Minimum KL contribution retained per latent dimension or block to reduce posterior collapse. |
| `kl_warmup_epochs` | Number of epochs over which the KL weight ramps from `kl_start_weight` to `kl_weight`. |
| `kl_start_weight` | Initial KL multiplier before warmup progresses. |
| `use_mean_in_eval` | If `true`, evaluation and export use posterior means instead of sampling noise. |

#### VariationalAutoencoderConfig

No extra fields beyond the base variational family.

#### BetaVariationalAutoencoderConfig

No new fields. Use `kl_weight` to express beta-style scaling.

#### DenoisingVariationalAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `noise_type` | Corruption mode for the denoising encoder input. |
| `noise_std` | Standard deviation for additive Gaussian corruption. |
| `masking_ratio` | Fraction of masked features when using masking noise. |

#### HierarchicalVariationalAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `top_latent_dim` | Width of the upper latent level in hierarchical VAE models. |

#### VampPriorVariationalAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `num_pseudo_inputs` | Number of learned pseudo-inputs used to define the VampPrior mixture. |

#### InformationVariationalAutoencoderConfig / MMDVariationalAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `mmd_weight` | Weight for the additional MMD regularizer. |
| `kernel_bandwidths` | Bandwidth list for the MMD kernel mixture. |

#### DIPVariationalAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `dip_type` | Which DIP-VAE covariance penalty variant to use. |
| `lambda_diag` | Weight on diagonal covariance matching. |
| `lambda_offdiag` | Weight on off-diagonal covariance suppression. |

#### BetaTCVariationalAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `tc_weight` | Weight on the total-correlation penalty. |

#### FactorVariationalAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `tc_weight` | Weight on the total-correlation penalty estimated through the discriminator. |
| `discriminator_hidden_dims` | Hidden widths for the auxiliary discriminator. |

### Quantized Family

#### BaseVectorQuantizedAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `codebook_size` | Number of discrete codes per learned codebook. |
| `commitment_weight` | Weight that pulls encoder outputs toward selected codes. |
| `codebook_weight` | Weight that pulls learned code vectors toward encoder outputs when explicit codebook loss is used. |
| `assignment_strategy` | How discrete indices are selected from codebook distances: `nearest` or `sinkhorn`. |
| `sinkhorn_epsilon` | Entropic regularization strength for Sinkhorn assignment. It may be one float shared across codebooks or one list with one value per codebook slot. A slot set to `0.0` falls back to nearest-neighbor assignment. |
| `sinkhorn_iters` | Number of Sinkhorn normalization iterations when `assignment_strategy` is `sinkhorn`. |
| `kmeans_init` | Whether learned codebooks are initialized from the first training batch of encoder latents instead of uniform random weights. |
| `kmeans_iters` | Number of Lloyd iterations used during codebook k-means initialization. |
| `use_ema_codebook` | Whether learned codebooks are updated by exponential moving averages instead of gradient updates. |
| `ema_decay` | EMA decay factor for codebook statistics. Lower values adapt faster; higher values are smoother. |
| `ema_epsilon` | Numerical stabilizer used when normalizing EMA cluster sizes. |
| `dead_code_reset` | Whether rarely used codes are reinitialized at the end of training epochs or steps. |
| `dead_code_threshold` | Usage-count threshold below which a code is considered dead for reset purposes. |

#### VectorQuantizedAutoencoderConfig

No extra fields beyond the base quantized family.

#### GumbelQuantizedAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `temperature` | Initial Gumbel-softmax temperature. |
| `min_temperature` | Lower bound for the annealed temperature. |
| `anneal_rate` | Multiplicative decay controlling how fast the temperature cools. |

#### FiniteScalarQuantizedAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `num_levels` | Number of scalar quantization levels per latent feature. |

#### ResidualFiniteScalarQuantizedAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `num_levels` | Number of scalar levels per residual quantizer. |
| `num_quantizers` | Number of residual scalar quantization stages. |

#### ProductQuantizedAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `num_codebooks` | Number of product-quantization subspaces. `latent_dim` must be divisible by this value. |

#### ResidualQuantizedAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `num_quantizers` | Number of residual vector quantization stages applied in sequence. |

#### HierarchicalVectorQuantizedAutoencoderConfig

| Field | Meaning |
| --- | --- |
| `top_latent_dim` | Width of the top-level codebook latents before they are combined with bottom-level latents. |

## Trainer Configs

### TrainingConfig

| Field | Meaning |
| --- | --- |
| `output_dir` | Directory where checkpoints, exported models, and metrics are written. |
| `epochs` | Maximum training epochs. Use `0` together with `patience` for early-stop-only training. |
| `patience` | Early stopping patience in epochs without validation improvement. |
| `learning_rate` | Base optimizer learning rate. |
| `optimizer_name` | Optimizer choice: `adam`, `adamw`, `sgd`, `rmsprop`, or `adagrad`. |
| `weight_decay` | Weight decay passed to the optimizer. |
| `lr_scheduler_type` | Learning-rate schedule: `none`, `constant`, `linear`, or `cosine`. |
| `warmup_epochs` | Number of epochs used for learning-rate warmup before the main scheduler takes over. |
| `grad_clip_norm` | If set, clip gradient norm to this value after backpropagation. |
| `batch_size` | Batch size used to build train/validation/test dataloaders. |
| `full_dataset_as_splits` | If `true`, reuse the full dataset for train, validation, and test instead of splitting it. |
| `device` | Device target such as `auto`, `cpu`, `cuda`, or `mps`. |
| `seed` | Global random seed for reproducibility. |
| `save_best_by` | Validation metric short names used to save best checkpoints. `loss` writes to `best/`; extra names such as `commit` write to `best-commit/`. |
| `show_only_best_epochs` | If `true`, only emit compact summaries for best-validation epochs instead of every epoch. |
| `advice` | If `true`, append trainer-generated tuning suggestions after the run. |

### AdversarialAutoencoderTrainingConfig

| Field | Meaning |
| --- | --- |
| `discriminator_learning_rate` | Optional learning rate override for the discriminator optimizer. |
| `generator_learning_rate` | Optional learning rate override for the generator or autoencoder optimizer. |
| `discriminator_steps` | Number of discriminator updates per generator-style update. |

### FactorVariationalAutoencoderTrainingConfig

| Field | Meaning |
| --- | --- |
| `discriminator_learning_rate` | Optional learning rate override for the FactorVAE discriminator. |
| `discriminator_steps` | Number of discriminator updates per training iteration. |
