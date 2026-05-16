"""Shared utilities for CLIP-backed multimodal embedding datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .base import (
    BaseDatasetConfig,
    CachedDataset,
    DatasetLoaders,
    DatasetSplits,
    ItemProgressBar,
    create_dataloaders,
    split_dataset,
)
from .embeddings import (
    EmbeddingMatrix,
    EmbeddingTensorDataset,
    load_embedding_artifact,
    save_embedding_artifact,
)


ClipModality = Literal["image", "text", "both"]


@dataclass
class CLIPRecord:
    """A multimodal record containing one image and one or more captions."""

    image_id: str
    image_path: Path
    captions: list[str]


class CLIPEmbeddingEncoder(ABC):
    """Minimal encoder contract for CLIP-style image and text embeddings."""

    model_name: str
    pretrained_name: str

    @abstractmethod
    def encode_images(self, image_paths: list[Path]) -> torch.Tensor:
        """Encode images into a dense float tensor of shape [N, D]."""

    @abstractmethod
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode raw texts into a dense float tensor of shape [N, D]."""


class OpenCLIPEmbeddingEncoder(CLIPEmbeddingEncoder):
    """OpenCLIP backend for CLIP-backed multimodal datasets."""

    def __init__(
        self,
        model_name: str,
        pretrained_name: str,
        *,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained_name = pretrained_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for CLIP-backed datasets. "
                "Install it with `pip install autoencoders[clip]` or "
                "`pip install open_clip_torch datasets pillow`."
            ) from exc
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for CLIP-backed datasets. "
                "Install it with `pip install autoencoders[clip]` or `pip install pillow`."
            ) from exc

        self._open_clip = open_clip
        self._image_lib = Image
        self.device = device or self._infer_device()
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_name,
            device=self.device,
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

    @staticmethod
    def _infer_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    def encode_images(self, image_paths: list[Path]) -> torch.Tensor:
        batches: list[torch.Tensor] = []
        progress = ItemProgressBar("Encoding Flickr30k images", len(image_paths))
        with torch.no_grad():
            try:
                for start in range(0, len(image_paths), self.batch_size):
                    batch_paths = image_paths[start : start + self.batch_size]
                    images = [self._preprocess(self._image_lib.open(path).convert("RGB")) for path in batch_paths]
                    pixel_values = torch.stack(images).to(self.device)
                    embeddings = self._model.encode_image(pixel_values)
                    if self.normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                    batches.append(embeddings.detach().to(dtype=torch.float32, device="cpu"))
                    progress.update(len(batch_paths))
            except RuntimeError as exc:
                if self.device.startswith("cuda") and (
                    "CUDNN_STATUS_NOT_INITIALIZED" in str(exc) or "CUDA" in str(exc)
                ):
                    raise RuntimeError(
                        "CLIP image encoding failed on CUDA. Try a smaller "
                        "`--encoder-batch-size` or run preprocessing with `--clip-device cpu`."
                    ) from exc
                raise
            finally:
                progress.close()
        return torch.cat(batches, dim=0)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        batches: list[torch.Tensor] = []
        progress = ItemProgressBar("Encoding Flickr30k captions", len(texts))
        with torch.no_grad():
            try:
                for start in range(0, len(texts), self.batch_size):
                    batch_texts = texts[start : start + self.batch_size]
                    tokenized = self._tokenizer(batch_texts).to(self.device)
                    embeddings = self._model.encode_text(tokenized)
                    if self.normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                    batches.append(embeddings.detach().to(dtype=torch.float32, device="cpu"))
                    progress.update(len(batch_texts))
            except RuntimeError as exc:
                if self.device.startswith("cuda") and (
                    "CUDNN_STATUS_NOT_INITIALIZED" in str(exc) or "CUDA" in str(exc)
                ):
                    raise RuntimeError(
                        "CLIP text encoding failed on CUDA. Try a smaller "
                        "`--encoder-batch-size` or run preprocessing with `--clip-device cpu`."
                    ) from exc
                raise
            finally:
                progress.close()
        return torch.cat(batches, dim=0)


class CLIPBackedDatasetConfig(BaseDatasetConfig):
    """Configuration shared by CLIP-backed datasets."""

    model_type = "clip_dataset"

    def __init__(
        self,
        *,
        encoder: str | None = None,
        clip_pretrained: str | None = None,
        encoder_batch_size: int = 64,
        clip_device: str | None = None,
        normalize_embeddings: bool = True,
        clip_modality: ClipModality = "both",
        max_vectors: int | None = None,
        **kwargs,
    ) -> None:
        if "encoder_name" in kwargs and encoder is None:
            encoder = kwargs.pop("encoder_name")
        if "encoder_pretrained" in kwargs and clip_pretrained is None:
            clip_pretrained = kwargs.pop("encoder_pretrained")
        if "encoder_device" in kwargs and clip_device is None:
            clip_device = kwargs.pop("encoder_device")
        if "modality" in kwargs and clip_modality == "both":
            clip_modality = kwargs.pop("modality")
        self.encoder = encoder
        self.clip_pretrained = clip_pretrained
        self.encoder_batch_size = encoder_batch_size
        self.clip_device = clip_device
        self.normalize_embeddings = normalize_embeddings
        self.clip_modality = clip_modality
        super().__init__(max_vectors=max_vectors, **kwargs)


class CLIPBackedDataset(CachedDataset, ABC):
    """Base class for datasets that materialize image/text embeddings with CLIP."""

    encoder_family = "open_clip"
    default_encoder_name = "ViT-B-32"
    default_pretrained_name = "laion2b_s34b_b79k"
    config_class = CLIPBackedDatasetConfig
    config: CLIPBackedDatasetConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.config.encoder is None:
            self.config.encoder = self.default_encoder_name
        if self.config.clip_pretrained is None:
            self.config.clip_pretrained = self.default_pretrained_name

    @property
    def encoder_name(self) -> str:
        return str(self.config.encoder)

    @property
    def encoder_pretrained(self) -> str:
        return str(self.config.clip_pretrained)

    @property
    def encoder_batch_size(self) -> int:
        return int(self.config.encoder_batch_size)

    @property
    def encoder_device(self) -> str | None:
        return self.config.clip_device

    @property
    def normalize_embeddings(self) -> bool:
        return bool(self.config.normalize_embeddings)

    @property
    def modality(self) -> ClipModality:
        return self.config.clip_modality

    @property
    def artifact_name(self) -> str:
        suffix = "full" if self.max_vectors is None else f"top-{self.max_vectors}"
        encoder_slug = self.encoder_name.replace("/", "--").replace(":", "--")
        pretrained_slug = self.encoder_pretrained.replace("/", "--").replace(":", "--")
        return f"{self.dataset_name}-{self.modality}-{encoder_slug}-{pretrained_slug}-{suffix}"

    @property
    def artifact_dir(self) -> Path:
        return self.processed_dir / self.artifact_name

    def is_prepared(self) -> bool:
        required_files = (
            self.artifact_dir / "embeddings.pt",
            self.artifact_dir / "tokens.txt",
            self.artifact_dir / "metadata.json",
        )
        return all(path.exists() for path in required_files)

    def load_embedding_matrix(self, *, download: bool = True) -> EmbeddingMatrix:
        self.ensure_prepared(download=download)
        return load_embedding_artifact(self.artifact_dir)

    def as_dataset(self, *, download: bool = True) -> EmbeddingTensorDataset:
        return EmbeddingTensorDataset(self.load_embedding_matrix(download=download))

    def get_splits(
        self,
        *,
        download: bool = True,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetSplits:
        dataset = self.as_dataset(download=download)
        return split_dataset(
            dataset,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    def get_dataloaders(
        self,
        *,
        download: bool = True,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> DatasetLoaders:
        splits = self.get_splits(
            download=download,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        return create_dataloaders(
            splits,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def build_encoder(self) -> CLIPEmbeddingEncoder:
        return OpenCLIPEmbeddingEncoder(
            self.encoder_name,
            self.encoder_pretrained,
            batch_size=self.encoder_batch_size,
            normalize_embeddings=self.normalize_embeddings,
            device=self.encoder_device,
        )

    @abstractmethod
    def load_records(self) -> list[CLIPRecord]:
        """Load raw image-caption records from the dataset cache."""

    def prepare(self) -> None:
        records = self.load_records()
        if not records:
            raise ValueError(f"No CLIP records were loaded for dataset {self.dataset_name!r}.")

        encoder = self.build_encoder()
        tokens: list[str] = []
        embeddings: list[torch.Tensor] = []

        if self.modality in {"image", "both"}:
            image_paths = [record.image_path for record in records]
            image_embeddings = encoder.encode_images(image_paths)
            image_tokens = [f"image:{record.image_id}" for record in records]
            tokens.extend(image_tokens)
            embeddings.append(image_embeddings)

        if self.modality in {"text", "both"}:
            text_tokens: list[str] = []
            text_values: list[str] = []
            for record in records:
                for caption_index, caption in enumerate(record.captions):
                    text_tokens.append(f"text:{record.image_id}:{caption_index}")
                    text_values.append(caption)
            text_embeddings = encoder.encode_texts(text_values)
            tokens.extend(text_tokens)
            embeddings.append(text_embeddings)

        matrix = torch.cat(embeddings, dim=0)
        if self.max_vectors is not None:
            tokens = tokens[: self.max_vectors]
            matrix = matrix[: self.max_vectors]

        token_to_index = {token: index for index, token in enumerate(tokens)}
        embedding_matrix = EmbeddingMatrix(
            tokens=tokens,
            matrix=matrix,
            token_to_index=token_to_index,
            source_path=str(self.raw_dir),
            name=self.artifact_name,
            metadata={
                "dataset_name": self.dataset_name,
                "encoder_family": self.encoder_family,
                "encoder_name": encoder.model_name,
                "encoder_pretrained": encoder.pretrained_name,
                "modality": self.modality,
            },
        )
        save_embedding_artifact(embedding_matrix, self.artifact_dir)
