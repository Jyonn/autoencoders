"""CLIP-backed dataset helpers for Flickr30k."""

from __future__ import annotations

import json
from pathlib import Path

from .clip import CLIPBackedDataset, CLIPRecord


class Flickr30kDataset(CLIPBackedDataset):
    """Materialize CLIP embeddings from Flickr30k image-caption pairs."""

    dataset_name = "flickr30k"
    hf_dataset_name = "AnyModal/flickr30k"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.manifest_path = self.raw_dir / "records.jsonl"
        self.images_dir = self.raw_dir / "images"

    def has_raw_data(self) -> bool:
        return self.manifest_path.exists() and self.images_dir.exists()

    def download(self, *, force: bool = False) -> None:
        try:
            from datasets import concatenate_datasets, load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for Flickr30k. Install it with "
                "`pip install autoencoders[clip]` or `pip install datasets`."
            ) from exc

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if force and self.manifest_path.exists():
            self.manifest_path.unlink()

        dataset_dict = load_dataset(self.hf_dataset_name)
        if hasattr(dataset_dict, "keys"):
            split_names = list(dataset_dict.keys())
            dataset = concatenate_datasets([dataset_dict[split_name] for split_name in split_names])
        else:
            dataset = dataset_dict

        with self.manifest_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(dataset):
                image = record["image"]
                filename = record.get("filename") or f"{record['img_id']}.jpg"
                image_path = self.images_dir / filename
                if force or not image_path.exists():
                    image.save(image_path)
                captions = self._extract_captions(record)
                payload = {
                    "image_id": str(record["img_id"]),
                    "filename": filename,
                    "captions": captions,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _extract_captions(record: dict) -> list[str]:
        for field_name in ("caption", "captions", "original_alt_text", "alt_text"):
            value = record.get(field_name)
            if value is None:
                continue
            if isinstance(value, str):
                normalized = value.strip()
                return [normalized] if normalized else []
            if isinstance(value, list):
                captions = [str(item).strip() for item in value if str(item).strip()]
                if captions:
                    return captions
        raise KeyError(
            "Flickr30k record did not contain a supported caption field. "
            "Expected one of: 'caption', 'captions', 'original_alt_text', 'alt_text'."
        )

    def load_records(self) -> list[CLIPRecord]:
        records: list[CLIPRecord] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                records.append(
                    CLIPRecord(
                        image_id=payload["image_id"],
                        image_path=self.images_dir / payload["filename"],
                        captions=list(payload["captions"]),
                    )
                )
        return records
