"""CLIP-backed dataset helpers for Flickr30k."""

from __future__ import annotations

import json
from pathlib import Path

from .clip import CLIPBackedDataset, CLIPRecord


class Flickr30kDataset(CLIPBackedDataset):
    """Materialize CLIP embeddings from Flickr30k image-caption pairs."""

    dataset_name = "flickr30k"
    hf_dataset_names = ("AnyModal/flickr30k", "cjc/flickr30k", "nlphuji/flickr30k")

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

        dataset_dict = self._load_hf_dataset(load_dataset)
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
                payload = {
                    "image_id": str(record["img_id"]),
                    "filename": filename,
                    "captions": list(record["caption"]),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _load_hf_dataset(self, load_dataset):
        last_error: Exception | None = None
        for dataset_name in self.hf_dataset_names:
            try:
                return load_dataset(dataset_name)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            "Unable to download Flickr30k from any configured Hugging Face dataset source: "
            + ", ".join(self.hf_dataset_names)
        ) from last_error

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
