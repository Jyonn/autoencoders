"""Encoder-backed sentence dataset for MultiNLI."""

from __future__ import annotations

import json

from .text import EncoderBackedTextDatasetConfig, TextEmbeddingExample, ZipBackedTextDataset


class MultiNLIDatasetConfig(EncoderBackedTextDatasetConfig):
    """Configuration for the MultiNLI sentence-embedding dataset."""

    model_type = "multinli_dataset"


class MultiNLIDataset(ZipBackedTextDataset):
    """Materialize sentence embeddings from the MultiNLI corpus."""

    dataset_name = "multinli"
    config_class = MultiNLIDatasetConfig
    config: MultiNLIDatasetConfig
    base_url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    required_members = (
        "multinli_1.0/multinli_1.0_train.jsonl",
        "multinli_1.0/multinli_1.0_dev_matched.jsonl",
        "multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
    )

    def load_examples(self) -> list[TextEmbeddingExample]:
        texts: list[str] = []
        for member_name in self.required_members:
            for line in self.read_archive_member_lines(member_name):
                if not line.strip():
                    continue
                record = json.loads(line)
                texts.append(record["sentence1"])
                texts.append(record["sentence2"])

        unique_texts = self.deduplicate_texts(texts)
        return [
            TextEmbeddingExample(sample_id=f"multinli-{index:07d}", text=text)
            for index, text in enumerate(unique_texts)
        ]
