"""Encoder-backed sentence dataset for Stanford Natural Language Inference."""

from __future__ import annotations

import json

from .text import EncoderBackedTextDatasetConfig, TextEmbeddingExample, ZipBackedTextDataset


class SNLIDatasetConfig(EncoderBackedTextDatasetConfig):
    """Configuration for the SNLI sentence-embedding dataset."""

    model_type = "snli_dataset"


class SNLIDataset(ZipBackedTextDataset):
    """Materialize sentence embeddings from the SNLI corpus."""

    dataset_name = "snli"
    config_class = SNLIDatasetConfig
    base_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    required_members = (
        "snli_1.0/snli_1.0_train.jsonl",
        "snli_1.0/snli_1.0_dev.jsonl",
        "snli_1.0/snli_1.0_test.jsonl",
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
            TextEmbeddingExample(sample_id=f"snli-{index:07d}", text=text)
            for index, text in enumerate(unique_texts)
        ]
