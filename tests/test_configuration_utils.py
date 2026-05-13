"""Tests for configuration utilities that do not require torch."""

from __future__ import annotations

import tempfile
import unittest

from autoencoders import AutoencoderConfig


class AutoencoderConfigTest(unittest.TestCase):
    def test_round_trip_save_and_load(self) -> None:
        config = AutoencoderConfig(
            input_dim=32,
            latent_dim=8,
            hidden_dims=[16, 12],
            activation="gelu",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = AutoencoderConfig.from_pretrained(tmpdir)

        self.assertEqual(loaded.input_dim, 32)
        self.assertEqual(loaded.latent_dim, 8)
        self.assertEqual(loaded.hidden_dims, [16, 12])
        self.assertEqual(loaded.activation, "gelu")
        self.assertEqual(loaded.model_type, "autoencoder")

    def test_to_dict_contains_model_type(self) -> None:
        config = AutoencoderConfig(input_dim=16, latent_dim=4)
        payload = config.to_dict()

        self.assertEqual(payload["model_type"], "autoencoder")
        self.assertEqual(payload["input_dim"], 16)
        self.assertEqual(payload["latent_dim"], 4)


if __name__ == "__main__":
    unittest.main()
