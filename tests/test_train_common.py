"""Tests for the unified YAML trainer entrypoint."""

from __future__ import annotations

import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from autoencoders.data import TensorSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

TRAINER_PATH = EXAMPLES_DIR / "trainer.py"
TRAINER_SPEC = importlib.util.spec_from_file_location("trainer_entry", TRAINER_PATH)
trainer_entry = importlib.util.module_from_spec(TRAINER_SPEC)
assert TRAINER_SPEC.loader is not None
TRAINER_SPEC.loader.exec_module(trainer_entry)

TRAIN_AE_PATH = EXAMPLES_DIR / "train_ae.py"
TRAIN_AE_SPEC = importlib.util.spec_from_file_location("train_ae", TRAIN_AE_PATH)
train_ae = importlib.util.module_from_spec(TRAIN_AE_SPEC)
assert TRAIN_AE_SPEC.loader is not None
TRAIN_AE_SPEC.loader.exec_module(train_ae)


class ConfigNode:
    def __init__(self, **values) -> None:
        self._values = dict(values)
        for key, value in values.items():
            setattr(self, key, value)

    def __call__(self) -> dict:
        return dict(self._values)

    def __bool__(self) -> bool:
        return bool(self._values)


class NamedConfigNode:
    def __init__(self, name=None, config: dict | None = None) -> None:
        self.name = name
        self._config = None if config is None else dict(config)

    @property
    def config(self):
        if self._config is None:
            return None
        return lambda: dict(self._config)

    def __bool__(self) -> bool:
        return self.name is not None or self._config is not None


class TrainerEntrypointTest(unittest.TestCase):
    def build_configurations(
        self,
        *,
        model_name: str = "ae",
        model_config: dict | None = None,
        encoder_name: str = "mlp",
        encoder_config: dict | None = None,
        decoder_name: str | None = "mlp",
        decoder_config: dict | None = None,
        trainer_config: dict | None = None,
    ):
        return ConfigNode(
            dataset=NamedConfigNode(
                name="glove",
                config={"dim": 50, "max_vectors": 128},
            ),
            model=NamedConfigNode(
                name=model_name,
                config=model_config or {"latent_dim": 8, "reconstruction_loss": "mse"},
            ),
            encoder=NamedConfigNode(
                name=encoder_name,
                config=encoder_config or {"hidden_dims": [32, 16], "activation": "relu", "use_bias": True},
            ),
            decoder=(
                None
                if decoder_name is None
                else NamedConfigNode(
                    name=decoder_name,
                    config=decoder_config or {"hidden_dims": [16, 50], "activation": "relu", "use_bias": True},
                )
            ),
            trainer=ConfigNode(
                **(
                    trainer_config
                    or {
                        "output_dir": "artifacts/test",
                        "epochs": 1,
                        "batch_size": 32,
                        "learning_rate": 1e-3,
                        "device": "cpu",
                        "validation_ratio": 0.2,
                        "test_ratio": 0.1,
                        "seed": 7,
                        "show_only_best_epochs": False,
                        "advice": True,
                    }
                )
            ),
        )

    def test_train_ae_wrapper_delegates_to_unified_entrypoint(self) -> None:
        self.assertTrue(callable(train_ae.main))
        self.assertEqual(train_ae.main.__name__, "main")

    def test_build_effective_config_keeps_name_and_config_structure(self) -> None:
        configurations = self.build_configurations()

        effective = trainer_entry._build_effective_config(configurations)

        self.assertEqual(effective["dataset"]["name"], "glove")
        self.assertEqual(effective["dataset"]["config"]["dim"], 50)
        self.assertEqual(effective["model"]["name"], "ae")
        self.assertEqual(effective["encoder"]["name"], "mlp")
        self.assertEqual(effective["decoder"]["name"], "mlp")
        self.assertEqual(effective["trainer"]["batch_size"], 32)

    def test_build_model_supports_explicit_decoder_from_yaml_flow(self) -> None:
        configurations = self.build_configurations(
            model_name="ae",
            model_config={"latent_dim": 8, "reconstruction_loss": "mse"},
            encoder_config={"hidden_dims": [32, 16], "activation": "relu", "use_bias": True},
            decoder_config={"hidden_dims": [16, 50], "activation": "relu", "use_bias": True},
        )

        model = trainer_entry.build_model(configurations, TensorSpec(shape=(50,)))
        outputs = model(inputs=torch.randn(2, 50))

        self.assertEqual(tuple(outputs.reconstruction.shape), (2, 50))
        self.assertEqual(tuple(outputs.latents.shape), (2, 8))

    def test_build_model_supports_auto_decoder_from_builtin_encoder(self) -> None:
        configurations = self.build_configurations(
            model_name="ae",
            model_config={"latent_dim": 8, "reconstruction_loss": "mse"},
            encoder_config={"hidden_dims": [16, 8], "activation": "relu", "use_bias": True},
            decoder_name=None,
            decoder_config=None,
        )

        model = trainer_entry.build_model(configurations, TensorSpec(shape=(50,)))
        outputs = model(inputs=torch.randn(2, 50))

        self.assertEqual(tuple(outputs.reconstruction.shape), (2, 50))
        self.assertIsNotNone(model.decoder)

    def test_select_trainer_components_maps_model_families(self) -> None:
        ae_trainer, ae_config = trainer_entry.select_trainer_components("ae")
        vae_trainer, vae_config = trainer_entry.select_trainer_components("vae")
        vq_trainer, vq_config = trainer_entry.select_trainer_components("vqvae")
        aae_trainer, aae_config = trainer_entry.select_trainer_components("aae")
        factor_trainer, factor_config = trainer_entry.select_trainer_components("factorvae")

        self.assertEqual(ae_trainer.__name__, "AETrainer")
        self.assertEqual(ae_config.__name__, "TrainingConfig")
        self.assertEqual(vae_trainer.__name__, "VAETrainer")
        self.assertEqual(vae_config.__name__, "TrainingConfig")
        self.assertEqual(vq_trainer.__name__, "VQTrainer")
        self.assertEqual(vq_config.__name__, "TrainingConfig")
        self.assertEqual(aae_trainer.__name__, "AdversarialAutoencoderTrainer")
        self.assertEqual(aae_config.__name__, "AdversarialAutoencoderTrainingConfig")
        self.assertEqual(factor_trainer.__name__, "FactorVAETrainer")
        self.assertEqual(factor_config.__name__, "FactorVariationalAutoencoderTrainingConfig")

    def test_print_effective_config_renders_named_sections(self) -> None:
        configurations = self.build_configurations()

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            trainer_entry._print_effective_config(configurations)
        output = buffer.getvalue()

        self.assertIn("Effective Config", output)
        self.assertIn("dataset", output)
        self.assertIn("model", output)
        self.assertIn("encoder", output)
        self.assertIn("decoder", output)
        self.assertIn("trainer", output)

    def test_print_pipeline_trace_uses_new_compact_output(self) -> None:
        configurations = self.build_configurations(
            model_name="ae",
            model_config={"latent_dim": 8, "reconstruction_loss": "mse"},
            encoder_config={"hidden_dims": [32, 16], "activation": "relu", "use_bias": True},
            decoder_config={"hidden_dims": [16, 50], "activation": "relu", "use_bias": True},
        )
        model = trainer_entry.build_model(configurations, TensorSpec(shape=(50,)))

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            trainer_entry._print_pipeline_trace(model)
        output = buffer.getvalue()

        self.assertIn("Shape Trace", output)
        self.assertIn("sample", output)
        self.assertIn("encoder[mlp]", output)
        self.assertIn("encoder_to_core_projection", output)
        self.assertIn("core", output)
        self.assertIn("core_to_decoder_projection", output)
        self.assertIn("decoder[mlp]", output)
        self.assertIn("linear(", output)


if __name__ == "__main__":
    unittest.main()
