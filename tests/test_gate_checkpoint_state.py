from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import diagnostics_nextbert  # noqa: E402
import evaluate_nextbert  # noqa: E402
from train_nextbert import (  # noqa: E402
    _initialize_backbone_from_checkpoint,
    _set_adaptation_update,
    _save_weights_only,
    restore_gate_state,
    train_supervised_step,
)


class GateToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_logits = nn.Parameter(torch.tensor([0.2]))
        self.adaptation_updates = 0
        self._gate_ramp_override: float | None = None
        self.special_token_ids = (0, 1, 2)

    @property
    def gate_ramp(self) -> float:
        if self._gate_ramp_override is not None:
            return self._gate_ramp_override
        return min(1.0, self.adaptation_updates / 64.0)

    def set_adaptation_step(self, update: int) -> None:
        self.adaptation_updates = max(0, int(update))
        self._gate_ramp_override = None

    def set_gate_ramp(self, ramp: float) -> None:
        self._gate_ramp_override = float(ramp)

    def effective_gate_values(self) -> dict[str, torch.Tensor]:
        return {"toy_gate": self.gate_logits.sigmoid() * self.gate_ramp}

    def forward_classification(self, input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               content_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        del attention_mask
        lengths = content_mask.sum(dim=1, keepdim=True).clamp_min(1)
        feature = (input_ids.float() * content_mask.float()).sum(dim=1, keepdim=True) / lengths
        alpha = self.effective_gate_values()["toy_gate"]
        return {"logits": torch.cat((feature * alpha, -feature * alpha,
                                      torch.zeros_like(feature)), dim=1)}


class GateCheckpointStateTests(unittest.TestCase):
    def test_saved_downstream_ramp_restores_for_evaluation_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "v3_qa.pt"
            source = GateToy()
            source.set_gate_ramp(1.0)
            _save_weights_only(checkpoint, model=source, step=688, task="qa",
                               variant="V3", seed=17)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["metadata"]["gate_ramp"], 1.0)
            self.assertEqual(payload["metadata"]["adaptation_updates"], 0)

            evaluated = GateToy()
            evaluate_nextbert._load_checkpoint(evaluated, checkpoint, torch.device("cpu"))
            self.assertEqual(evaluated.gate_ramp, 1.0)
            self.assertGreater(float(evaluated.effective_gate_values()["toy_gate"]), 0.5)

            model_dir = root / "model"
            model_dir.mkdir()
            diagnosed = GateToy()
            with patch.object(diagnostics_nextbert, "NextBert", return_value=diagnosed):
                diagnostics_nextbert._load_model(
                    model_dir, "V3", tokenizer=None, device=torch.device("cpu"),
                    checkpoint=checkpoint,
                )
            self.assertEqual(diagnosed.gate_ramp, 1.0)
            self.assertGreater(float(diagnosed.effective_gate_values()["toy_gate"]), 0.5)

    def test_legacy_mlm_ramp_is_inferred_but_legacy_downstream_stays_disabled(self) -> None:
        state = GateToy().state_dict()
        mlm_model = GateToy()
        mlm_model.load_state_dict(state)
        restored_mlm = restore_gate_state(
            mlm_model, {"task": "mlm", "variant": "V3", "step": 32, "model": state}
        )
        self.assertEqual(restored_mlm["adaptation_updates"], 32)
        self.assertEqual(mlm_model.gate_ramp, 0.5)
        self.assertGreater(float(mlm_model.effective_gate_values()["toy_gate"].detach()), 0.0)
        # Resuming at absolute optimizer step 32 must keep the original global
        # warmup schedule: the next update uses update counter 33, not a restart.
        _set_adaptation_update(mlm_model, update=32, ramp_updates=64)
        self.assertEqual(mlm_model.adaptation_updates, 33)
        self.assertEqual(mlm_model.gate_ramp, 33 / 64)

        downstream = GateToy()
        downstream.set_gate_ramp(1.0)
        restored_downstream = restore_gate_state(
            downstream, {"task": "qa", "variant": "V3", "step": 688, "model": state}
        )
        self.assertEqual(restored_downstream["gate_ramp"], 0.0)
        self.assertEqual(downstream.gate_ramp, 0.0)
        self.assertEqual(float(downstream.effective_gate_values()["toy_gate"].detach()), 0.0)

    def test_downstream_initialization_activates_gates_and_gate_receives_gradient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v3_mlm.pt"
            source = GateToy()
            with torch.no_grad():
                source.gate_logits.fill_(0.4)
            torch.save({"task": "mlm", "variant": "V3", "step": 512,
                        "model": source.state_dict()}, checkpoint)

            model = GateToy()
            _initialize_backbone_from_checkpoint(
                model, checkpoint, torch.device("cpu"), variant="V3",
            )

        self.assertEqual(model.gate_ramp, 1.0)
        self.assertGreater(float(model.effective_gate_values()["toy_gate"].detach()), 0.5)
        optimizer = AdamW(model.parameters(), lr=1e-2)
        batch = {"input_ids": [[2, 10, 1], [2, 20, 1]], "labels": [0, 0]}
        train_supervised_step(
            model, optimizer, "classification", [batch],
            device=torch.device("cpu"), step=0,
        )
        self.assertIsNotNone(model.gate_logits.grad)
        self.assertGreater(float(model.gate_logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
