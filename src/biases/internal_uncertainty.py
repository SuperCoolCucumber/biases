from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from biases.paths import configure_artifact_environment, output_path
from biases.utils import write_jsonl


configure_artifact_environment()


@dataclass(frozen=True)
class InternalScore:
    record_id: str
    pair_id: str | None
    variant_id: str | None
    scores: dict[str, float]
    metadata: dict[str, Any]


class MahalanobisDistanceSeq:
    def __init__(self, *, regularization: float = 1e-4) -> None:
        self.regularization = regularization
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, hidden_states: np.ndarray) -> "MahalanobisDistanceSeq":
        x = _as_2d(hidden_states)
        self.mean_ = x.mean(axis=0)
        covariance = np.cov(x, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.array([[float(covariance)]])
        covariance = covariance + np.eye(covariance.shape[0]) * self.regularization
        self.precision_ = np.linalg.pinv(covariance)
        return self

    def score(self, hidden_states: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise ValueError("MahalanobisDistanceSeq must be fit before scoring")
        x = _as_2d(hidden_states)
        delta = x - self.mean_
        return np.sqrt(np.einsum("ij,jk,ik->i", delta, self.precision_, delta))


class RelativeMahalanobisDistanceSeq:
    def __init__(self, *, regularization: float = 1e-4) -> None:
        self.foreground = MahalanobisDistanceSeq(regularization=regularization)
        self.background = MahalanobisDistanceSeq(regularization=regularization)

    def fit(self, foreground_states: np.ndarray, background_states: np.ndarray) -> "RelativeMahalanobisDistanceSeq":
        self.foreground.fit(foreground_states)
        self.background.fit(background_states)
        return self

    def score(self, hidden_states: np.ndarray) -> np.ndarray:
        return self.foreground.score(hidden_states) - self.background.score(hidden_states)


class LinearProbeScorer:
    """Small logistic-regression probe trained on calibration hidden states."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        epochs: int = 500,
        l2: float = 1e-3,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, hidden_states: np.ndarray, labels: Iterable[bool | int]) -> "LinearProbeScorer":
        x = _as_2d(hidden_states)
        y = np.asarray([1.0 if label else 0.0 for label in labels], dtype=np.float64)
        if len(x) != len(y):
            raise ValueError("hidden_states and labels must have the same length")
        if len(np.unique(y)) < 2:
            raise ValueError("linear probe requires both positive and negative labels")

        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        x_norm = (x - self.mean_) / self.scale_
        self.weights_ = np.zeros(x_norm.shape[1], dtype=np.float64)
        self.bias_ = 0.0

        for _ in range(self.epochs):
            logits = x_norm @ self.weights_ + self.bias_
            probs = _sigmoid(logits)
            error = probs - y
            grad_w = x_norm.T @ error / len(x_norm) + self.l2 * self.weights_
            grad_b = float(error.mean())
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def score(self, hidden_states: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("LinearProbeScorer must be fit before scoring")
        x = (_as_2d(hidden_states) - self.mean_) / self.scale_
        return _sigmoid(x @ self.weights_ + self.bias_)


def _as_2d(hidden_states: np.ndarray) -> np.ndarray:
    x = np.asarray(hidden_states, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("hidden_states must be a 1D or 2D array")
    return x


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def score_internal_states(
    *,
    record_rows: list[dict[str, Any]],
    hidden_states: np.ndarray,
    calibration_mask: Iterable[bool],
    event_labels: Iterable[bool | int],
    background_hidden_states: np.ndarray | None = None,
) -> list[InternalScore]:
    x = _as_2d(hidden_states)
    if len(record_rows) != len(x):
        raise ValueError("record_rows and hidden_states must have the same length")
    calibration = np.asarray([bool(value) for value in calibration_mask])
    labels = np.asarray([bool(value) for value in event_labels])
    if len(calibration) != len(x) or len(labels) != len(x):
        raise ValueError("calibration_mask and event_labels must match hidden_states length")

    md = MahalanobisDistanceSeq().fit(x[calibration])
    md_scores = md.score(x)

    rmd_scores: np.ndarray | None = None
    if background_hidden_states is not None:
        background = _as_2d(background_hidden_states)
        rmd = RelativeMahalanobisDistanceSeq().fit(x[calibration], background)
        rmd_scores = rmd.score(x)

    probe_scores: np.ndarray | None = None
    if len(np.unique(labels[calibration])) == 2:
        probe = LinearProbeScorer().fit(x[calibration], labels[calibration])
        probe_scores = probe.score(x)

    results: list[InternalScore] = []
    for index, row in enumerate(record_rows):
        scores = {"mahalanobis_distance": float(md_scores[index])}
        if rmd_scores is not None:
            scores["relative_mahalanobis_distance"] = float(rmd_scores[index])
        if probe_scores is not None:
            scores["linear_probe_probability"] = float(probe_scores[index])
        results.append(
            InternalScore(
                record_id=str(row.get("record_id")),
                pair_id=row.get("pair_id"),
                variant_id=row.get("variant_id"),
                scores=scores,
                metadata={
                    "routing_split": row.get("routing_split"),
                    "event_label": bool(labels[index]),
                },
            )
        )
    return results


def write_internal_scores(
    scores: list[InternalScore],
    path: Path = output_path("internal_uncertainty", "internal_scores.jsonl"),
) -> None:
    rows = [
        {
            "record_id": score.record_id,
            "pair_id": score.pair_id,
            "variant_id": score.variant_id,
            **score.scores,
            **score.metadata,
        }
        for score in scores
    ]
    write_jsonl(path, rows)


def collect_decision_hidden_states_hf(
    *,
    model_name: str,
    prompt_texts: list[str],
    device: str = "cuda",
    dtype: str = "auto",
) -> np.ndarray:
    """Collect last-prompt-position hidden states with a Hugging Face forward pass.

    This is intentionally separate from vLLM decoding. It does not generate; it
    only runs the prompt through a white-box HF model and returns the final layer
    hidden state at the last prompt token.
    """

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("collect_decision_hidden_states_hf requires torch and transformers") from exc

    torch_dtype = "auto"
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype in {"bfloat16", "bf16"}:
        torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
    ).to(device)
    model.eval()

    states: list[np.ndarray] = []
    with torch.no_grad():
        for prompt in prompt_texts:
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            output = model(**encoded, output_hidden_states=True)
            state = output.hidden_states[-1][0, -1].detach().float().cpu().numpy()
            states.append(state)
    return np.stack(states, axis=0)


def load_hidden_state_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line)["hidden_state"])
    return np.asarray(rows, dtype=np.float64)
