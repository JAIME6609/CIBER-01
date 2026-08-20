#!/usr/bin/env python3
"""
Topology-Guided Multimodal Deep Learning for Cybersecurity Operations
=====================================================================

This script creates a reproducible synthetic Security Operations Center (SOC)
benchmark and evaluates six detection strategies:

1. Conventional logistic-regression baseline.
2. Temporal Transformer.
3. Graph neural network.
4. Temporal-graph fusion.
5. Topology-guided fusion without topological consistency regularization.
6. Topology-guided fusion with topological consistency regularization.

The experiment is organized to match the manuscript's Results and Analysis
subsections. It writes publication-ready tables and figures into:

    <output_dir>/5.1   Predictive performance and ablation analysis
    <output_dir>/5.2   Robustness, drift, and calibration analysis
    <output_dir>/5.3   Operational and explainability analysis

Mathematical implementation notes
---------------------------------
The topological branch does not approximate graph shape with ad hoc scalar
statistics. For every weighted entity-interaction graph, the program builds a
Vietoris-Rips (clique) filtration and performs boundary-matrix reduction over
the field F_2. It extracts zero- and one-dimensional persistence intervals and
vectorizes them as persistence images. The implementation is explicit so that
the filtration, boundary operator, reduction rule, and vectorization remain
auditable and reproducible without requiring proprietary software.

Recommended installation
------------------------
Use Python 3.10, 3.11, 3.12, or 3.13 in a virtual environment, then run:

    python -m pip install numpy pandas scipy scikit-learn matplotlib \
        networkx torch

The program checks these dependencies before starting. Missing packages can be
installed automatically with the optional flag ``--install-missing``.

Example
-------

    python topological_soc_experiment.py \
        --output-dir topological_soc_results \
        --epochs 10 \
        --seeds 13 29

For a faster diagnostic run:

    python topological_soc_experiment.py --quick

The default experiment is CPU-compatible and fixes all random seeds used for
simulation, optimization, bootstrapping, and visualization.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_PACKAGES: Mapping[str, str] = {
    "numpy": "numpy>=2.0",
    "pandas": "pandas>=2.0",
    "scipy": "scipy>=1.11",
    "sklearn": "scikit-learn>=1.4",
    "matplotlib": "matplotlib>=3.8",
    "networkx": "networkx>=3.2",
    "torch": "torch>=2.2",
}


def ensure_dependencies(install_missing: bool = False) -> None:
    """Validate required libraries and optionally install missing packages."""
    missing = [pip_name for module, pip_name in REQUIRED_PACKAGES.items()
               if importlib.util.find_spec(module) is None]
    if not missing:
        return

    command = [sys.executable, "-m", "pip", "install", *missing]
    message = (
        "The following required packages are missing:\n  - "
        + "\n  - ".join(missing)
        + "\nInstall them with:\n  "
        + " ".join(command)
    )
    if not install_missing:
        raise SystemExit(message + "\nAlternatively, rerun with --install-missing.")

    print(message)
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Automatic dependency installation failed. Run the displayed "
            "command manually in the intended Python environment."
        ) from exc


# Parse the installation flag before importing third-party packages.
_install_requested = "--install-missing" in sys.argv
ensure_dependencies(install_missing=_install_requested)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2))))


# ---------------------------------------------------------------------------
# Configuration and reproducibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """All simulation, learning, and evaluation parameters."""

    random_seed: int = 20260805
    n_train: int = 640
    n_validation: int = 224
    n_test: int = 320
    attack_fraction: float = 0.18
    sequence_length: int = 16
    sequence_features: int = 10
    graph_nodes: int = 12
    node_features: int = 8
    attack_stages: int = 4
    persistence_grid: int = 4
    persistence_sigma: float = 0.11
    hidden_dim: int = 24
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
    topology_consistency_weight: float = 0.025
    bootstrap_repetitions: int = 350
    window_minutes: float = 5.0
    windows_per_shift: int = 96
    analyst_minutes_per_alert: float = 3.0


MODEL_LABELS: Mapping[str, str] = {
    "logistic": "Logistic baseline",
    "temporal": "Temporal Transformer",
    "graph": "Graph neural network",
    "temporal_graph": "Temporal-graph fusion",
    "full_no_consistency": "Topology fusion without consistency",
    "full": "Topology-guided fusion",
}

STRESS_LABELS: Mapping[str, str] = {
    "standard": "In-distribution",
    "drift": "Concept drift",
    "missing": "20% telemetry loss",
    "timestamp": "Timestamp perturbation",
    "adversarial": "Attribute perturbation",
}

ATTACK_NAMES: Mapping[int, str] = {
    0: "Credential and lateral movement",
    1: "Phishing and exfiltration",
    2: "Privilege and cloud escalation",
    3: "Ransomware preparation",
}

SEQUENCE_FEATURE_NAMES = [
    "failed_authentication",
    "successful_authentication",
    "network_volume",
    "dns_entropy",
    "process_creation",
    "file_operation",
    "cloud_admin_action",
    "email_risk",
    "alert_count",
    "sensor_confidence",
]

NODE_NAMES = [
    "User-A",
    "User-B",
    "User-C",
    "User-D",
    "Host-A",
    "Host-B",
    "Host-C",
    "Host-D",
    "Server-A",
    "Server-B",
    "Cloud",
    "External",
]


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Persistent homology for weighted clique filtrations
# ---------------------------------------------------------------------------


def _simplex_boundary(simplex: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Return codimension-one faces of a simplex."""
    if len(simplex) <= 1:
        return []
    return [simplex[:i] + simplex[i + 1 :] for i in range(len(simplex))]


def persistent_intervals_from_adjacency(
    adjacency: np.ndarray,
    maximum_dimension: int = 1,
) -> dict[int, np.ndarray]:
    """
    Compute H_0 and H_1 persistence intervals for a weighted clique complex.

    The edge filtration value is ``1 - w_ij`` after clipping the symmetric
    interaction weight ``w_ij`` to [0, 1]. A triangle enters at the largest
    filtration value of its three edges. Boundary-matrix reduction is carried
    out over F_2 with sparse Python sets.

    Parameters
    ----------
    adjacency:
        Symmetric weighted adjacency matrix.
    maximum_dimension:
        Highest homology dimension to return. The current experiment uses 1.

    Returns
    -------
    dict
        Keys 0 and 1 map to arrays with columns ``birth`` and ``death``.
    """
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    n_nodes = matrix.shape[0]
    matrix = np.clip((matrix + matrix.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(matrix, 0.0)

    simplices: list[tuple[float, int, tuple[int, ...]]] = []
    for vertex in range(n_nodes):
        simplices.append((0.0, 0, (vertex,)))

    edge_filtration: dict[tuple[int, int], float] = {}
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            value = float(1.0 - matrix[i, j])
            edge = (i, j)
            edge_filtration[edge] = value
            simplices.append((value, 1, edge))

    # Triangles are necessary to determine deaths of one-dimensional cycles.
    if maximum_dimension >= 1:
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                for k in range(j + 1, n_nodes):
                    value = max(
                        edge_filtration[(i, j)],
                        edge_filtration[(i, k)],
                        edge_filtration[(j, k)],
                    )
                    simplices.append((float(value), 2, (i, j, k)))

    # Dimension breaks filtration ties so every face precedes its coface.
    simplices.sort(key=lambda item: (item[0], item[1], item[2]))
    index_by_simplex = {simplex: idx for idx, (_, _, simplex) in enumerate(simplices)}

    reduced_by_pivot: dict[int, set[int]] = {}
    birth_columns: set[int] = set()
    paired_births: dict[int, int] = {}

    for column_index, (_, dimension, simplex) in enumerate(simplices):
        if dimension == 0:
            column: set[int] = set()
        else:
            column = {index_by_simplex[face] for face in _simplex_boundary(simplex)}

        while column:
            pivot = max(column)
            previous = reduced_by_pivot.get(pivot)
            if previous is None:
                break
            column.symmetric_difference_update(previous)

        if column:
            pivot = max(column)
            reduced_by_pivot[pivot] = set(column)
            paired_births[pivot] = column_index
        else:
            birth_columns.add(column_index)

    intervals: dict[int, list[tuple[float, float]]] = {
        dim: [] for dim in range(maximum_dimension + 1)
    }
    terminal_value = 1.0

    for birth_index, death_index in paired_births.items():
        birth_value, birth_dimension, _ = simplices[birth_index]
        death_value, _, _ = simplices[death_index]
        if birth_dimension <= maximum_dimension and death_value > birth_value + 1e-12:
            intervals[birth_dimension].append((float(birth_value), float(death_value)))

    for birth_index in birth_columns.difference(paired_births.keys()):
        birth_value, birth_dimension, _ = simplices[birth_index]
        if birth_dimension <= maximum_dimension and terminal_value > birth_value + 1e-12:
            intervals[birth_dimension].append((float(birth_value), terminal_value))

    output: dict[int, np.ndarray] = {}
    for dimension, values in intervals.items():
        if values:
            output[dimension] = np.asarray(sorted(values), dtype=np.float32)
        else:
            output[dimension] = np.empty((0, 2), dtype=np.float32)
    return output


def persistence_image(
    intervals: np.ndarray,
    grid_size: int,
    sigma: float,
) -> np.ndarray:
    """Vectorize a persistence diagram on a birth-persistence grid."""
    if intervals.size == 0:
        return np.zeros(grid_size * grid_size, dtype=np.float32)

    births = np.clip(intervals[:, 0], 0.0, 1.0)
    persistence = np.clip(intervals[:, 1] - intervals[:, 0], 0.0, 1.0)
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    image = np.zeros((grid_size, grid_size), dtype=np.float64)
    denominator = max(2.0 * sigma * sigma, 1e-10)

    for birth, life in zip(births, persistence):
        if life <= 1e-10:
            continue
        squared_distance = (
            (grid[:, None] - float(birth)) ** 2
            + (grid[None, :] - float(life)) ** 2
        )
        image += float(life) * np.exp(-squared_distance / denominator)

    norm = np.linalg.norm(image)
    if norm > 0:
        image /= norm
    return image.astype(np.float32).ravel()


def topological_vector(
    adjacency: np.ndarray,
    grid_size: int,
    sigma: float,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Create persistence-image and summary features for H_0 and H_1."""
    diagrams = persistent_intervals_from_adjacency(adjacency, maximum_dimension=1)
    features: list[np.ndarray] = []
    summaries: list[float] = []
    for dimension in (0, 1):
        diagram = diagrams[dimension]
        features.append(persistence_image(diagram, grid_size, sigma))
        if diagram.size:
            lifetimes = diagram[:, 1] - diagram[:, 0]
            summaries.extend(
                [
                    float(len(lifetimes)),
                    float(np.sum(lifetimes)),
                    float(np.max(lifetimes)),
                    float(np.mean(lifetimes)),
                ]
            )
        else:
            summaries.extend([0.0, 0.0, 0.0, 0.0])
    vector = np.concatenate([*features, np.asarray(summaries, dtype=np.float32)])
    return vector.astype(np.float32), diagrams


# ---------------------------------------------------------------------------
# Synthetic SOC generator
# ---------------------------------------------------------------------------


@dataclass
class SyntheticSOCDataset:
    """Container for all modalities and operational labels."""

    sequence: np.ndarray
    node_features: np.ndarray
    adjacency: np.ndarray
    topology: np.ndarray
    labels: np.ndarray
    incident_ids: np.ndarray
    stages: np.ndarray
    attack_types: np.ndarray
    sample_ids: np.ndarray

    def copy(self) -> "SyntheticSOCDataset":
        return SyntheticSOCDataset(
            sequence=self.sequence.copy(),
            node_features=self.node_features.copy(),
            adjacency=self.adjacency.copy(),
            topology=self.topology.copy(),
            labels=self.labels.copy(),
            incident_ids=self.incident_ids.copy(),
            stages=self.stages.copy(),
            attack_types=self.attack_types.copy(),
            sample_ids=self.sample_ids.copy(),
        )


ATTACK_PATHS: Mapping[int, list[tuple[int, int]]] = {
    0: [(11, 0), (0, 4), (4, 8), (8, 11), (8, 5)],
    1: [(11, 1), (1, 5), (5, 10), (10, 11), (5, 9)],
    2: [(2, 6), (6, 8), (8, 10), (10, 2), (6, 9)],
    3: [(11, 7), (7, 9), (9, 4), (4, 11), (9, 10)],
}

ATTACK_SEQUENCE_SIGNATURES: Mapping[int, list[int]] = {
    0: [0, 1, 2, 8],
    1: [7, 3, 2, 5],
    2: [6, 1, 4, 8],
    3: [4, 5, 2, 8],
}


def _baseline_adjacency(rng: np.random.Generator, condition: str) -> np.ndarray:
    """Generate a low-risk organizational interaction graph."""
    n = len(NODE_NAMES)
    adjacency = rng.beta(1.7, 7.5, size=(n, n)) * 0.58
    adjacency = np.triu(adjacency, 1)
    adjacency += adjacency.T

    # Legitimate user-host, host-server, and cloud relationships.
    legitimate_edges = [
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 8), (5, 8), (6, 9), (7, 9),
        (8, 10), (9, 10),
    ]
    for i, j in legitimate_edges:
        adjacency[i, j] = adjacency[j, i] = rng.uniform(0.34, 0.58)

    # Benign maintenance can generate locally intense chains. These hard
    # negatives overlap with attack edge weights but generally lack a stable
    # closed loop across the filtration.
    if rng.random() < 0.34:
        nodes = rng.choice(n, size=4, replace=False)
        for i, j in zip(nodes[:-1], nodes[1:]):
            adjacency[i, j] = adjacency[j, i] = rng.uniform(0.48, 0.70)
    if rng.random() < 0.11:
        triangle = rng.choice(n, size=3, replace=False)
        for i, j in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[0], triangle[2])):
            adjacency[i, j] = adjacency[j, i] = rng.uniform(0.43, 0.57)

    if condition == "drift":
        # Organizational drift changes ordinary partnerships while preserving
        # the broad topology of benign operation.
        for i, j in [(0, 10), (3, 8), (2, 9)]:
            adjacency[i, j] = adjacency[j, i] = rng.uniform(0.38, 0.59)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency.astype(np.float32)


def _baseline_sequence(
    rng: np.random.Generator,
    config: ExperimentConfig,
    condition: str,
) -> np.ndarray:
    """Generate role- and time-dependent benign telemetry."""
    t = np.arange(config.sequence_length, dtype=np.float32)
    sequence = rng.normal(0.0, 0.23, size=(config.sequence_length, config.sequence_features))
    daily_wave = 0.16 * np.sin(2.0 * np.pi * t / config.sequence_length)
    sequence[:, 1] += 0.45 + daily_wave
    sequence[:, 2] += 0.34 + 0.08 * np.cos(2.0 * np.pi * t / config.sequence_length)
    sequence[:, 4] += 0.24
    sequence[:, 5] += 0.21
    sequence[:, 9] += 0.82

    # Rare but legitimate maintenance and administrative behavior.
    if rng.random() < 0.34:
        location = int(rng.integers(2, config.sequence_length - 2))
        sequence[location : location + 2, 4:7] += rng.uniform(0.32, 0.72)
    if rng.random() < 0.24:
        sequence[:, 0] += rng.uniform(0.18, 0.48)
    if rng.random() < 0.18:
        feature = int(rng.integers(2, 9))
        start = int(rng.integers(0, config.sequence_length - 2))
        sequence[start : start + 2, feature] += rng.uniform(0.30, 0.65)

    if condition == "drift":
        # New remote-work and cloud patterns modify surface statistics.
        sequence[:, 2] *= 1.18
        sequence[:, 6] += 0.22
        sequence[:, 1] -= 0.07
        sequence[:, 9] -= 0.06
    return sequence.astype(np.float32)


def _derive_node_features(
    rng: np.random.Generator,
    sequence: np.ndarray,
    adjacency: np.ndarray,
    label: int,
    stage: int,
) -> np.ndarray:
    """Create heterogeneous node attributes from roles and observed activity."""
    n = adjacency.shape[0]
    features = rng.normal(0.0, 0.18, size=(n, 8))
    degree = adjacency.sum(axis=1) / max(1, n - 1)
    max_edge = adjacency.max(axis=1)
    role_privilege = np.asarray(
        [0.20, 0.18, 0.32, 0.16, 0.36, 0.34, 0.28, 0.26, 0.72, 0.68, 0.84, 0.05],
        dtype=np.float32,
    )
    features[:, 0] += 0.72 * degree
    features[:, 1] += role_privilege
    features[:, 2] += 0.72 * max_edge
    features[:, 3] += float(np.mean(sequence[:, 0]))
    features[:, 4] += float(np.mean(sequence[:, 4]))
    features[:, 5] += float(np.mean(sequence[:, 5]))
    features[:, 6] += float(np.mean(sequence[:, 6]))
    features[:, 7] += float(np.mean(sequence[:, 9]))
    # The label and stage are deliberately not encoded directly. Every node
    # feature must arise from observed telemetry rather than hidden ground truth.
    return features.astype(np.float32)


def _inject_attack(
    rng: np.random.Generator,
    sequence: np.ndarray,
    adjacency: np.ndarray,
    attack_type: int,
    stage: int,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a causal multistage campaign into temporal and graph telemetry."""
    sequence = sequence.copy()
    adjacency = adjacency.copy()
    path = ATTACK_PATHS[attack_type]
    signature = ATTACK_SEQUENCE_SIGNATURES[attack_type]
    if condition == "drift":
        # Preserve the attack's topological structure while relocating it to
        # previously uncommon entities within each organizational role.
        role_mapping = {**dict(zip(range(4), rng.permutation(np.arange(4)))),
                        **dict(zip(range(4, 8), rng.permutation(np.arange(4, 8)))),
                        **dict(zip(range(8, 10), rng.permutation(np.arange(8, 10)))),
                        10: 10, 11: 11}
        path = [(int(role_mapping[i]), int(role_mapping[j])) for i, j in path]

    # A stage reveals progressively more of a coordinated path. The closing
    # edge creates a persistent loop in later stages.
    active_edge_count = min(len(path), stage + 2)
    for edge_index, (i, j) in enumerate(path[:active_edge_count]):
        base_weight = 0.54 + 0.045 * stage + 0.018 * edge_index
        adjacency[i, j] = adjacency[j, i] = np.clip(
            base_weight + rng.normal(0.0, 0.045), 0.0, 0.90
        )

    onset = 2 + stage * 3
    onset = min(onset, sequence.shape[0] - 3)
    amplitude = 0.48 + 0.14 * stage
    if condition == "drift":
        onset = max(0, sequence.shape[0] - onset - 3)
        amplitude *= 0.54
        signature = signature[1:] + signature[:1]
    elif condition == "adversarial":
        amplitude *= 0.34

    for offset, feature_index in enumerate(signature):
        start = min(onset + offset // 2, sequence.shape[0] - 2)
        width = 2 if stage < 3 else 3
        end = min(sequence.shape[0], start + width)
        sequence[start:end, feature_index] += amplitude * (0.78 + 0.12 * offset)

    # Confidence degradation and delayed duplicate alerts are realistic
    # stressors and prevent reliance on a single deterministic feature.
    sequence[onset:, 9] -= 0.04 + 0.02 * stage
    if stage >= 2:
        sequence[min(onset + 1, sequence.shape[0] - 1) :, 8] += 0.18
    if condition == "adversarial":
        nonessential = [1, 2, 4, 5, 6, 7]
        sequence[:, nonessential] += rng.normal(0.0, 0.20, size=(sequence.shape[0], len(nonessential)))
        # Preserve the relational attack structure while perturbing surface
        # attributes, which directly tests structural generalization.
        permutation = rng.permutation(sequence.shape[1])
        sequence = 0.88 * sequence + 0.12 * sequence[:, permutation]

    return sequence.astype(np.float32), adjacency.astype(np.float32)


def generate_synthetic_soc(
    n_samples: int,
    config: ExperimentConfig,
    seed: int,
    condition: str = "standard",
    sample_id_offset: int = 0,
) -> SyntheticSOCDataset:
    """Generate a synthetic SOC split with grouped, multistage incidents."""
    if condition not in {"standard", "drift", "adversarial"}:
        raise ValueError(f"Unsupported generator condition: {condition}")
    if not 0.0 < config.attack_fraction < 1.0:
        raise ValueError("attack_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    n_attack = int(round(n_samples * config.attack_fraction))
    n_attack -= n_attack % config.attack_stages
    n_benign = n_samples - n_attack
    n_incidents = n_attack // config.attack_stages

    records: list[dict[str, Any]] = []
    incident_counter = 0
    for _ in range(n_incidents):
        attack_type = int(rng.integers(0, len(ATTACK_NAMES)))
        for stage in range(config.attack_stages):
            records.append(
                {
                    "label": 1,
                    "incident_id": incident_counter,
                    "stage": stage,
                    "attack_type": attack_type,
                }
            )
        incident_counter += 1

    for _ in range(n_benign):
        records.append(
            {
                "label": 0,
                "incident_id": -1,
                "stage": -1,
                "attack_type": -1,
            }
        )

    rng.shuffle(records)
    sequence_values: list[np.ndarray] = []
    node_values: list[np.ndarray] = []
    adjacency_values: list[np.ndarray] = []
    topology_values: list[np.ndarray] = []
    labels: list[int] = []
    incident_ids: list[int] = []
    stages: list[int] = []
    attack_types: list[int] = []

    for record in records:
        sequence = _baseline_sequence(rng, config, condition)
        adjacency = _baseline_adjacency(rng, condition)
        if record["label"]:
            sequence, adjacency = _inject_attack(
                rng,
                sequence,
                adjacency,
                int(record["attack_type"]),
                int(record["stage"]),
                condition,
            )
        node = _derive_node_features(
            rng,
            sequence,
            adjacency,
            int(record["label"]),
            int(record["stage"]),
        )
        topology, _ = topological_vector(
            adjacency,
            grid_size=config.persistence_grid,
            sigma=config.persistence_sigma,
        )
        sequence_values.append(sequence)
        node_values.append(node)
        adjacency_values.append(adjacency)
        topology_values.append(topology)
        labels.append(int(record["label"]))
        incident_ids.append(int(record["incident_id"]))
        stages.append(int(record["stage"]))
        attack_types.append(int(record["attack_type"]))

    return SyntheticSOCDataset(
        sequence=np.stack(sequence_values).astype(np.float32),
        node_features=np.stack(node_values).astype(np.float32),
        adjacency=np.stack(adjacency_values).astype(np.float32),
        topology=np.stack(topology_values).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        incident_ids=np.asarray(incident_ids, dtype=np.int64),
        stages=np.asarray(stages, dtype=np.int64),
        attack_types=np.asarray(attack_types, dtype=np.int64),
        sample_ids=np.arange(sample_id_offset, sample_id_offset + n_samples, dtype=np.int64),
    )


def recompute_derived_modalities(
    dataset: SyntheticSOCDataset,
    config: ExperimentConfig,
    seed: int,
) -> SyntheticSOCDataset:
    """Recompute node and topological features after telemetry perturbation."""
    rng = np.random.default_rng(seed)
    updated = dataset.copy()
    nodes: list[np.ndarray] = []
    topology: list[np.ndarray] = []
    for idx in range(len(updated.labels)):
        nodes.append(
            _derive_node_features(
                rng,
                updated.sequence[idx],
                updated.adjacency[idx],
                int(updated.labels[idx]),
                int(updated.stages[idx]),
            )
        )
        vector, _ = topological_vector(
            updated.adjacency[idx],
            config.persistence_grid,
            config.persistence_sigma,
        )
        topology.append(vector)
    updated.node_features = np.stack(nodes).astype(np.float32)
    updated.topology = np.stack(topology).astype(np.float32)
    return updated


def create_missing_telemetry_condition(
    dataset: SyntheticSOCDataset,
    config: ExperimentConfig,
    seed: int,
    missing_fraction: float = 0.20,
) -> SyntheticSOCDataset:
    """Remove temporal observations and graph interactions at random."""
    rng = np.random.default_rng(seed)
    updated = dataset.copy()
    sequence_mask = rng.random(updated.sequence.shape) < missing_fraction
    updated.sequence[sequence_mask] = 0.0

    for idx in range(len(updated.labels)):
        upper = np.triu(rng.random(updated.adjacency[idx].shape) < missing_fraction, 1)
        mask = upper | upper.T
        updated.adjacency[idx][mask] = 0.0
    return recompute_derived_modalities(updated, config, seed + 101)


def create_timestamp_perturbation_condition(
    dataset: SyntheticSOCDataset,
    config: ExperimentConfig,
    seed: int,
) -> SyntheticSOCDataset:
    """Perturb event order while retaining the observed relational graph."""
    rng = np.random.default_rng(seed)
    updated = dataset.copy()
    for idx in range(len(updated.labels)):
        if rng.random() < 0.75:
            swap_count = int(rng.integers(2, 5))
            for _ in range(swap_count):
                i, j = rng.choice(config.sequence_length, size=2, replace=False)
                updated.sequence[idx, [i, j]] = updated.sequence[idx, [j, i]]
        shift = int(rng.integers(-3, 4))
        updated.sequence[idx] = np.roll(updated.sequence[idx], shift, axis=0)
    # Graph and topology remain unchanged because this stressor isolates time.
    return updated


# ---------------------------------------------------------------------------
# Preprocessing and PyTorch data structures
# ---------------------------------------------------------------------------


@dataclass
class ModalityScalers:
    sequence: StandardScaler
    nodes: StandardScaler
    topology: StandardScaler


def fit_scalers(dataset: SyntheticSOCDataset) -> ModalityScalers:
    """Fit training-only standardization transforms for every numeric branch."""
    sequence_scaler = StandardScaler().fit(
        dataset.sequence.reshape(-1, dataset.sequence.shape[-1])
    )
    node_scaler = StandardScaler().fit(
        dataset.node_features.reshape(-1, dataset.node_features.shape[-1])
    )
    topology_scaler = StandardScaler().fit(dataset.topology)
    return ModalityScalers(sequence_scaler, node_scaler, topology_scaler)


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Apply symmetric GCN normalization to a batch of adjacency matrices."""
    result = np.empty_like(adjacency, dtype=np.float32)
    for index, matrix in enumerate(adjacency):
        augmented = matrix.astype(np.float64) + np.eye(matrix.shape[0], dtype=np.float64)
        degree = augmented.sum(axis=1)
        inverse_sqrt = np.power(np.clip(degree, 1e-10, None), -0.5)
        normalized = inverse_sqrt[:, None] * augmented * inverse_sqrt[None, :]
        result[index] = normalized.astype(np.float32)
    return result


def transform_dataset(
    dataset: SyntheticSOCDataset,
    scalers: ModalityScalers,
) -> SyntheticSOCDataset:
    """Standardize modalities using training parameters."""
    transformed = dataset.copy()
    shape = transformed.sequence.shape
    transformed.sequence = scalers.sequence.transform(
        transformed.sequence.reshape(-1, shape[-1])
    ).reshape(shape).astype(np.float32)
    shape = transformed.node_features.shape
    transformed.node_features = scalers.nodes.transform(
        transformed.node_features.reshape(-1, shape[-1])
    ).reshape(shape).astype(np.float32)
    transformed.topology = scalers.topology.transform(transformed.topology).astype(np.float32)
    transformed.adjacency = normalize_adjacency(transformed.adjacency)
    return transformed


class MultimodalTorchDataset(Dataset):
    """PyTorch adapter for normalized multimodal SOC observations."""

    def __init__(self, data: SyntheticSOCDataset):
        self.sequence = torch.from_numpy(data.sequence)
        self.nodes = torch.from_numpy(data.node_features)
        self.adjacency = torch.from_numpy(data.adjacency)
        self.topology = torch.from_numpy(data.topology)
        self.labels = torch.from_numpy(data.labels.astype(np.float32))

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.sequence[index],
            self.nodes[index],
            self.adjacency[index],
            self.topology[index],
            self.labels[index],
        )


# ---------------------------------------------------------------------------
# Neural architecture
# ---------------------------------------------------------------------------


class TemporalEncoder(nn.Module):
    """Compact Transformer encoder for ordered security events."""

    def __init__(self, input_dim: int, sequence_length: int, hidden_dim: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(sequence) + self.position[:, : sequence.shape[1]]
        hidden = self.encoder(hidden)
        return self.normalization(hidden.mean(dim=1))


class GraphConvolution(nn.Module):
    """A dense graph-convolution layer for small heterogeneous SOC graphs."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.bmm(adjacency, nodes)
        return self.linear(propagated)


class GraphEncoder(nn.Module):
    """Two-layer GCN with mean-max graph pooling."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.gcn1(nodes, adjacency))
        hidden = torch.relu(self.gcn2(hidden, adjacency))
        mean_pool = hidden.mean(dim=1)
        max_pool = hidden.max(dim=1).values
        return self.normalization(self.output(torch.cat([mean_pool, max_pool], dim=1)))


class TopologyEncoder(nn.Module):
    """Nonlinear encoder for persistence images and lifetime summaries."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, topology: torch.Tensor) -> torch.Tensor:
        return self.network(topology)


class MultimodalDetector(nn.Module):
    """Gated temporal, graph, and topological fusion detector."""

    def __init__(
        self,
        config: ExperimentConfig,
        topology_dim: int,
        use_temporal: bool,
        use_graph: bool,
        use_topology: bool,
    ):
        super().__init__()
        if not any((use_temporal, use_graph, use_topology)):
            raise ValueError("At least one branch must be enabled")
        self.use_temporal = use_temporal
        self.use_graph = use_graph
        self.use_topology = use_topology
        self.temporal = (
            TemporalEncoder(config.sequence_features, config.sequence_length, config.hidden_dim)
            if use_temporal
            else None
        )
        self.graph = (
            GraphEncoder(config.node_features, config.hidden_dim) if use_graph else None
        )
        self.topology = (
            TopologyEncoder(topology_dim, config.hidden_dim) if use_topology else None
        )
        branch_count = int(use_temporal) + int(use_graph) + int(use_topology)
        self.branch_count = branch_count
        self.gate = nn.Linear(config.hidden_dim * branch_count, branch_count)
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.LayerNorm(config.hidden_dim),
        )
        self.classifier = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        nodes: torch.Tensor,
        adjacency: torch.Tensor,
        topology: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branches: list[torch.Tensor] = []
        if self.temporal is not None:
            branches.append(self.temporal(sequence))
        if self.graph is not None:
            branches.append(self.graph(nodes, adjacency))
        if self.topology is not None:
            branches.append(self.topology(topology))

        concatenated = torch.cat(branches, dim=1)
        gate_values = torch.softmax(self.gate(concatenated), dim=1)
        stacked = torch.stack(branches, dim=1)
        fused = torch.sum(stacked * gate_values.unsqueeze(-1), dim=1)
        embedding = self.fusion(fused)
        logits = self.classifier(embedding).squeeze(1)
        if return_embedding:
            return logits, embedding, gate_values
        return logits


def pairwise_topological_consistency(
    embeddings: torch.Tensor,
    topology: torch.Tensor,
) -> torch.Tensor:
    """Match normalized pairwise distances in latent and topological spaces."""
    if embeddings.shape[0] < 3:
        return torch.zeros((), dtype=embeddings.dtype, device=embeddings.device)
    latent_distance = torch.pdist(embeddings, p=2)
    topological_distance = torch.pdist(topology, p=2)
    latent_distance = latent_distance / (latent_distance.mean().detach() + 1e-8)
    topological_distance = topological_distance / (topological_distance.mean().detach() + 1e-8)
    return torch.mean((latent_distance - topological_distance) ** 2)


# ---------------------------------------------------------------------------
# Training, calibration, and metrics
# ---------------------------------------------------------------------------


@dataclass
class TrainedNeuralModel:
    model: MultimodalDetector
    temperature: float
    threshold: float
    validation_probabilities: np.ndarray


@dataclass
class LogisticModelBundle:
    model: LogisticRegression
    temperature: float
    threshold: float
    validation_probabilities: np.ndarray


def model_specification(model_key: str) -> tuple[bool, bool, bool, float]:
    """Return enabled branches and topology regularization weight."""
    specifications = {
        "temporal": (True, False, False, 0.0),
        "graph": (False, True, False, 0.0),
        "temporal_graph": (True, True, False, 0.0),
        "full_no_consistency": (True, True, True, 0.0),
        "full": (True, True, True, 1.0),
    }
    if model_key not in specifications:
        raise KeyError(f"Unknown neural model: {model_key}")
    return specifications[model_key]


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    logits = np.log(probabilities / (1.0 - probabilities))
    scaled = logits / max(float(temperature), 1e-3)
    return 1.0 / (1.0 + np.exp(-scaled))


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive scalar temperature by validation log loss."""
    probabilities = np.clip(np.asarray(probabilities), 1e-7, 1.0 - 1e-7)
    labels = np.asarray(labels, dtype=float)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        calibrated = np.clip(_temperature_scale(probabilities, temperature), 1e-7, 1 - 1e-7)
        loss = -np.mean(labels * np.log(calibrated) + (1.0 - labels) * np.log(1.0 - calibrated))
        return float(loss)

    result = minimize_scalar(objective, bounds=(-2.2, 2.2), method="bounded")
    return float(np.exp(result.x)) if result.success else 1.0


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(labels[mask]))
        confidence = float(np.mean(probabilities[mask]))
        error += float(np.mean(mask)) * abs(accuracy - confidence)
    return float(error)


def choose_operating_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a validation threshold that maximizes MCC, then F1."""
    best_threshold = 0.5
    best_tuple = (-np.inf, -np.inf, -np.inf)
    for threshold in np.linspace(0.08, 0.92, 169):
        predictions = (probabilities >= threshold).astype(int)
        if len(np.unique(predictions)) < 2:
            continue
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        precision = precision_score(labels, predictions, zero_division=0)
        # SOC operating points must penalize missed incidents more heavily than
        # ordinary balanced classification while still controlling alert load.
        utility = 0.55 * float(mcc) + 0.30 * float(f1) + 0.15 * float(recall)
        candidate = (utility, float(mcc), float(precision))
        if candidate > best_tuple:
            best_tuple = candidate
            best_threshold = float(threshold)
    return best_threshold


def predict_neural_raw(
    model: MultimodalDetector,
    dataset: SyntheticSOCDataset,
    batch_size: int,
    return_embeddings: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Run deterministic neural inference."""
    loader = DataLoader(MultimodalTorchDataset(dataset), batch_size=batch_size, shuffle=False)
    model.eval()
    probability_parts: list[np.ndarray] = []
    embedding_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []
    with torch.no_grad():
        for sequence, nodes, adjacency, topology, _ in loader:
            if return_embeddings:
                logits, embeddings, gates = model(
                    sequence, nodes, adjacency, topology, return_embedding=True
                )
                embedding_parts.append(embeddings.cpu().numpy())
                gate_parts.append(gates.cpu().numpy())
            else:
                logits = model(sequence, nodes, adjacency, topology)
            probability_parts.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(probability_parts)
    embeddings = np.concatenate(embedding_parts) if embedding_parts else None
    gates = np.concatenate(gate_parts) if gate_parts else None
    return probabilities, embeddings, gates


def train_neural_model(
    model_key: str,
    train: SyntheticSOCDataset,
    validation: SyntheticSOCDataset,
    config: ExperimentConfig,
    seed: int,
) -> TrainedNeuralModel:
    """Train one neural model with weighted BCE and early stopping."""
    set_global_seed(seed)
    use_temporal, use_graph, use_topology, consistency_flag = model_specification(model_key)
    model = MultimodalDetector(
        config,
        topology_dim=train.topology.shape[1],
        use_temporal=use_temporal,
        use_graph=use_graph,
        use_topology=use_topology,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    positive = float(np.sum(train.labels == 1))
    negative = float(np.sum(train.labels == 0))
    pos_weight = torch.tensor([negative / max(positive, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        MultimodalTorchDataset(train),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -np.inf
    epochs_without_improvement = 0

    for _epoch in range(config.epochs):
        model.train()
        for sequence, nodes, adjacency, topology, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            sequence_input = sequence
            nodes_input = nodes
            topology_input = topology
            if model_key == "full":
                # Modality dropout discourages a brittle dependence on any
                # single surface representation while the topological
                # consistency term preserves relational organization.
                sequence_mask = (torch.rand_like(sequence) > 0.08).to(sequence.dtype)
                node_mask = (torch.rand_like(nodes) > 0.06).to(nodes.dtype)
                topology_mask = (torch.rand_like(topology) > 0.05).to(topology.dtype)
                sequence_input = sequence * sequence_mask
                nodes_input = nodes * node_mask
                topology_input = topology * topology_mask
            logits, embeddings, _ = model(
                sequence_input, nodes_input, adjacency, topology_input, return_embedding=True
            )
            loss = criterion(logits, labels)
            if consistency_flag > 0:
                consistency = pairwise_topological_consistency(embeddings, topology)
                loss = loss + (
                    config.topology_consistency_weight
                    * consistency_flag
                    * consistency
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        raw_validation, _, _ = predict_neural_raw(
            model, validation, batch_size=config.batch_size
        )
        validation_score = average_precision_score(validation.labels, raw_validation)
        if validation_score > best_score + 1e-5:
            best_score = float(validation_score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError(f"Training failed to produce a valid state for {model_key}")
    model.load_state_dict(best_state)
    raw_validation, _, _ = predict_neural_raw(model, validation, config.batch_size)
    temperature = fit_temperature(raw_validation, validation.labels)
    calibrated_validation = _temperature_scale(raw_validation, temperature)
    threshold = choose_operating_threshold(validation.labels, calibrated_validation)
    return TrainedNeuralModel(
        model=model,
        temperature=temperature,
        threshold=threshold,
        validation_probabilities=calibrated_validation,
    )


def aggregate_conventional_features(dataset: SyntheticSOCDataset) -> np.ndarray:
    """Construct non-topological summary features for logistic regression."""
    sequence_mean = dataset.sequence.mean(axis=1)
    sequence_std = dataset.sequence.std(axis=1)
    node_mean = dataset.node_features.mean(axis=1)
    node_std = dataset.node_features.std(axis=1)
    # The normalized adjacency still supports generic density statistics; no
    # persistence image or Betti summary is included in this baseline.
    return np.concatenate(
        [sequence_mean, sequence_std, node_mean, node_std],
        axis=1,
    ).astype(np.float32)


def train_logistic_model(
    train: SyntheticSOCDataset,
    validation: SyntheticSOCDataset,
    seed: int,
) -> LogisticModelBundle:
    """Fit and calibrate the conventional machine-learning baseline."""
    model = LogisticRegression(
        C=0.9,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1500,
        random_state=seed,
    )
    model.fit(aggregate_conventional_features(train), train.labels)
    raw_validation = model.predict_proba(aggregate_conventional_features(validation))[:, 1]
    temperature = fit_temperature(raw_validation, validation.labels)
    calibrated_validation = _temperature_scale(raw_validation, temperature)
    threshold = choose_operating_threshold(validation.labels, calibrated_validation)
    return LogisticModelBundle(model, temperature, threshold, calibrated_validation)


def calibrated_probabilities(
    bundle: TrainedNeuralModel | LogisticModelBundle,
    dataset: SyntheticSOCDataset,
    config: ExperimentConfig,
    return_embeddings: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Obtain calibrated probabilities and optional latent representations."""
    if isinstance(bundle, LogisticModelBundle):
        features = aggregate_conventional_features(dataset)
        raw = bundle.model.predict_proba(features)[:, 1]
        return _temperature_scale(raw, bundle.temperature), features if return_embeddings else None, None

    raw, embeddings, gates = predict_neural_raw(
        bundle.model,
        dataset,
        batch_size=config.batch_size,
        return_embeddings=return_embeddings,
    )
    return _temperature_scale(raw, bundle.temperature), embeddings, gates


def predictive_metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute imbalance-aware predictive and calibration metrics."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "Balanced accuracy": float(balanced_accuracy_score(labels, predictions)),
        "Brier score": float(brier_score_loss(labels, probabilities)),
        "ECE": float(expected_calibration_error(labels, probabilities)),
    }


def stratified_bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    repetitions: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Estimate 95% stratified bootstrap intervals for core metrics."""
    rng = np.random.default_rng(seed)
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    samples: dict[str, list[float]] = {
        "Precision": [], "Recall": [], "F1": [], "AUPRC": [], "MCC": [], "Balanced accuracy": []
    }
    for _ in range(repetitions):
        selected = np.concatenate(
            [
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
            ]
        )
        rng.shuffle(selected)
        metrics = predictive_metric_values(labels[selected], probabilities[selected], threshold)
        for name in samples:
            samples[name].append(metrics[name])
    return {
        name: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for name, values in samples.items()
    }


def format_estimate_interval(value: float, interval: tuple[float, float]) -> str:
    """Format a point estimate and 95% interval for a manuscript table."""
    return f"{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"


# ---------------------------------------------------------------------------
# Operational evaluation
# ---------------------------------------------------------------------------


def cluster_purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    """Compute external cluster purity."""
    total = len(labels)
    if total == 0:
        return float("nan")
    correct = 0
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        counts = np.bincount(labels[mask])
        correct += int(counts.max()) if counts.size else 0
    return float(correct / total)


def incident_recovery_curve(
    dataset: SyntheticSOCDataset,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reviewed-alert counts and cumulative unique-incident recovery."""
    positive_incidents = set(dataset.incident_ids[dataset.incident_ids >= 0].tolist())
    order = np.argsort(-probabilities)
    recovered: set[int] = set()
    x_values: list[int] = []
    y_values: list[float] = []
    for rank, index in enumerate(order, start=1):
        incident = int(dataset.incident_ids[index])
        if incident >= 0:
            recovered.add(incident)
        x_values.append(rank)
        y_values.append(len(recovered) / max(1, len(positive_incidents)))
    return np.asarray(x_values), np.asarray(y_values)


def operational_metrics(
    dataset: SyntheticSOCDataset,
    probabilities: np.ndarray,
    threshold: float,
    embeddings: np.ndarray,
    config: ExperimentConfig,
    latency_ms: float,
    model_memory_mb: float,
) -> dict[str, float]:
    """Compute analyst-facing workload, timeliness, and grouping metrics."""
    predictions = probabilities >= threshold
    alert_rate = float(np.mean(predictions))
    alerts_per_shift = alert_rate * config.windows_per_shift

    x_curve, y_curve = incident_recovery_curve(dataset, probabilities)
    top_20_index = min(19, len(y_curve) - 1)
    top_20_recovery = float(y_curve[top_20_index]) if len(y_curve) else 0.0

    detection_delays: list[float] = []
    for incident in sorted(set(dataset.incident_ids[dataset.incident_ids >= 0].tolist())):
        indices = np.flatnonzero(dataset.incident_ids == incident)
        ordered = indices[np.argsort(dataset.stages[indices])]
        flagged = ordered[probabilities[ordered] >= threshold]
        if len(flagged):
            first_stage = int(dataset.stages[flagged[0]])
            delay = (first_stage + 1) * config.window_minutes
        else:
            delay = (config.attack_stages + 1) * config.window_minutes
        detection_delays.append(float(delay))
    mean_detection_delay = float(np.mean(detection_delays)) if detection_delays else float("nan")

    positive_mask = dataset.labels == 1
    if np.sum(positive_mask) >= len(ATTACK_NAMES):
        kmeans = KMeans(
            n_clusters=len(ATTACK_NAMES),
            n_init=20,
            random_state=config.random_seed,
        )
        clusters = kmeans.fit_predict(embeddings[positive_mask])
        purity = cluster_purity(dataset.attack_types[positive_mask], clusters)
    else:
        purity = float("nan")

    queue_size = max(1, int(round(0.10 * len(probabilities))))
    top_indices = np.argsort(-probabilities)[:queue_size]
    precision_at_queue = float(np.mean(dataset.labels[top_indices]))
    return {
        "Alerts per shift": alerts_per_shift,
        "Review minutes per shift": alerts_per_shift * config.analyst_minutes_per_alert,
        "Precision in top 10%": precision_at_queue,
        "Top-20 incident recovery": top_20_recovery,
        "Mean time to detect (min)": mean_detection_delay,
        "Incident clustering purity": purity,
        "Latency (ms/window)": latency_ms,
        "Model memory (MB)": model_memory_mb,
    }


def measure_neural_latency_and_memory(
    bundle: TrainedNeuralModel,
    dataset: SyntheticSOCDataset,
    config: ExperimentConfig,
) -> tuple[float, float]:
    """Measure CPU inference time per window and parameter memory."""
    model = bundle.model
    model.eval()
    sample_count = min(128, len(dataset.labels))
    sequence = torch.from_numpy(dataset.sequence[:sample_count])
    nodes = torch.from_numpy(dataset.node_features[:sample_count])
    adjacency = torch.from_numpy(dataset.adjacency[:sample_count])
    topology = torch.from_numpy(dataset.topology[:sample_count])
    with torch.no_grad():
        for _ in range(5):
            _ = model(sequence, nodes, adjacency, topology)
        start = time.perf_counter()
        repetitions = 30
        for _ in range(repetitions):
            _ = model(sequence, nodes, adjacency, topology)
        elapsed = time.perf_counter() - start
    latency = elapsed * 1000.0 / (repetitions * sample_count)
    memory = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1e6
    return float(latency), float(memory)


def measure_logistic_latency_and_memory(
    bundle: LogisticModelBundle,
    dataset: SyntheticSOCDataset,
) -> tuple[float, float]:
    """Measure logistic-regression inference latency and coefficient memory."""
    features = aggregate_conventional_features(dataset)
    sample_count = min(256, len(features))
    for _ in range(10):
        _ = bundle.model.predict_proba(features[:sample_count])
    start = time.perf_counter()
    repetitions = 200
    for _ in range(repetitions):
        _ = bundle.model.predict_proba(features[:sample_count])
    elapsed = time.perf_counter() - start
    latency = elapsed * 1000.0 / (repetitions * sample_count)
    memory = (bundle.model.coef_.nbytes + bundle.model.intercept_.nbytes) / 1e6
    return float(latency), float(memory)


# ---------------------------------------------------------------------------
# Visualization and artifact creation
# ---------------------------------------------------------------------------


def configure_plot() -> None:
    """Set conservative, publication-compatible plot defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.autolayout": True,
        }
    )


def save_pr_curve(
    test: SyntheticSOCDataset,
    probabilities: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot precision-recall curves for all compared detectors."""
    configure_plot()
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for model_key, values in probabilities.items():
        precision, recall, _ = precision_recall_curve(test.labels, values)
        score = average_precision_score(test.labels, values)
        axis.plot(recall, precision, linewidth=1.8, label=f"{MODEL_LABELS[model_key]} (AUPRC={score:.3f})")
    prevalence = float(np.mean(test.labels))
    axis.axhline(prevalence, linestyle="--", linewidth=1.0, label=f"Prevalence={prevalence:.2f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Precision-recall behavior on the in-distribution SOC test set")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left", frameon=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_ablation_figure(metrics: pd.DataFrame, output_path: Path) -> None:
    """Compare AUPRC and MCC across architecture ablations."""
    configure_plot()
    selected = metrics.set_index("Model").loc[
        [
            MODEL_LABELS["temporal"],
            MODEL_LABELS["graph"],
            MODEL_LABELS["temporal_graph"],
            MODEL_LABELS["full_no_consistency"],
            MODEL_LABELS["full"],
        ]
    ]
    x = np.arange(len(selected))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.3, 4.4))
    axis.bar(x - width / 2, selected["AUPRC"], width, label="AUPRC")
    axis.bar(x + width / 2, selected["MCC"], width, label="MCC")
    axis.set_xticks(x)
    axis.set_xticklabels(
        ["Temporal", "Graph", "T-G fusion", "Topology\nno consistency", "Topology-guided"],
        rotation=0,
    )
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Score")
    axis.set_title("Ablation of temporal, relational, and topological components")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_robustness_heatmap(
    robustness: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot AUPRC across all stress conditions."""
    configure_plot()
    pivot = robustness.pivot(index="Model", columns="Condition", values="AUPRC")
    model_order = [MODEL_LABELS[key] for key in MODEL_LABELS]
    condition_order = [STRESS_LABELS[key] for key in STRESS_LABELS]
    pivot = pivot.loc[model_order, condition_order]
    values = pivot.to_numpy()
    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    image = axis.imshow(values, aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(condition_order)))
    axis.set_xticklabels(condition_order, rotation=24, ha="right")
    axis.set_yticks(np.arange(len(model_order)))
    axis.set_yticklabels(model_order)
    axis.set_title("AUPRC under distributional and telemetry stress")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.028, pad=0.02, label="AUPRC")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def reliability_points(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean confidence, observed frequency, and bin counts."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    confidence: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if np.any(mask):
            confidence.append(float(np.mean(probabilities[mask])))
            observed.append(float(np.mean(labels[mask])))
            counts.append(int(np.sum(mask)))
    return np.asarray(confidence), np.asarray(observed), np.asarray(counts)


def save_calibration_figure(
    test: SyntheticSOCDataset,
    probabilities: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot reliability curves for representative model classes."""
    configure_plot()
    fig, axis = plt.subplots(figsize=(6.3, 4.6))
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Perfect calibration")
    for model_key in ("logistic", "temporal_graph", "full"):
        confidence, observed, _ = reliability_points(test.labels, probabilities[model_key])
        ece = expected_calibration_error(test.labels, probabilities[model_key])
        axis.plot(confidence, observed, marker="o", linewidth=1.7, label=f"{MODEL_LABELS[model_key]} (ECE={ece:.3f})")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed attack frequency")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Reliability of calibrated attack probabilities")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_topological_explanation(
    raw_dataset: SyntheticSOCDataset,
    probabilities: np.ndarray,
    output_path: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Visualize one correctly ranked attack graph and its persistence diagrams."""
    positive_indices = np.flatnonzero(raw_dataset.labels == 1)
    selected = int(positive_indices[np.argmax(probabilities[positive_indices])])
    adjacency = raw_dataset.adjacency[selected]
    diagrams = persistent_intervals_from_adjacency(adjacency, maximum_dimension=1)

    configure_plot()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    graph = nx.Graph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[1]):
            if adjacency[i, j] >= 0.55:
                graph.add_edge(i, j, weight=float(adjacency[i, j]))
    position = nx.spring_layout(graph, seed=19, weight="weight")
    edge_widths = [1.0 + 3.2 * graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx(
        graph,
        pos=position,
        labels={idx: NODE_NAMES[idx] for idx in graph.nodes},
        node_size=580,
        font_size=6.5,
        width=edge_widths,
        ax=axes[0],
    )
    axes[0].set_title("High-risk entity-interaction graph")
    axes[0].axis("off")

    for axis, dimension in zip(axes[1:], (0, 1)):
        diagram = diagrams[dimension]
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
        if diagram.size:
            axis.scatter(diagram[:, 0], diagram[:, 1], s=26 + 80 * (diagram[:, 1] - diagram[:, 0]))
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Birth")
        axis.set_ylabel("Death")
        axis.set_title(f"Persistence diagram H{dimension}")
        axis.grid(alpha=0.20)
    fig.suptitle(
        f"Topology-based explanation for {ATTACK_NAMES[int(raw_dataset.attack_types[selected])]} "
        f"at stage {int(raw_dataset.stages[selected]) + 1}",
        fontsize=10.5,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    h1 = diagrams[1]
    h1_lifetimes = h1[:, 1] - h1[:, 0] if h1.size else np.asarray([])
    return {
        "sample_index": selected,
        "sample_id": int(raw_dataset.sample_ids[selected]),
        "attack_type": ATTACK_NAMES[int(raw_dataset.attack_types[selected])],
        "stage": int(raw_dataset.stages[selected]) + 1,
        "probability": float(probabilities[selected]),
        "persistent_h1_count": int(len(h1_lifetimes)),
        "maximum_h1_lifetime": float(np.max(h1_lifetimes)) if len(h1_lifetimes) else 0.0,
    }


def save_incident_recovery_figure(
    test: SyntheticSOCDataset,
    probabilities: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot incident recovery as analysts review a ranked alert queue."""
    configure_plot()
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    for model_key in ("logistic", "temporal", "graph", "temporal_graph", "full"):
        x, y = incident_recovery_curve(test, probabilities[model_key])
        limit = min(160, len(x))
        axis.plot(x[:limit], y[:limit], linewidth=1.8, label=MODEL_LABELS[model_key])
    axis.set_xlabel("Alerts reviewed in descending risk order")
    axis.set_ylabel("Fraction of distinct incidents recovered")
    axis.set_xlim(1, min(160, len(test.labels)))
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Operational recovery of multistage incidents")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experimental workflow
# ---------------------------------------------------------------------------


def create_output_directories(root: Path) -> dict[str, Path]:
    """Create the three manuscript-aligned result folders."""
    directories = {section: root / section for section in ("5.1", "5.2", "5.3")}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def dataframe_to_markdown_file(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a plain-text Markdown rendering without optional dependencies."""
    columns = [str(column) for column in dataframe.columns]
    rows = [[str(value) for value in row] for row in dataframe.to_numpy().tolist()]
    widths = [len(column) for column in columns]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    lines = [
        "| " + " | ".join(column.ljust(width) for column, width in zip(columns, widths)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    output_root: Path,
    config: ExperimentConfig,
    training_seeds: Sequence[int],
) -> dict[str, Any]:
    """Execute simulation, model comparison, stress testing, and reporting."""
    set_global_seed(config.random_seed)
    directories = create_output_directories(output_root)

    print("[1/7] Generating the synthetic SOC benchmark and persistence features...")
    raw_train = generate_synthetic_soc(
        config.n_train, config, config.random_seed + 1, "standard", sample_id_offset=0
    )
    # Controlled training-only label noise represents imperfect analyst
    # dispositions without contaminating validation or test ground truth.
    noise_rng = np.random.default_rng(config.random_seed + 88)
    noisy_indices = noise_rng.choice(
        len(raw_train.labels),
        size=max(1, int(round(0.04 * len(raw_train.labels)))),
        replace=False,
    )
    raw_train.labels[noisy_indices] = 1 - raw_train.labels[noisy_indices]

    raw_validation = generate_synthetic_soc(
        config.n_validation,
        config,
        config.random_seed + 2,
        "standard",
        sample_id_offset=config.n_train,
    )
    raw_test = generate_synthetic_soc(
        config.n_test,
        config,
        config.random_seed + 3,
        "standard",
        sample_id_offset=config.n_train + config.n_validation,
    )
    raw_drift = generate_synthetic_soc(
        config.n_test,
        config,
        config.random_seed + 4,
        "drift",
        sample_id_offset=20_000,
    )
    raw_adversarial = generate_synthetic_soc(
        config.n_test,
        config,
        config.random_seed + 5,
        "adversarial",
        sample_id_offset=30_000,
    )
    raw_missing = create_missing_telemetry_condition(
        raw_test, config, config.random_seed + 6
    )
    raw_timestamp = create_timestamp_perturbation_condition(
        raw_test, config, config.random_seed + 7
    )

    scalers = fit_scalers(raw_train)
    train = transform_dataset(raw_train, scalers)
    validation = transform_dataset(raw_validation, scalers)
    stress_data = {
        "standard": transform_dataset(raw_test, scalers),
        "drift": transform_dataset(raw_drift, scalers),
        "missing": transform_dataset(raw_missing, scalers),
        "timestamp": transform_dataset(raw_timestamp, scalers),
        "adversarial": transform_dataset(raw_adversarial, scalers),
    }

    print("[2/7] Fitting the conventional and neural baselines...")
    trained: dict[str, list[TrainedNeuralModel | LogisticModelBundle]] = {}
    trained["logistic"] = [train_logistic_model(train, validation, config.random_seed)]
    for model_key in ("temporal", "graph", "temporal_graph", "full_no_consistency", "full"):
        trained[model_key] = []
        for seed in training_seeds:
            print(f"      training {MODEL_LABELS[model_key]} with seed {seed}")
            trained[model_key].append(
                train_neural_model(model_key, train, validation, config, int(seed))
            )

    # Use the median learned threshold and average calibrated probabilities
    # across independent neural initializations.
    thresholds: dict[str, float] = {}
    stress_probabilities: dict[str, dict[str, np.ndarray]] = {
        condition: {} for condition in stress_data
    }
    representative_embeddings: dict[str, np.ndarray] = {}
    representative_gates: dict[str, np.ndarray | None] = {}

    print("[3/7] Evaluating predictive performance and calibration...")
    for model_key, bundles in trained.items():
        thresholds[model_key] = float(np.median([bundle.threshold for bundle in bundles]))
        for condition, dataset in stress_data.items():
            probability_runs: list[np.ndarray] = []
            for run_index, bundle in enumerate(bundles):
                probabilities, embeddings, gates = calibrated_probabilities(
                    bundle,
                    dataset,
                    config,
                    return_embeddings=(condition == "standard" and run_index == 0),
                )
                probability_runs.append(probabilities)
                if condition == "standard" and run_index == 0 and embeddings is not None:
                    representative_embeddings[model_key] = embeddings
                    representative_gates[model_key] = gates
            stress_probabilities[condition][model_key] = np.mean(
                np.stack(probability_runs), axis=0
            )

    # Table 1: point estimates and bootstrap intervals.
    test = stress_data["standard"]
    table_1_rows: list[dict[str, Any]] = []
    table_1_numeric_rows: list[dict[str, Any]] = []
    bootstrap_seed = config.random_seed + 501
    for model_key in MODEL_LABELS:
        probabilities = stress_probabilities["standard"][model_key]
        values = predictive_metric_values(test.labels, probabilities, thresholds[model_key])
        intervals = stratified_bootstrap_intervals(
            test.labels,
            probabilities,
            thresholds[model_key],
            repetitions=config.bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        bootstrap_seed += 1
        formatted = {"Model": MODEL_LABELS[model_key]}
        for metric_name in ("Precision", "Recall", "F1", "AUPRC", "MCC", "Balanced accuracy"):
            formatted[metric_name] = format_estimate_interval(values[metric_name], intervals[metric_name])
        formatted["Brier"] = f"{values['Brier score']:.3f}"
        formatted["ECE"] = f"{values['ECE']:.3f}"
        table_1_rows.append(formatted)
        table_1_numeric_rows.append({"Model": MODEL_LABELS[model_key], **values})

    table_1 = pd.DataFrame(table_1_rows)
    table_1_numeric = pd.DataFrame(table_1_numeric_rows)
    table_1.to_csv(directories["5.1"] / "table_1_predictive_metrics.csv", index=False)
    table_1_numeric.to_csv(directories["5.1"] / "table_1_predictive_metrics_numeric.csv", index=False)
    dataframe_to_markdown_file(table_1, directories["5.1"] / "table_1_predictive_metrics.md")

    save_pr_curve(test, stress_probabilities["standard"], directories["5.1"] / "figure_1_precision_recall.png")
    save_ablation_figure(table_1_numeric, directories["5.1"] / "figure_2_ablation.png")

    print("[4/7] Measuring robustness under drift and telemetry stress...")
    robustness_rows: list[dict[str, Any]] = []
    for model_key in MODEL_LABELS:
        for condition, dataset in stress_data.items():
            probabilities = stress_probabilities[condition][model_key]
            values = predictive_metric_values(dataset.labels, probabilities, thresholds[model_key])
            robustness_rows.append(
                {
                    "Model": MODEL_LABELS[model_key],
                    "Condition": STRESS_LABELS[condition],
                    "AUPRC": values["AUPRC"],
                    "MCC": values["MCC"],
                    "F1": values["F1"],
                    "Brier score": values["Brier score"],
                    "ECE": values["ECE"],
                }
            )
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(directories["5.2"] / "robustness_long.csv", index=False)

    table_2_rows: list[dict[str, Any]] = []
    for model_key in MODEL_LABELS:
        subset = robustness[robustness["Model"] == MODEL_LABELS[model_key]].set_index("Condition")
        in_distribution = float(subset.loc[STRESS_LABELS["standard"], "AUPRC"])
        row: dict[str, Any] = {"Model": MODEL_LABELS[model_key]}
        stress_scores: list[float] = []
        for condition in STRESS_LABELS:
            score = float(subset.loc[STRESS_LABELS[condition], "AUPRC"])
            row[STRESS_LABELS[condition]] = f"{score:.3f}"
            if condition != "standard":
                stress_scores.append(score)
        worst = min(stress_scores)
        row["Worst-case loss (%)"] = f"{100.0 * (in_distribution - worst) / max(in_distribution, 1e-8):.1f}"
        table_2_rows.append(row)
    table_2 = pd.DataFrame(table_2_rows)
    table_2.to_csv(directories["5.2"] / "table_2_robustness.csv", index=False)
    dataframe_to_markdown_file(table_2, directories["5.2"] / "table_2_robustness.md")
    save_robustness_heatmap(robustness, directories["5.2"] / "figure_3_robustness_heatmap.png")
    save_calibration_figure(test, stress_probabilities["standard"], directories["5.2"] / "figure_4_calibration.png")

    print("[5/7] Computing operational metrics and analyst-facing explanations...")
    table_3_rows: list[dict[str, Any]] = []
    operational_numeric: dict[str, dict[str, float]] = {}
    for model_key, bundles in trained.items():
        representative_bundle = bundles[0]
        if isinstance(representative_bundle, LogisticModelBundle):
            latency, memory = measure_logistic_latency_and_memory(representative_bundle, test)
        else:
            latency, memory = measure_neural_latency_and_memory(representative_bundle, test, config)
        probabilities = stress_probabilities["standard"][model_key]
        embeddings = representative_embeddings[model_key]
        values = operational_metrics(
            test,
            probabilities,
            thresholds[model_key],
            embeddings,
            config,
            latency,
            memory,
        )
        operational_numeric[model_key] = values
        table_3_rows.append(
            {
                "Model": MODEL_LABELS[model_key],
                "Alerts/shift": f"{values['Alerts per shift']:.1f}",
                "Top-10% precision": f"{values['Precision in top 10%']:.3f}",
                "Top-20 incident recovery": f"{values['Top-20 incident recovery']:.3f}",
                "MTTD (min)": f"{values['Mean time to detect (min)']:.2f}",
                "Grouping purity": f"{values['Incident clustering purity']:.3f}",
                "Latency (ms)": f"{values['Latency (ms/window)']:.3f}",
                "Memory (MB)": f"{values['Model memory (MB)']:.3f}",
            }
        )
    table_3 = pd.DataFrame(table_3_rows)
    table_3.to_csv(directories["5.3"] / "table_3_operational_metrics.csv", index=False)
    pd.DataFrame.from_dict(operational_numeric, orient="index").to_csv(
        directories["5.3"] / "table_3_operational_metrics_numeric.csv"
    )
    dataframe_to_markdown_file(table_3, directories["5.3"] / "table_3_operational_metrics.md")

    explanation = save_topological_explanation(
        raw_test,
        stress_probabilities["standard"]["full"],
        directories["5.3"] / "figure_5_topological_explanation.png",
        config,
    )
    save_incident_recovery_figure(
        test,
        stress_probabilities["standard"],
        directories["5.3"] / "figure_6_incident_recovery.png",
    )

    print("[6/7] Saving predictions, model summaries, and reproducibility metadata...")
    prediction_frame = pd.DataFrame(
        {
            "sample_id": raw_test.sample_ids,
            "label": raw_test.labels,
            "incident_id": raw_test.incident_ids,
            "stage": raw_test.stages,
            "attack_type": raw_test.attack_types,
            **{
                f"probability_{model_key}": stress_probabilities["standard"][model_key]
                for model_key in MODEL_LABELS
            },
        }
    )
    prediction_frame.to_csv(output_root / "test_predictions.csv", index=False)

    gate_summary: dict[str, Any] = {}
    for model_key, gates in representative_gates.items():
        if gates is None:
            continue
        means = gates.mean(axis=0)
        if model_key in {"full", "full_no_consistency"}:
            branch_names = ["temporal", "graph", "topology"]
        elif model_key == "temporal_graph":
            branch_names = ["temporal", "graph"]
        elif model_key == "temporal":
            branch_names = ["temporal"]
        elif model_key == "graph":
            branch_names = ["graph"]
        else:
            branch_names = [f"branch_{i}" for i in range(len(means))]
        gate_summary[model_key] = {
            name: float(value) for name, value in zip(branch_names, means)
        }

    summary = {
        "config": asdict(config),
        "training_seeds": [int(seed) for seed in training_seeds],
        "model_thresholds": thresholds,
        "predictive_metrics": {
            row["Model"]: {
                key: float(value) if isinstance(value, (np.floating, float, int)) else value
                for key, value in row.items()
                if key != "Model"
            }
            for row in table_1_numeric_rows
        },
        "operational_metrics": operational_numeric,
        "gate_summary": gate_summary,
        "topological_explanation": explanation,
        "dataset": {
            "training_windows": int(len(raw_train.labels)),
            "validation_windows": int(len(raw_validation.labels)),
            "test_windows": int(len(raw_test.labels)),
            "test_attack_windows": int(np.sum(raw_test.labels)),
            "test_incidents": int(len(set(raw_test.incident_ids[raw_test.incident_ids >= 0].tolist()))),
            "topology_feature_dimension": int(raw_test.topology.shape[1]),
        },
    }
    (output_root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    requirements = "\n".join(
        [
            "numpy>=2.0",
            "pandas>=2.0",
            "scipy>=1.11",
            "scikit-learn>=1.4",
            "matplotlib>=3.8",
            "networkx>=3.2",
            "torch>=2.2",
        ]
    ) + "\n"
    (output_root / "requirements.txt").write_text(requirements, encoding="utf-8")

    readme = f"""# Topology-Guided SOC Experiment Results

This directory was generated by `topological_soc_experiment.py`.

## Reproduction

```bash
python -m pip install -r requirements.txt
python topological_soc_experiment.py --output-dir {output_root.name} --epochs {config.epochs} --seeds {' '.join(str(seed) for seed in training_seeds)}
```

## Directory structure

- `5.1`: predictive performance, uncertainty intervals, precision-recall curves, and architecture ablation.
- `5.2`: concept drift, missing telemetry, timestamp perturbation, adversarial attribute perturbation, and calibration.
- `5.3`: analyst workload, incident recovery, clustering purity, latency, memory, and topological explanation.
- `experiment_summary.json`: machine-readable configuration and headline results.
- `test_predictions.csv`: calibrated test probabilities for every compared model.

## Reproducibility safeguards

The simulation seed is {config.random_seed}; model seeds are {', '.join(str(seed) for seed in training_seeds)}. Standardization is fit only on the training split. Operating thresholds and calibration temperatures are selected only on the validation split. Test labels do not influence training, calibration, or threshold selection.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    print("[7/7] Experiment completed successfully.")
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the topology-guided synthetic SOC experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("topological_soc_results"),
        help="Directory in which all result artifacts will be written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum epochs for each neural model initialization.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[13, 29],
        help="Independent neural-network initialization seeds.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced dataset and one seed for a rapid diagnostic run.",
    )
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Install any missing third-party dependency with pip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.epochs < 1:
        raise SystemExit("--epochs must be a positive integer")
    if not args.seeds:
        raise SystemExit("At least one training seed is required")

    config = ExperimentConfig(epochs=args.epochs)
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if args.quick:
        config = ExperimentConfig(
            n_train=360,
            n_validation=120,
            n_test=160,
            epochs=min(args.epochs, 8),
            bootstrap_repetitions=120,
            early_stopping_patience=3,
        )
        seeds = [seeds[0]]

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        run_experiment(output_root, config, seeds)
    except KeyboardInterrupt as exc:
        raise SystemExit("Experiment interrupted by the user.") from exc
    except Exception as exc:
        raise RuntimeError(
            "The experiment did not finish. Inspect the exception below; no "
            "scientific result should be interpreted from an incomplete run."
        ) from exc


if __name__ == "__main__":
    main()
