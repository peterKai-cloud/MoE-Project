"""Models and reproducible training for the four-scene heterogeneous MoE."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # fail with an actionable cloud-environment message
    raise ImportError(
        "mamba_ssm is required. Install a CUDA/PyTorch-compatible mamba-ssm "
        "build before training or loading the Mamba/MoE models."
    ) from exc

from experiment_utils import (
    PHYSICAL_PARAMETER_BOUNDS,
    append_parameter_snapshot,
    iter_physical_experts,
    project_physical_parameters,
    seed_everything,
    serializable_stats,
    unwrap_model,
    write_json,
)


DT = 0.1
MIN_ACCELERATION = -4.0
MAX_ACCELERATION = 3.0
MAX_JERK = 5.0

PRIVATE_EXPERT_NAMES = [
    "Mamba",
    "Gipps",
    "GM",
    "FVD",
    "Wiedemann",
    "NETSIM",
    "Mod_NETSIM",
    "Fritzsche",
]
ALL_EXPERT_NAMES = PRIVATE_EXPERT_NAMES + ["Transformer_shared", "IDM_shared"]

MOE_MODEL_NAMES = (
    "MoE",
    "MoE_NoScenePrior",
    "MoE_UniformExperts",
    "MoE_OpenLoop",
)


def is_moe_model_name(model_name: str) -> bool:
    return model_name in MOE_MODEL_NAMES


def gm_expert_core(v_ego, v_lead, gap, C, m, l):
    gap = torch.clamp(gap, min=1.5)
    v_ego_safe = torch.clamp(v_ego, min=0.1)
    delta_v = v_lead - v_ego
    acceleration = C * (v_ego_safe**m) / ((gap + 1e-2) ** l) * delta_v
    return torch.clamp(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)


def gipps_expert_core(v_ego, v_lead, gap, a_n, b_n, b_hat, tau, v_des):
    gap = torch.clamp(gap, min=0.5)
    v_ego_safe = torch.clamp(v_ego, min=0.0)
    v_lead_safe = torch.clamp(v_lead, min=0.0)
    velocity_ratio = v_ego_safe / v_des
    free_velocity = v_ego_safe + 2.5 * a_n * tau * (1.0 - velocity_ratio) * torch.sqrt(
        torch.clamp(0.025 + velocity_ratio, min=1e-5)
    )
    inner = 2.0 * gap - v_ego_safe * tau + (v_lead_safe**2) / b_hat
    safe_term = (b_n**2) * (tau**2) + b_n * inner
    safe_velocity = -b_n * tau + torch.sqrt(torch.clamp(safe_term, min=1e-5))
    next_velocity = torch.minimum(free_velocity, safe_velocity)
    acceleration = (next_velocity - v_ego_safe) / tau
    return torch.clamp(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)


def idm_expert_core(
    v_ego, v_lead, gap, a_max, v_des, beta, s_jam, T_headway, a_comf
):
    gap = torch.clamp(gap, min=1.0)
    v_ego_safe = torch.clamp(v_ego, min=0.0)
    delta_v = v_lead - v_ego_safe
    desired_gap_term = v_ego_safe * T_headway - (
        v_ego_safe * delta_v
    ) / (2.0 * torch.sqrt(a_max * a_comf))
    desired_gap = s_jam + F.relu(desired_gap_term)
    velocity_ratio = torch.clamp(v_ego_safe / v_des, min=0.0)
    acceleration = a_max * (
        1.0 - velocity_ratio**beta - (desired_gap / gap) ** 2
    )
    return torch.clamp(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)


def fvd_expert_core(v_ego, v_lead, gap, alpha, lambda_0, s_c, v0, b, beta):
    gap = torch.clamp(gap, min=1.0)
    optimal_velocity = (v0 / 2.0) * (
        torch.tanh(gap / b - beta) - torch.tanh(-beta)
    )
    interaction = torch.where(gap <= s_c, lambda_0, torch.zeros_like(lambda_0))
    acceleration = alpha * (optimal_velocity - v_ego) + interaction * (
        v_lead - v_ego
    )
    return torch.clamp(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)


def wiedemann_expert_core(v_ego, v_lead, gap, th_dv, th_gap):
    gap = torch.clamp(gap, min=1.0)
    delta_v = v_lead - v_ego
    is_close = torch.sigmoid(th_gap - gap)
    is_closing = torch.sigmoid(-delta_v - th_dv)
    acceleration = -3.5 * is_close - 1.5 * is_closing
    acceleration += (
        0.8
        * (1.0 - is_close)
        * (1.0 - is_closing)
        * torch.sigmoid(28.0 - v_ego)
    )
    return torch.clamp(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)


def netsim_expert_core(
    v_ego, v_lead, gap, T, a_max_accel, a_min_brake, b_max_brake
):
    """Safety expert named NETSIM to match the manuscript terminology."""
    v_ego_safe = torch.clamp(v_ego, min=0.0)
    v_lead_safe = torch.clamp(v_lead, min=0.0)
    term1 = v_ego_safe * T
    term2 = 0.5 * a_max_accel * (T**2)
    term3 = ((v_ego_safe + a_max_accel * T) ** 2) / (2.0 * a_min_brake)
    term4 = (v_lead_safe**2) / (2.0 * b_max_brake)
    minimum_distance = F.relu(term1 + term2 + term3 - term4)
    danger_probability = torch.sigmoid(minimum_distance - gap)
    acceleration = -a_min_brake * danger_probability
    return torch.clamp(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)


def mod_netsim_expert_core(
    v_ego, v_lead, gap, T, a_min_brake, a_max_brake, b_max_brake, v_max
):
    """Modified NETSIM safety expert using speed-dependent braking."""
    v_ego_safe = torch.clamp(v_ego, min=0.0)
    v_lead_safe = torch.clamp(v_lead, min=0.0)
    braking = a_min_brake + (v_ego_safe / v_max) * (a_max_brake - a_min_brake)
    minimum_distance = F.relu(
        v_ego_safe * T
        + (v_ego_safe**2) / (2.0 * braking)
        - (v_lead_safe**2) / (2.0 * b_max_brake)
    )
    danger_probability = torch.sigmoid(minimum_distance - gap)
    acceleration = -braking * danger_probability
    return torch.clamp(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)


def fritzsche_expert_core(
    v_ego, v_lead, gap, sdv, cldv, d_safe, a_emerg, a_clos_k, v_des
):
    relative_velocity = v_ego - v_lead
    effective_gap = torch.clamp(gap - d_safe, min=0.5)
    time_to_collision = effective_gap / torch.clamp(relative_velocity, min=0.1)
    danger = torch.sigmoid(3.5 - time_to_collision) * torch.sigmoid(
        relative_velocity - 0.5
    )
    closing = torch.sigmoid(relative_velocity - cldv) * (1.0 - danger)
    free = torch.clamp(1.0 - danger - closing, min=0.0)
    acceleration = (
        danger * (-a_emerg)
        + closing * (-a_clos_k * relative_velocity**2 / effective_gap)
        + free * (0.4 * (v_des - v_ego))
    )
    return torch.clamp(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)


class ExpertBase(nn.Module):
    def forward(self, v_ego, v_lead, gap, state_vector_phys=None):
        raise NotImplementedError


class GM_Expert(ExpertBase):
    def __init__(self):
        super().__init__()
        self.C = nn.Parameter(torch.tensor(1.4313))
        self.m = nn.Parameter(torch.tensor(0.0997))
        self.l = nn.Parameter(torch.tensor(0.7506))

    def forward(self, v_ego, v_lead, gap, state_vector_phys=None):
        return gm_expert_core(v_ego, v_lead, gap, self.C, self.m, self.l)


class Gipps_Expert(ExpertBase):
    def __init__(self):
        super().__init__()
        self.a_n = nn.Parameter(torch.tensor(0.298804))
        self.b_n = nn.Parameter(torch.tensor(0.186928))
        self.b_hat = nn.Parameter(torch.tensor(0.15516))
        self.tau = nn.Parameter(torch.tensor(3.86953))
        self.v_des = nn.Parameter(torch.tensor(27.38153))

    def forward(self, v_ego, v_lead, gap, state_vector_phys=None):
        return gipps_expert_core(
            v_ego,
            v_lead,
            gap,
            self.a_n,
            self.b_n,
            self.b_hat,
            self.tau,
            self.v_des,
        )


class IDM_Expert(ExpertBase):
    def __init__(self):
        super().__init__()
        self.a_max = nn.Parameter(torch.tensor(0.797336))
        self.v_des = nn.Parameter(torch.tensor(40.0))
        self.beta = nn.Parameter(torch.tensor(1.488607))
        self.s_jam = nn.Parameter(torch.tensor(1.369863))
        self.T_headway = nn.Parameter(torch.tensor(0.934492))
        self.a_comf = nn.Parameter(torch.tensor(0.623689))

    def forward(self, v_ego, v_lead, gap, state_vector_phys=None):
        return idm_expert_core(
            v_ego,
            v_lead,
            gap,
            self.a_max,
            self.v_des,
            self.beta,
            self.s_jam,
            self.T_headway,
            self.a_comf,
        )


class FVD_Expert(ExpertBase):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.009))
        self.lambda_0 = nn.Parameter(torch.tensor(0.2324))
        self.s_c = nn.Parameter(torch.tensor(42.3362))
        self.v0 = nn.Parameter(torch.tensor(40.0))
        self.b = nn.Parameter(torch.tensor(17.6909))
        self.beta = nn.Parameter(torch.tensor(-0.5158))

    def forward(self, v_ego, v_lead, gap, state_vector_phys=None):
        return fvd_expert_core(
            v_ego,
            v_lead,
            gap,
            self.alpha,
            self.lambda_0,
            self.s_c,
            self.v0,
            self.b,
            self.beta,
        )


class Wiedemann_Expert(ExpertBase):
    def __init__(self):
        super().__init__()
        self.th_dv = nn.Parameter(torch.tensor(1.000432))
        self.th_gap = nn.Parameter(torch.tensor(0.242639))

    def forward(self, v_ego, v_lead, gap, state_vector_phys=None):
        return wiedemann_expert_core(v_ego, v_lead, gap, self.th_dv, self.th_gap)


class NETSIM_Expert(ExpertBase):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(0.281312))
        self.a_max_accel = nn.Parameter(torch.tensor(0.932616))
        self.a_min_brake = nn.Parameter(torch.tensor(2.256824))
        self.b_max_brake = nn.Parameter(torch.tensor(1.940827))

    def forward(self, v_ego, v_lead, gap, state_vector_phys=None):
        return netsim_expert_core(
            v_ego,
            v_lead,
            gap,
            self.T,
            self.a_max_accel,
            self.a_min_brake,
            self.b_max_brake,
        )


class Mod_NETSIM_Expert(ExpertBase):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(1.29452765))
        self.a_min_brake = nn.Parameter(torch.tensor(0.721253633))
        self.a_max_brake = nn.Parameter(torch.tensor(0.779269636))
        self.b_max_brake = nn.Parameter(torch.tensor(0.565747857))
        self.v_max = nn.Parameter(torch.tensor(33.40899277))

    def forward(self, v_ego, v_lead, gap, state_vector_phys=None):
        return mod_netsim_expert_core(
            v_ego,
            v_lead,
            gap,
            self.T,
            self.a_min_brake,
            self.a_max_brake,
            self.b_max_brake,
            self.v_max,
        )


class Fritzsche_Expert(ExpertBase):
    def __init__(self):
        super().__init__()
        self.sdv = nn.Parameter(torch.tensor(55.0))
        self.cldv = nn.Parameter(torch.tensor(0.5))
        self.d_safe = nn.Parameter(torch.tensor(12.99444866))
        self.a_emerg = nn.Parameter(torch.tensor(0.874132514))
        self.a_clos_k = nn.Parameter(torch.tensor(0.506190896))
        self.v_des = nn.Parameter(torch.tensor(11.66852283))

    def forward(self, v_ego, v_lead, gap, state_vector_phys=None):
        return fritzsche_expert_core(
            v_ego,
            v_lead,
            gap,
            self.sdv,
            self.cldv,
            self.d_safe,
            self.a_emerg,
            self.a_clos_k,
            self.v_des,
        )


class NeuralExpert(ExpertBase):
    def __init__(self, input_dim=4, hidden_dim=128, mean=None, std=None):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32) if mean is not None else torch.zeros(input_dim)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32) if std is not None else torch.ones(input_dim)
        )
        self.hidden_dim = hidden_dim

    def normalize(self, physical_state):
        return (physical_state - self.mean) / (self.std + 1e-6)


class PositionalEncoding(nn.Module):
    def __init__(self, dimension, max_len=100):
        super().__init__()
        encoding = torch.zeros(max_len, dimension)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, inputs):
        return inputs + self.encoding[:, : inputs.size(1), :]


class ARTEMIS_Transformer_Expert(NeuralExpert):
    def __init__(
        self, input_dim=4, hidden_dim=128, nhead=8, num_layers=3, mean=None, std=None
    ):
        super().__init__(input_dim, hidden_dim, mean, std)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.ego_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, v_ego, v_lead, gap, state_vector_phys):
        normalized = self.normalize(state_vector_phys)
        sequence = self.positional_encoding(self.input_projection(normalized))
        sequence_feature = self.transformer(sequence)[:, -1, :]
        ego_feature = self.ego_mlp(normalized[:, -1, :])
        raw = self.output_head(torch.cat([sequence_feature, ego_feature], dim=1)).squeeze(1)
        return 3.0 * torch.tanh(raw / 4.0) - 0.5


class Mamba_Expert(NeuralExpert):
    def __init__(
        self, input_dim=4, hidden_dim=128, num_layers=2, mean=None, std=None
    ):
        super().__init__(input_dim, hidden_dim, mean, std)
        self.input_projection = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
        )
        self.short_branch = Mamba(d_model=hidden_dim, expand=2, d_conv=3, d_state=16)
        self.mid_branch = Mamba(d_model=hidden_dim, expand=2, d_conv=3, d_state=16)
        self.long_branch = Mamba(d_model=hidden_dim, expand=2, d_conv=3, d_state=16)
        self.fusion_attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, v_ego, v_lead, gap, state_vector_phys):
        normalized = torch.clamp(self.normalize(state_vector_phys), -15.0, 15.0)
        projected = self.input_projection(normalized)
        length = projected.size(1)
        short = self.short_branch(projected[:, max(0, length - 5) :, :])[:, -1, :]
        middle = self.mid_branch(projected[:, max(0, length - 15) :, :])[:, -1, :]
        long = self.long_branch(projected)[:, -1, :]
        weights = F.softmax(
            self.fusion_attention(torch.cat([short, middle, long], dim=-1)), dim=-1
        )
        fused = short * weights[:, 0:1] + middle * weights[:, 1:2] + long * weights[:, 2:3]
        raw = self.output_head(fused).squeeze(1)
        return 3.0 * torch.tanh(raw / 4.0) - 0.5


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequence):
        weights = F.softmax(self.attention(sequence), dim=1)
        return torch.sum(sequence * weights, dim=1), weights


class CNNLSTMEncoder(nn.Module):
    def __init__(self, input_dim=4, cnn_out_channels=32, lstm_hidden=128, lstm_layers=3):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, cnn_out_channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )
        self.output_dim = lstm_hidden * 2
        self.temporal_attention = TemporalAttention(self.output_dim)

    def forward(self, inputs):
        convolution = F.relu(self.cnn(inputs.transpose(1, 2))).transpose(1, 2)
        self.lstm.flatten_parameters()
        sequence, _ = self.lstm(convolution.float())
        context, _ = self.temporal_attention(sequence)
        return torch.nan_to_num(context, nan=0.0)


class ScenePerceptionModule(nn.Module):
    """Pyramidal 9 -> 512 -> 256 -> 128 -> 64 -> 4 scene MLP."""

    def __init__(self, phys_dim=9, hidden=128, num_scenes=4, dropout=0.10):
        super().__init__()
        if num_scenes != 4:
            raise ValueError(f"The manuscript defines four scenes, got {num_scenes}.")
        wide, middle, narrow, bottleneck = hidden * 4, hidden * 2, hidden, hidden // 2
        self.encoder = nn.Sequential(
            spectral_norm(nn.Linear(phys_dim, wide)),
            nn.LayerNorm(wide),
            nn.SiLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(wide, middle)),
            nn.LayerNorm(middle),
            nn.SiLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(middle, narrow)),
            nn.LayerNorm(narrow),
            nn.SiLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(narrow, bottleneck)),
            nn.LayerNorm(bottleneck),
            nn.SiLU(),
            spectral_norm(nn.Linear(bottleneck, num_scenes)),
        )

    def forward(self, physical_features):
        logits = torch.nan_to_num(
            self.encoder(physical_features), nan=0.0, posinf=20.0, neginf=-20.0
        )
        return F.softmax(logits, dim=-1)


class ImprovedPrivateMoERouter(nn.Module):
    def __init__(
        self,
        gating_input_dim,
        num_private_experts=8,
        hidden=512,
        nhead=4,
        num_scenes=4,
    ):
        super().__init__()
        self.num_scenes = int(num_scenes)
        self.input_projection = spectral_norm(
            nn.Linear(gating_input_dim + self.num_scenes, hidden)
        )
        self.temporal_gating = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.router_mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.15),
            spectral_norm(nn.Linear(hidden, hidden // 2)),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden // 2, num_private_experts)),
        )
        if self.num_scenes > 0:
            self.scene_expert_prior = nn.Parameter(
                torch.zeros(self.num_scenes, num_private_experts)
            )
        else:
            self.register_parameter("scene_expert_prior", None)
        self.temperature_parameter = nn.Parameter(torch.tensor(1.0))

    def forward(self, gating_input, scene_probabilities, history_context, training=True):
        if self.num_scenes > 0:
            if scene_probabilities is None:
                raise ValueError("Scene probabilities are required by this router.")
            router_input = torch.cat(
                [gating_input, scene_probabilities], dim=-1
            )
        else:
            router_input = gating_input
        hidden = self.input_projection(router_input)
        hidden = hidden + self.temporal_gating(history_context).mean(dim=1)
        logits = self.router_mlp(hidden)
        if self.num_scenes > 0:
            logits = logits + torch.matmul(
                scene_probabilities, self.scene_expert_prior
            )
        if training:
            logits = logits + torch.randn_like(logits) * 0.15
            temperature = torch.clamp(torch.abs(self.temperature_parameter), max=100.0)
            logits = logits / (1.0 + 0.001 * temperature)
        return torch.nan_to_num(logits, nan=-1e4)


class SelfAttention_MoE_Net(nn.Module):
    def __init__(
        self,
        input_dim=4,
        history_len=20,
        future_len=1,
        num_experts=10,
        mean=None,
        std=None,
        top_k=2,
        lstm_hidden_dim=128,
        lstm_layers=3,
        cnn_out_channels=32,
        router_hidden_dim=512,
        num_scenes=4,
        scene_hidden_dim=128,
        use_scene_prior=True,
        aggregation_mode="top_k",
    ):
        super().__init__()
        if num_scenes != 4:
            raise ValueError("Final manuscript experiments require num_scenes=4.")
        self.register_buffer("mean_buf", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std_buf", torch.tensor(std, dtype=torch.float32))
        self.top_k = int(top_k)
        self.num_scenes = 4
        self.num_physical_features = 9
        self.use_scene_prior = bool(use_scene_prior)
        self.aggregation_mode = str(aggregation_mode)
        if self.aggregation_mode not in {"top_k", "uniform_10"}:
            raise ValueError(
                f"Unsupported aggregation mode: {self.aggregation_mode}"
            )
        if not 1 <= self.top_k <= 8:
            raise ValueError(f"top_k must be between 1 and 8, got {self.top_k}.")

        if self.aggregation_mode == "top_k":
            self.encoder = CNNLSTMEncoder(
                input_dim=input_dim,
                cnn_out_channels=cnn_out_channels,
                lstm_hidden=lstm_hidden_dim,
                lstm_layers=lstm_layers,
            )
            gating_input_dim = (
                self.encoder.output_dim
                + input_dim
                + self.num_physical_features
            )
            self.scene_perception = (
                ScenePerceptionModule(
                    phys_dim=self.num_physical_features,
                    hidden=scene_hidden_dim,
                    num_scenes=4,
                )
                if self.use_scene_prior
                else None
            )
            self.router = ImprovedPrivateMoERouter(
                gating_input_dim=gating_input_dim,
                num_private_experts=8,
                hidden=router_hidden_dim,
                nhead=4,
                num_scenes=4 if self.use_scene_prior else 0,
            )
            self.history_projection = nn.Linear(input_dim, router_hidden_dim)
        else:
            # The uniform ablation removes the complete gating path.  Every
            # one of the ten expert outputs receives the fixed weight 0.1.
            self.encoder = None
            self.scene_perception = None
            self.router = None
            self.history_projection = None

        self.shared_transformer_scale = nn.Parameter(torch.tensor(1.0))
        self.shared_idm_scale = nn.Parameter(torch.tensor(1.0))
        self.private_scale = nn.Parameter(torch.tensor(1.0))

        self.shared_transformer = ARTEMIS_Transformer_Expert(
            input_dim=4, hidden_dim=128, nhead=8, num_layers=3, mean=mean, std=std
        )
        self.shared_idm = IDM_Expert()
        self.private_experts = nn.ModuleList(
            [
                Mamba_Expert(input_dim=4, hidden_dim=128, num_layers=2, mean=mean, std=std),
                Gipps_Expert(),
                GM_Expert(),
                FVD_Expert(),
                Wiedemann_Expert(),
                NETSIM_Expert(),
                Mod_NETSIM_Expert(),
                Fritzsche_Expert(),
            ]
        )

    @staticmethod
    def _physical_features(physical_history):
        sequence_length = physical_history.size(1)
        speed = physical_history[:, :, 1]
        acceleration = (speed[:, 1:] - speed[:, :-1]) / DT
        average_speed = speed.mean(dim=1)
        average_acceleration = acceleration.mean(dim=1)
        speed_std = torch.sqrt(torch.var(speed, dim=1, unbiased=False) + 1e-5)
        acceleration_std = torch.sqrt(
            torch.var(acceleration, dim=1, unbiased=False) + 1e-5
        )
        if sequence_length > 2:
            jerk = (acceleration[:, 1:] - acceleration[:, :-1]) / DT
            average_jerk = torch.abs(jerk).mean(dim=1)
            jerk_std = torch.sqrt(torch.var(jerk, dim=1, unbiased=False) + 1e-5)
        else:
            average_jerk = torch.zeros_like(average_speed)
            jerk_std = torch.zeros_like(average_speed)

        last = physical_history[:, -1, :]
        ego_speed = torch.clamp(last[:, 1], min=0.1)
        gap = torch.clamp(last[:, 0], min=0.1)
        gap_sequence = torch.clamp(physical_history[:, :, 0], min=0.1)
        time_headway = torch.clamp(gap / ego_speed, max=10.0)
        inverse_ttc = F.relu(-last[:, 2]) / gap
        inverse_ttc_sequence = F.relu(-physical_history[:, :, 2]) / gap_sequence
        inverse_ttc_change = inverse_ttc_sequence[:, -1] - inverse_ttc_sequence[:, 0]

        features = torch.stack(
            [
                average_speed / 30.0,
                speed_std / 5.0,
                average_acceleration / 3.0,
                inverse_ttc,
                average_jerk / 5.0,
                time_headway / 2.0,
                inverse_ttc_change / 0.5,
                jerk_std / 5.0,
                acceleration_std / 3.0,
            ],
            dim=1,
        )
        return torch.clamp(
            torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0),
            -15.0,
            15.0,
        )

    def forward(self, inputs, return_diagnostics=False):
        inputs = torch.nan_to_num(inputs, nan=0.0)
        physical_history = inputs * self.std_buf + self.mean_buf
        last_physical = physical_history[:, -1, :]
        last_normalized = inputs[:, -1, :]
        gap = last_physical[:, 0]
        ego_speed = last_physical[:, 1]
        lead_speed = last_physical[:, 3]

        transformer_output = self.shared_transformer(
            ego_speed, lead_speed, gap, physical_history
        )
        idm_output = self.shared_idm(ego_speed, lead_speed, gap, last_physical)

        private_outputs = []
        for expert in self.private_experts:
            state = physical_history if isinstance(expert, Mamba_Expert) else last_physical
            private_outputs.append(expert(ego_speed, lead_speed, gap, state))
        private_outputs = torch.nan_to_num(torch.stack(private_outputs, dim=1), nan=0.0)

        if self.aggregation_mode == "uniform_10":
            batch_size = inputs.shape[0]
            private_weights = torch.full(
                (batch_size, 8),
                0.1,
                dtype=private_outputs.dtype,
                device=private_outputs.device,
            )
            router_probabilities = None
            scene_probabilities = None
            expert_outputs = torch.cat(
                [
                    private_outputs,
                    transformer_output.unsqueeze(1),
                    idm_output.unsqueeze(1),
                ],
                dim=1,
            )
            components = expert_outputs * 0.1
        else:
            physical_features = self._physical_features(physical_history)
            scene_probabilities = (
                self.scene_perception(physical_features)
                if self.scene_perception is not None
                else None
            )
            context = self.encoder(inputs.float())
            gating_input = torch.cat(
                [context, last_normalized, physical_features], dim=1
            )
            history_context = self.history_projection(inputs.float())
            logits = self.router(
                gating_input,
                scene_probabilities,
                history_context,
                training=self.training,
            )
            # Keep sparse routing probabilities in FP32. Under CUDA autocast,
            # softmax may promote its output while zeros_like(logits) remains
            # FP16, which makes scatter_ fail because source and destination
            # dtypes differ. FP32 routing is also more stable for top-k gating.
            routing_logits = logits.float()
            router_probabilities = F.softmax(routing_logits, dim=1)
            top_values, top_indices = torch.topk(
                routing_logits, k=self.top_k, dim=1
            )
            top_weights = F.softmax(top_values, dim=1)
            hard_weights = torch.zeros_like(router_probabilities).scatter_(
                1, top_indices, top_weights
            )
            if self.training:
                private_weights = (
                    hard_weights.detach()
                    - router_probabilities.detach()
                    + router_probabilities
                )
            else:
                private_weights = hard_weights

            private_components = (
                self.private_scale * private_outputs * private_weights
            )
            transformer_component = (
                self.shared_transformer_scale * transformer_output
            ).unsqueeze(1)
            idm_component = (
                self.shared_idm_scale * idm_output
            ).unsqueeze(1)
            components = torch.cat(
                [private_components, transformer_component, idm_component],
                dim=1,
            )
            expert_outputs = torch.cat(
                [
                    private_outputs,
                    transformer_output.unsqueeze(1),
                    idm_output.unsqueeze(1),
                ],
                dim=1,
            )
        final_output = torch.clamp(
            torch.nan_to_num(components.sum(dim=1), nan=0.0), -10.0, 10.0
        )
        contribution_share = torch.abs(components)
        contribution_share = contribution_share / (
            contribution_share.sum(dim=1, keepdim=True) + 1e-8
        )

        diagnostics = {
            "private_router_weights": private_weights,
            "router_probabilities": router_probabilities,
            "private_outputs": private_outputs,
            "scene_probabilities": scene_probabilities,
            "contribution_share": contribution_share,
            "expert_outputs": expert_outputs,
        }
        if return_diagnostics:
            return final_output, diagnostics
        if self.training:
            return (
                final_output,
                contribution_share,
                router_probabilities,
                private_outputs,
                scene_probabilities,
            )
        return final_output, contribution_share


PHYSICAL_EXPERT_FACTORIES = {
    "GM": GM_Expert,
    "Gipps": Gipps_Expert,
    "IDM": IDM_Expert,
    "FVD": FVD_Expert,
    "Wiedemann": Wiedemann_Expert,
    "NETSIM": NETSIM_Expert,
    "Mod_NETSIM": Mod_NETSIM_Expert,
    "Fritzsche": Fritzsche_Expert,
}


class StandaloneExpertModel(nn.Module):
    """Independently trained baseline; never reuses a jointly trained MoE expert."""

    def __init__(self, model_name, mean, std):
        super().__init__()
        self.model_name = model_name
        self.register_buffer("mean_buf", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std_buf", torch.tensor(std, dtype=torch.float32))
        if model_name == "Mamba":
            self.expert = Mamba_Expert(mean=mean, std=std)
        elif model_name == "Transformer":
            self.expert = ARTEMIS_Transformer_Expert(mean=mean, std=std)
        elif model_name in PHYSICAL_EXPERT_FACTORIES:
            self.expert = PHYSICAL_EXPERT_FACTORIES[model_name]()
        else:
            raise ValueError(f"Unsupported standalone model: {model_name}")

    def forward(self, inputs, return_diagnostics=False):
        physical_history = inputs * self.std_buf + self.mean_buf
        last = physical_history[:, -1, :]
        gap, ego_speed, lead_speed = last[:, 0], last[:, 1], last[:, 3]
        full_state_models = (Mamba_Expert, ARTEMIS_Transformer_Expert)
        state = physical_history if isinstance(self.expert, full_state_models) else last
        prediction = self.expert(ego_speed, lead_speed, gap, state)
        if return_diagnostics:
            return prediction, {}
        return prediction


def model_config_from_args(args: dict, model_name: str = "MoE") -> dict:
    config = {
        "history_len": int(args.get("history_len", 20)),
        "future_len": 1,
        "top_k": int(args.get("top_k", 2)),
        "lstm_hidden_dim": int(args.get("lstm_hidden_dim", 128)),
        "lstm_layers": int(args.get("lstm_layers", 3)),
        "cnn_out_channels": int(args.get("cnn_out_channels", 32)),
        "router_hidden_dim": int(args.get("router_hidden_dim", 512)),
        "num_scenes": 4,
        "scene_hidden_dim": int(args.get("scene_hidden_dim", 128)),
        "use_scene_prior": model_name != "MoE_NoScenePrior",
        "aggregation_mode": (
            "uniform_10"
            if model_name == "MoE_UniformExperts"
            else "top_k"
        ),
    }
    return config


def build_model(model_name: str, stats: dict, model_config: Optional[dict] = None):
    model_config = dict(model_config or {})
    mean, std = stats["mean"], stats["std"]
    if is_moe_model_name(model_name):
        return SelfAttention_MoE_Net(mean=mean, std=std, **model_config)
    return StandaloneExpertModel(model_name, mean, std)


def build_optimizer(model, model_name: str, args: dict):
    base_learning_rate = float(args.get("lr", 1e-3))
    if not is_moe_model_name(model_name):
        learning_rate = float(args.get("standalone_lr", 1e-4))
        if model_name in PHYSICAL_EXPERT_FACTORIES:
            learning_rate = float(args.get("physical_baseline_lr", base_learning_rate))
        return optim.Adam(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
        )

    base = unwrap_model(model)
    physical_modules = [module for _, module in iter_physical_experts(base)]
    physical_ids = {id(parameter) for module in physical_modules for parameter in module.parameters()}
    gating_modules = [
        base.encoder,
        base.router,
        base.history_projection,
        base.scene_perception,
    ]
    gating_modules = [module for module in gating_modules if module is not None]
    gating_parameters = []
    for module in gating_modules:
        gating_parameters.extend(list(module.parameters()))
    gating_ids = {id(parameter) for parameter in gating_parameters}
    neural_parameters = [
        parameter
        for parameter in base.parameters()
        if parameter.requires_grad
        and id(parameter) not in physical_ids
        and id(parameter) not in gating_ids
    ]
    groups = []
    if gating_parameters:
        groups.append(
            {"params": gating_parameters, "lr": base_learning_rate * 0.01}
        )
    if neural_parameters:
        groups.append(
            {"params": neural_parameters, "lr": base_learning_rate * 0.01}
        )
    multipliers = {
        "Gipps_Expert": 1.0,
        "GM_Expert": 1.0,
        "FVD_Expert": 0.1,
        "Wiedemann_Expert": 1.0,
        "NETSIM_Expert": 0.1,
        "Mod_NETSIM_Expert": 1.0,
        "Fritzsche_Expert": 1.0,
        "IDM_Expert": 0.01,
    }
    for module in physical_modules:
        groups.append(
            {
                "params": module.parameters(),
                "lr": base_learning_rate
                * multipliers.get(module.__class__.__name__, 0.5),
            }
        )
    return optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)


def compute_rollout_loss(model, history, labels, future, mean_t, std_t, args):
    rollout_steps = min(int(args.get("train_rollout_steps", 5)), labels.shape[1])
    rollout_mode = str(args.get("training_rollout_mode", "closed_loop"))
    if rollout_mode not in {"closed_loop", "teacher_forcing"}:
        raise ValueError(f"Unsupported training rollout mode: {rollout_mode}")
    last = history[:, -1, :] * std_t + mean_t
    current_gap = last[:, 0]
    current_speed = last[:, 1]
    current_relative_speed = last[:, 2]
    previous_speed = history[:, -2, 1] * std_t[1] + mean_t[1]
    current_acceleration = (current_speed - previous_speed) / DT
    current_window = history

    losses = {
        name: history.new_tensor(0.0)
        for name in (
            "acceleration",
            "speed",
            "spacing",
            "relative_speed",
            "relative_spacing",
            "smoothness",
            "router_balance",
            "router_entropy",
            "private_experts",
            "scene_occupancy",
        )
    }
    last_diagnostics = {}

    for step in range(rollout_steps):
        prediction, diagnostics = model(current_window, return_diagnostics=True)
        prediction = torch.nan_to_num(prediction.reshape(-1), nan=0.0)
        target_acceleration = labels[:, step]
        bounded_target = torch.clamp(prediction, MIN_ACCELERATION, MAX_ACCELERATION)
        maximum_change = MAX_JERK * DT
        applied_acceleration = current_acceleration + torch.clamp(
            bounded_target - current_acceleration, -maximum_change, maximum_change
        )

        losses["acceleration"] += F.mse_loss(applied_acceleration, target_acceleration)
        losses["smoothness"] += F.mse_loss(applied_acceleration, current_acceleration)

        next_speed = torch.clamp(
            current_speed + applied_acceleration * DT, min=0.0, max=55.0
        )
        next_lead_speed = future[:, step, 3]  # observed lead speed is exogenous
        next_relative_speed = next_lead_speed - next_speed
        next_gap = current_gap + DT * (
            current_relative_speed + next_relative_speed
        ) / 2.0

        true_speed = future[:, step, 1]
        true_gap = future[:, step, 0]
        losses["speed"] += F.mse_loss(next_speed, true_speed)
        losses["spacing"] += F.mse_loss(next_gap, true_gap)
        losses["relative_speed"] += torch.mean(
            ((next_speed - true_speed) / (true_speed + 1.0)) ** 2
        )
        losses["relative_spacing"] += torch.mean(
            ((next_gap - true_gap) / (true_gap + 1.0)) ** 2
        )

        if diagnostics:
            router_probabilities = diagnostics.get("router_probabilities")
            private_weights = diagnostics.get("private_router_weights")
            private_outputs = diagnostics.get("private_outputs")
            scene_probabilities = diagnostics.get("scene_probabilities")
            if router_probabilities is not None:
                losses["router_balance"] += torch.sum(
                    (router_probabilities.mean(dim=0) - 1.0 / 8.0) ** 2
                ) * 20.0
            if private_weights is not None:
                normalized_private = private_weights / (
                    private_weights.sum(dim=1, keepdim=True) + 1e-8
                )
                losses["router_entropy"] += torch.mean(
                    -torch.sum(
                        normalized_private * torch.log(normalized_private + 1e-8),
                        dim=1,
                    )
                )
            if private_outputs is not None:
                target_expanded = target_acceleration.unsqueeze(1).expand_as(
                    private_outputs
                )
                losses["private_experts"] += F.mse_loss(
                    private_outputs, target_expanded
                )
            if scene_probabilities is not None:
                minimum_usage = float(args.get("minimum_scene_usage", 0.01))
                mean_scene = scene_probabilities.mean(dim=0)
                losses["scene_occupancy"] += torch.sum(
                    F.relu(minimum_usage - mean_scene) ** 2
                )

        if rollout_mode == "teacher_forcing":
            # Keep the five supervised prediction steps, but update the next
            # input window with the observed follower and leader state.  This
            # removes prediction-state feedback without shortening the loss
            # horizon, which isolates closed-loop rollout training cleanly.
            next_frame = future[:, step, :]
            updated_gap = true_gap
            updated_speed = true_speed
            updated_relative_speed = future[:, step, 2]
            updated_acceleration = target_acceleration
        else:
            next_frame = torch.stack(
                [next_gap, next_speed, next_relative_speed, next_lead_speed],
                dim=1,
            )
            updated_gap = next_gap
            updated_speed = next_speed
            updated_relative_speed = next_relative_speed
            updated_acceleration = applied_acceleration
        next_normalized = torch.clamp((next_frame - mean_t) / std_t, -25.0, 25.0)
        current_window = torch.cat(
            [current_window[:, 1:, :], next_normalized.unsqueeze(1)], dim=1
        )
        current_gap = updated_gap
        current_speed = updated_speed
        current_relative_speed = updated_relative_speed
        current_acceleration = updated_acceleration
        last_diagnostics = diagnostics

    scale = 1.0 / rollout_steps
    for name in losses:
        losses[name] = losses[name] * scale
    total = (
        50.0 * losses["acceleration"]
        + 30.0 * losses["speed"]
        + 2.0 * losses["spacing"]
        + 10.0 * losses["relative_speed"]
        + 10.0 * losses["relative_spacing"]
        + losses["smoothness"]
        + 0.01 * losses["router_balance"]
        - 0.01 * losses["router_entropy"]
        + 0.2 * losses["private_experts"]
        + float(args.get("scene_occupancy_weight", 10.0))
        * losses["scene_occupancy"]
    )
    return total, losses, last_diagnostics


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_model(args, train_loader, val_loader, test_loader, stats):
    """Train one model/seed; select by validation RMSE, never by test metrics."""
    from evaluation import evaluate_model

    args = dict(args)
    seed = int(args.get("seed", 42))
    seed_everything(seed)
    model_name = args.get("model_name", "MoE")
    args["training_rollout_mode"] = (
        "teacher_forcing"
        if model_name == "MoE_OpenLoop"
        else "closed_loop"
    )
    dataset_name = args.get("dataset_name", "SH")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = (
        model_config_from_args(args, model_name)
        if is_moe_model_name(model_name)
        else {}
    )
    model = build_model(model_name, stats, model_config).to(device)
    project_physical_parameters(model)

    if torch.cuda.device_count() > 1 and bool(args.get("data_parallel", False)):
        model = nn.DataParallel(model)

    optimizer = build_optimizer(model, model_name, args)
    use_amp = device.type == "cuda" and bool(args.get("amp", True))
    scaler = _grad_scaler(use_amp)
    run_dir = (
        Path(args.get("run_dir", "./runs"))
        / dataset_name
        / model_name
        / f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_config.json", dict(args))

    best_path = run_dir / "best_validation.pth"
    last_path = run_dir / "last.pth"
    log_path = run_dir / "training_log.csv"
    parameter_path = run_dir / "expert_parameters_by_epoch.csv"
    if parameter_path.exists():
        parameter_path.unlink()
    best_key = (float("inf"), float("inf"))
    best_epoch = 0
    epochs = int(args.get("epochs", 200))
    validation_interval = int(args.get("val_interval", 1))
    early_stopping_patience = max(
        0, int(args.get("early_stopping_patience", 0))
    )
    epochs_completed = 0
    stopped_early = False
    mean_t = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
    std_t = torch.tensor(stats["std"], device=device, dtype=torch.float32)

    log_fields = [
        "epoch",
        "train_loss",
        "val_rmse_spacing",
        "val_mae_spacing",
        "val_mae_acc",
        "val_collision_rate",
        "val_unsafe_event_rate",
        "is_best_validation",
        "epochs_since_best",
        "duration_seconds",
    ] + [f"scene_{index + 1}_hard_fraction" for index in range(4)]
    with log_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=log_fields).writeheader()

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total_loss_value = 0.0
        valid_batches = 0
        scene_counts = torch.zeros(4, device=device)
        scene_total = 0

        for history, labels, future in train_loader:
            history = history.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            history = history + torch.randn_like(history) * float(
                args.get("input_noise_std", 0.02)
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                loss, _, diagnostics = compute_rollout_loss(
                    model, history, labels, future, mean_t, std_t, args
                )
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            project_physical_parameters(model)
            total_loss_value += float(loss.detach().cpu())
            valid_batches += 1

            scene_probabilities = diagnostics.get("scene_probabilities") if diagnostics else None
            if scene_probabilities is not None:
                hard_ids = torch.argmax(scene_probabilities.detach(), dim=1)
                scene_counts += torch.bincount(hard_ids, minlength=4).float()
                scene_total += len(hard_ids)

        if valid_batches == 0:
            raise RuntimeError("No finite training batch was completed.")

        average_train_loss = total_loss_value / valid_batches
        validation_summary = {
            "rmse_spacing": float("nan"),
            "mae_spacing": float("nan"),
            "mae_acc": float("nan"),
            "collision_rate": float("nan"),
            "unsafe_event_rate": float("nan"),
        }
        validation_ran = epoch % validation_interval == 0 or epoch == epochs
        if validation_ran:
            validation_summary, _ = evaluate_model(
                args,
                model,
                val_loader,
                mean=stats["mean"],
                std=stats["std"],
                split_name="validation",
                output_csv=None,
                show_progress=False,
            )

        checkpoint = {
            "epoch": epoch,
            "seed": seed,
            "dataset": dataset_name,
            "model_name": model_name,
            "model_config": model_config,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "stats": serializable_stats(stats),
            "validation_summary": validation_summary,
        }
        torch.save(checkpoint, last_path)

        append_parameter_snapshot(
            model,
            parameter_path,
            dataset_name,
            model_name,
            seed,
            epoch,
            is_best=False,
        )

        rmse_value = validation_summary["rmse_spacing"]
        collision_value = validation_summary["collision_rate"]
        improved_this_epoch = False
        if np.isfinite(rmse_value):
            current_key = (rmse_value, collision_value)
            if current_key < best_key:
                best_key = current_key
                best_epoch = epoch
                improved_this_epoch = True
                torch.save(checkpoint, best_path)
                append_parameter_snapshot(
                    model,
                    parameter_path,
                    dataset_name,
                    model_name,
                    seed,
                    epoch,
                    is_best=True,
                )

        scene_fraction = (
            (scene_counts / max(scene_total, 1)).detach().cpu().tolist()
        )
        row = {
            "epoch": epoch,
            "train_loss": average_train_loss,
            "val_rmse_spacing": validation_summary["rmse_spacing"],
            "val_mae_spacing": validation_summary["mae_spacing"],
            "val_mae_acc": validation_summary["mae_acc"],
            "val_collision_rate": validation_summary["collision_rate"],
            "val_unsafe_event_rate": validation_summary["unsafe_event_rate"],
            "is_best_validation": int(improved_this_epoch),
            "epochs_since_best": (
                epoch - best_epoch if best_epoch > 0 else epoch
            ),
            "duration_seconds": time.time() - start,
        }
        for index, value in enumerate(scene_fraction):
            row[f"scene_{index + 1}_hard_fraction"] = value
        with log_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=log_fields).writerow(row)
        print(
            f"[{dataset_name}/{model_name}/seed={seed}] epoch {epoch}/{epochs} "
            f"loss={average_train_loss:.4f} val_RMSE={rmse_value:.4f} "
            f"collision={collision_value:.4f} scenes={scene_fraction}"
        )
        epochs_completed = epoch
        if (
            early_stopping_patience > 0
            and validation_ran
            and best_epoch > 0
            and epoch - best_epoch >= early_stopping_patience
            and epoch < epochs
        ):
            stopped_early = True
            print(
                f"[{dataset_name}/{model_name}/seed={seed}] early stopping "
                f"at epoch {epoch}: best epoch={best_epoch}, no validation "
                f"improvement for {epoch - best_epoch} epochs "
                f"(patience={early_stopping_patience})."
            )
            break

    if not best_path.exists():
        raise RuntimeError(
            "No validation checkpoint was selected. Check that validation "
            "events contain future frames after the initialization window."
        )
    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_epoch": best_epoch,
        "best_validation_rmse": best_key[0],
        "best_validation_collision_rate": best_key[1],
        "epochs_completed": epochs_completed,
        "stopped_early": stopped_early,
        "early_stopping_patience": early_stopping_patience,
        "run_dir": str(run_dir),
    }


def load_model_from_checkpoint(checkpoint_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    stats = checkpoint["stats"]
    model = build_model(
        checkpoint.get("model_name", "MoE"),
        stats,
        checkpoint.get("model_config", {}),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, stats, checkpoint
