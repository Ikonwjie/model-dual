import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unsupported RevIN mode: {mode}")

    def _get_statistics(self, x):
        reduce_dims = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=reduce_dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=reduce_dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        return x * self.stdev + self.mean


class ZeroInitResidualAdapter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return x + self.proj(self.norm(x))


class P2TAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_modes: int = 32,
        top_k: int = 3,
        low_rank: int = 64,
        anchor_radius: int = 8,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_modes = max(1, int(num_modes))
        self.top_k = max(1, int(top_k))
        self.low_rank = max(1, int(low_rank))
        self.anchor_radius = max(1, int(anchor_radius))

        self.anchor_score_proj = nn.Linear(self.d_model, 1)
        self.translation_a = nn.Parameter(
            torch.randn(self.num_modes, self.d_model, self.low_rank) * 0.01
        )
        self.translation_b = nn.Parameter(
            torch.randn(self.num_modes, self.d_model, self.low_rank) * 0.01
        )
        self.var_gate_scale = nn.Parameter(torch.tensor(1.0))
        self.var_gate_bias = nn.Parameter(torch.tensor(1.25))
        self.output_norm = nn.LayerNorm(self.d_model)

    def _build_neighborhood_stats(self, ts_hidden, patch_positions, anchor_idx):
        batch_size = ts_hidden.size(0)
        patch_positions = patch_positions.view(1, -1).expand(batch_size, -1)
        anchor_pos = torch.gather(patch_positions, 1, anchor_idx)
        patch_positions = patch_positions.unsqueeze(1)
        distances = (patch_positions - anchor_pos.unsqueeze(-1)).abs()
        neighborhood_mask = (distances <= self.anchor_radius).float()
        spread_logits = -distances / max(float(self.anchor_radius), 1.0)
        spread_logits = spread_logits.masked_fill(neighborhood_mask == 0, -1e4)
        spread_weights = torch.softmax(spread_logits, dim=-1)
        weighted_mask = neighborhood_mask.unsqueeze(-1)

        ts_context = ts_hidden.unsqueeze(1)
        denom = weighted_mask.sum(dim=2).clamp_min(1.0)
        neighborhood_mean = (ts_context * weighted_mask).sum(dim=2) / denom
        centered = (ts_context - neighborhood_mean.unsqueeze(2)) * weighted_mask
        neighborhood_var = centered.pow(2).sum(dim=2) / denom
        return neighborhood_var.mean(dim=-1, keepdim=True), spread_weights

    def forward(self, pt_hidden, ts_hidden, patch_positions, mode_centroids):
        batch_size, seq_len, hidden_dim = pt_hidden.shape
        top_k = min(self.top_k, seq_len)

        anchor_scores = self.anchor_score_proj(pt_hidden).squeeze(-1)
        topk_scores, anchor_idx = torch.topk(anchor_scores, k=top_k, dim=1)
        anchor_vectors = torch.gather(
            pt_hidden,
            1,
            anchor_idx.unsqueeze(-1).expand(-1, -1, hidden_dim),
        )

        normalized_modes = F.normalize(mode_centroids, dim=-1, eps=1e-6)
        basis_logits = torch.matmul(
            F.normalize(anchor_vectors, dim=-1, eps=1e-6),
            normalized_modes.transpose(0, 1),
        )
        basis_weights = torch.softmax(basis_logits, dim=-1)

        dynamic_a = torch.einsum("bkm,mdr->bkdr", basis_weights, self.translation_a)
        dynamic_b = torch.einsum("bkm,mdr->bkdr", basis_weights, self.translation_b)
        anchor_low_rank = torch.einsum("bkd,bkdr->bkr", anchor_vectors, dynamic_a)
        translated_signal = torch.einsum("bkr,bkdr->bkd", anchor_low_rank, dynamic_b)

        local_variance, spread_weights = self._build_neighborhood_stats(
            ts_hidden,
            patch_positions,
            anchor_idx,
        )
        injection_gate = torch.ones_like(local_variance)
        gated_signal = translated_signal

        anchor_update = torch.einsum("bkl,bkd->bld", spread_weights, gated_signal)
        ts_hidden_aug = self.output_norm(ts_hidden + anchor_update)

        anchor_mask = torch.zeros(
            batch_size,
            seq_len,
            device=pt_hidden.device,
            dtype=torch.bool,
        )
        anchor_mask.scatter_(1, anchor_idx, True)

        stats = {
            "anchor_idx": anchor_idx,
            "anchor_mask": anchor_mask,
            "anchor_scores": topk_scores,
            "gate_mean": injection_gate.mean().detach(),
            "injection_gate": injection_gate.detach(),
            "local_variance": local_variance.detach(),
            "spread_weights": spread_weights.detach(),
            "mode_entropy": (
                -(
                    basis_weights
                    * basis_weights.clamp_min(1e-8).log()
                ).sum(dim=-1).mean()
            ).detach(),
        }
        return ts_hidden_aug, stats


class T2PAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 256,
        window_size: int = 5,
        trigger_k: int = 3,
        confidence_threshold: float = 0.7,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.window_size = max(1, int(window_size))
        self.trigger_k = max(1, int(trigger_k))
        self.confidence_threshold = float(confidence_threshold)

        self.change_proj = nn.Linear(self.d_model + 1, self.d_model)
        self.confidence_mlp = nn.Sequential(
            nn.Linear(self.d_model + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.direction_proj = nn.Linear(self.d_model, 1)
        self.update_proj = nn.Linear(self.d_model, self.d_model)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.register_buffer(
            "window_template",
            torch.ones(1, 1, self.window_size),
            persistent=False,
        )


class TrackAwareEvidenceChain(nn.Module):
    def __init__(
        self,
        d_model: int,
        init_blend: float = 0.35,
        init_bias: float = 2.2,
        init_scale: float = 6.0,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model, elementwise_affine=False)

        blend = min(max(float(init_blend), 1e-4), 1.0 - 1e-4)
        self.blend_logit = nn.Parameter(torch.logit(torch.tensor(blend)))
        self.transition_bias = nn.Parameter(torch.tensor(float(init_bias)))
        self.transition_scale = nn.Parameter(torch.tensor(float(init_scale)))

    @staticmethod
    def _safe_normalize(x, eps: float = 1e-4):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    def _build_transition_score(self, query):
        prev_query = torch.cat([query[:, :1, :], query[:, :-1, :]], dim=1)
        cosine_delta = 1.0 - (query * prev_query).sum(dim=-1, keepdim=True)
        magnitude_delta = (query - prev_query).pow(2).mean(dim=-1, keepdim=True).sqrt()
        return cosine_delta + 0.5 * magnitude_delta

    def _track_hidden(self, hidden, stay_gate):
        steps = [hidden[:, 0, :]]
        carry = hidden[:, 0, :]
        for idx in range(1, hidden.size(1)):
            carry = stay_gate[:, idx] * carry + (1.0 - stay_gate[:, idx]) * hidden[:, idx, :]
            steps.append(carry)
        return torch.stack(steps, dim=1)

    def forward(self, ts_patch_embeds):
        base_hidden = self.query_norm(ts_patch_embeds)
        base_query = self._safe_normalize(base_hidden)
        transition_score = self._build_transition_score(base_query)
        stay_gate = torch.sigmoid(
            self.transition_bias - F.softplus(self.transition_scale) * transition_score
        )
        tracked_hidden = self._track_hidden(base_hidden, stay_gate)
        blend = torch.sigmoid(self.blend_logit)
        final_query = self._safe_normalize(tracked_hidden)

        stats = {
            "transition_mean": transition_score.mean().detach(),
            "stay_gate_mean": stay_gate.mean().detach(),
            "blend": blend.detach(),
        }
        cache = {
            "stay_gate": stay_gate,
            "transition_score": transition_score,
        }
        return final_query, stats, cache


class AntiCollapseSelectiveBridge(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        bridge_scale: float = 0.24,
        agreement_center: float = 0.72,
        agreement_width: float = 0.20,
        decay_start: float = 0.35,
        decay_floor: float = 0.35,
    ):
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.agreement_center = float(agreement_center)
        self.agreement_width = max(1e-3, float(agreement_width))
        self.decay_start = float(decay_start)
        self.decay_floor = float(decay_floor)
        self.training_progress = 0.0
        self.analysis_mode = "selective"
        self.ts_norm = nn.LayerNorm(d_model)
        self.pt_norm = nn.LayerNorm(d_model)
        self.shared_proj = nn.Linear(d_model * 2 + 1, d_model)
        self.ts_update = nn.Linear(d_model, d_model, bias=False)
        self.pt_update = nn.Linear(d_model, d_model, bias=False)
        self.bridge_scale = nn.Parameter(torch.tensor(float(bridge_scale)))

        nn.init.xavier_uniform_(self.shared_proj.weight, gain=0.30)
        nn.init.zeros_(self.shared_proj.bias)
        nn.init.xavier_uniform_(self.ts_update.weight, gain=0.18)
        nn.init.xavier_uniform_(self.pt_update.weight, gain=0.18)

    def set_progress(self, progress: float):
        self.training_progress = float(progress)

    def set_analysis_mode(self, mode: str = "selective"):
        mode = str(mode).strip().lower()
        if mode not in {"selective", "always_on"}:
            raise ValueError(
                f"Unsupported analysis bridge mode={mode!r}; expected selective/always_on"
            )
        self.analysis_mode = mode

    def _curriculum_factor(self):
        return 1.0

    def _causal_pool(self, x):
        if self.kernel_size <= 1:
            return x
        xt = x.transpose(1, 2)
        xt = F.pad(xt, (self.kernel_size - 1, 0), mode="replicate")
        xt = F.avg_pool1d(xt, kernel_size=self.kernel_size, stride=1)
        return xt.transpose(1, 2).contiguous()

    def _agreement_focus(self, agreement):
        normalized = (agreement - self.agreement_center) / self.agreement_width
        band_focus = torch.exp(-0.5 * normalized.pow(2))
        return 0.20 + 0.80 * band_focus

    def forward(self, ts_hidden, pt_hidden, stability_gate):
        ts_norm = self.ts_norm(ts_hidden)
        pt_norm = self.pt_norm(pt_hidden)
        agreement = 0.5 * (
            1.0 + F.cosine_similarity(ts_norm, pt_norm, dim=-1, eps=1e-6).unsqueeze(-1)
        )
        agreement_focus = self._agreement_focus(agreement)
        curriculum_factor = ts_hidden.new_tensor(self._curriculum_factor())
        raw_bridge_gate = agreement * agreement_focus * stability_gate * curriculum_factor
        if self.analysis_mode == "always_on":
            bridge_gate = torch.ones_like(raw_bridge_gate) * curriculum_factor
        else:
            bridge_gate = raw_bridge_gate

        shared_state = self.shared_proj(torch.cat([ts_norm, pt_norm, bridge_gate], dim=-1))
        shared_state = self._causal_pool(shared_state)

        blend = torch.sigmoid(self.bridge_scale) * 0.35
        local_blend = blend * bridge_gate
        ts_hidden = ts_hidden + local_blend * self.ts_update(shared_state - ts_norm)
        pt_hidden = pt_hidden + local_blend * self.pt_update(shared_state - pt_norm)
        self._last_forward_cache = {
            "agreement": agreement,
            "agreement_focus": agreement_focus,
            "stability_gate": stability_gate,
            "raw_bridge_gate": raw_bridge_gate,
            "bridge_gate": bridge_gate,
            "local_blend": local_blend,
            "curriculum_factor": curriculum_factor,
            "analysis_mode": self.analysis_mode,
        }

        stats = {
            "agreement_mean": agreement.mean().detach(),
            "agreement_focus_mean": agreement_focus.mean().detach(),
            "stability_mean": stability_gate.mean().detach(),
            "raw_bridge_gate_mean": raw_bridge_gate.mean().detach(),
            "bridge_gate_mean": bridge_gate.mean().detach(),
            "bridge_blend": blend.detach(),
            "curriculum_factor": curriculum_factor.detach(),
            "analysis_mode": self.analysis_mode,
        }
        return ts_hidden, pt_hidden, stats


class BALM_MedualTime(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = int(getattr(config, "hidden_size", getattr(config, "n_embd", 768)))
        self.num_hidden_layers = int(
            getattr(config, "num_hidden_layers", getattr(config, "n_layer", 6))
        )
        self.pred_len = int(getattr(config, "pred_len", 96))
        self.ts_config = config.ts_config
        self.context_len = int(getattr(self.ts_config, "context_points", 96))
        self.patch_len = int(getattr(self.ts_config, "patch_len", 16))
        self.stride = int(getattr(self.ts_config, "stride", 8))
        self.output_vars = int(getattr(self.ts_config, "vars", 7))
        self.patch_nums = max(1, (self.context_len - self.patch_len) // self.stride + 1)
        self.soft_vocab_topk = int(getattr(config, "soft_vocab_topk", 8))
        self.gumbel_tau_start = float(getattr(config, "gumbel_tau", 0.7))
        self.gumbel_tau_end = float(getattr(config, "gumbel_tau_end", 0.1))
        self.gumbel_tau = self.gumbel_tau_start
        self.use_straight_through_tokens = bool(
            getattr(config, "use_straight_through_tokens", True)
        )
        self.adapter_coop_weight = float(getattr(config, "adapter_coop_weight", 0.05))
        self.num_modes = int(getattr(config, "mode_clusters", 32))
        self.max_centroid_patches = int(getattr(config, "max_centroid_patches", 200000))
        self.centroid_cache_dir = str(
            getattr(config, "centroid_cache_dir", "./checkpoints/centroids")
        )

        try:
            pretrained_gpt2 = GPT2Model.from_pretrained("gpt2", local_files_only=True)
        except Exception as exc:
            raise RuntimeError("Unable to load local pretrained GPT-2 weights for model_dual.py") from exc

        self.wte = pretrained_gpt2.wte
        self.wpe = pretrained_gpt2.wpe
        self.gpt_blocks = nn.ModuleList(
            [pretrained_gpt2.h[i] for i in range(self.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=pretrained_gpt2.config.layer_norm_epsilon)
        self.ln_f.load_state_dict(pretrained_gpt2.ln_f.state_dict())

        for param in self.wte.parameters():
            param.requires_grad = False
        for param in self.wpe.parameters():
            param.requires_grad = False
        for param in self.gpt_blocks.parameters():
            param.requires_grad = False
        for param in self.ln_f.parameters():
            param.requires_grad = False

        del pretrained_gpt2

        self.normalize = RevIN(num_features=self.output_vars, eps=1e-5, affine=True)
        self.ts_patch_layer = nn.Linear(self.patch_len, self.embed_dim)
        self.pt_patch_norm = nn.LayerNorm(self.embed_dim)
        self.input_dropout = nn.Dropout(float(getattr(config, "embd_pdrop", 0.1)))
        self.disable_ts_input_residual = bool(
            getattr(config, "disable_ts_input_residual", False)
        )
        self.disable_evidence_chain = bool(getattr(config, "disable_evidence_chain", False))
        self.disable_selective_bridge = bool(getattr(config, "disable_selective_bridge", False))
        self.disable_p2t = bool(getattr(config, "disable_p2t", False))
        if self.disable_ts_input_residual:
            self.ts_branch_adapter = nn.Identity()
        else:
            self.ts_branch_adapter = ZeroInitResidualAdapter(self.embed_dim)
        self.pt_token_bridge = ZeroInitResidualAdapter(self.embed_dim)
        self.pt_branch_adapter = ZeroInitResidualAdapter(self.embed_dim)

        initial_mode_centroids = self._initialize_mode_centroids()
        self.register_buffer(
            "mode_centroids_seed",
            initial_mode_centroids.detach().clone(),
            persistent=False,
        )
        self.mode_centroids_patch = nn.Parameter(initial_mode_centroids.detach().clone())
        centroid_hidden = max(64, min(256, self.embed_dim // 4))
        self.mode_centroid_refiner = nn.Sequential(
            nn.Linear(self.embed_dim, centroid_hidden),
            nn.GELU(),
            nn.Linear(centroid_hidden, self.embed_dim),
        )
        nn.init.zeros_(self.mode_centroid_refiner[-1].weight)
        nn.init.zeros_(self.mode_centroid_refiner[-1].bias)

        self.p2t_adapter = P2TAdapter(
            d_model=self.embed_dim,
            num_modes=self.num_modes,
            top_k=int(getattr(config, "p2t_topk", 5)),
            low_rank=int(getattr(config, "p2t_low_rank", 64)),
            anchor_radius=int(getattr(config, "p2t_anchor_radius", self.stride)),
        )
        # Keep RNG consumption aligned with the old code path where T2P was
        # instantiated, while permanently disabling T2P in the forward pass.
        _ = T2PAdapter(
            d_model=self.embed_dim,
            hidden_dim=int(getattr(config, "t2p_hidden_dim", max(128, self.embed_dim // 3))),
            window_size=int(getattr(config, "t2p_window_size", 5)),
            trigger_k=int(getattr(config, "t2p_trigger_k", 3)),
            confidence_threshold=float(getattr(config, "t2p_conf_threshold", 0.6)),
        )
        self.pattern_vocab_chunk = int(getattr(config, "pattern_vocab_chunk", 4096))
        self.register_buffer(
            "normalized_word_embeddings",
            F.normalize(self.wte.weight.detach().clone(), dim=-1, eps=1e-6),
            persistent=False,
        )

        self.final_output_branch = str(getattr(config, "final_output_branch", "dual")).lower()
        if self.final_output_branch not in {"dual", "ts", "pt"}:
            raise ValueError(
                f"Unsupported final_output_branch={self.final_output_branch!r}; expected one of dual/ts/pt"
            )
        output_branch_count = 2 if self.final_output_branch == "dual" else 1
        self.pred_head = nn.Linear(
            self.patch_nums * self.embed_dim * output_branch_count,
            self.pred_len,
        )
        self.evidence_chain = TrackAwareEvidenceChain(
            d_model=self.embed_dim,
            init_blend=float(getattr(config, "vocab_track_blend", 0.35)),
            init_bias=float(getattr(config, "vocab_track_bias", 2.2)),
            init_scale=float(getattr(config, "vocab_track_scale", 6.0)),
        )
        self.bridge_layers = self._parse_bridge_layers(
            getattr(config, "syncbridge_layers", "1,3,5")
        )
        self.selective_bridge = AntiCollapseSelectiveBridge(
            d_model=self.embed_dim,
            kernel_size=int(getattr(config, "syncbridge_kernel", 3)),
            bridge_scale=float(getattr(config, "syncbridge_scale", 0.24)),
            agreement_center=float(getattr(config, "anticollapse_agreement_center", 0.72)),
            agreement_width=float(getattr(config, "anticollapse_agreement_width", 0.20)),
            decay_start=float(getattr(config, "bridge_decay_start", 0.35)),
            decay_floor=float(getattr(config, "bridge_decay_floor", 0.35)),
        )
        self._last_evidence_stats = {}
        self._last_evidence_cache = {}
        self._analysis_enabled = False
        self._analysis_move_to_cpu = True
        self._last_analysis = {}
        self.set_train_stage("adapter_warmup", unfreeze_last_n=0)
        self.set_analysis_bridge_mode("selective")
        self.set_gumbel_tau_progress(0.0)

    def enable_analysis_cache(self, enabled: bool = True, move_to_cpu: bool = True):
        self._analysis_enabled = bool(enabled)
        self._analysis_move_to_cpu = bool(move_to_cpu)
        if not self._analysis_enabled:
            self._last_analysis = {}
        return self

    def _cache_analysis_value(self, value):
        if torch.is_tensor(value):
            cached = value.detach()
            if self._analysis_move_to_cpu:
                cached = cached.cpu()
            return cached
        if isinstance(value, dict):
            return {key: self._cache_analysis_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            cached_items = [self._cache_analysis_value(item) for item in value]
            return type(value)(cached_items) if isinstance(value, tuple) else cached_items
        return value

    def train(self, mode: bool = True):
        super().train(mode)
        self.wte.eval()
        self.wpe.eval()
        self.ln_f.eval()
        for block in self.gpt_blocks:
            should_train = mode and any(
                param.requires_grad for param in block.parameters()
            )
            if should_train:
                block.train()
            else:
                block.eval()
        return self

    def set_train_stage(
        self,
        stage_name: str,
        unfreeze_last_n: int = 2,
        unfreeze_pos_norm: bool = False,
    ):
        for name, param in self.named_parameters():
            if name.startswith(("wte.", "wpe.", "gpt_blocks.", "ln_f.")):
                param.requires_grad = False
            else:
                param.requires_grad = True

        self._unfrozen_last_n = 0
        if stage_name == "late_unfreeze":
            self._unfrozen_last_n = min(max(0, int(unfreeze_last_n)), self.num_hidden_layers)
            start_idx = self.num_hidden_layers - self._unfrozen_last_n
            for block_idx in range(start_idx, self.num_hidden_layers):
                for param in self.gpt_blocks[block_idx].parameters():
                    param.requires_grad = True
        if unfreeze_pos_norm:
            for param in self.wpe.parameters():
                param.requires_grad = True
            for block in self.gpt_blocks:
                for norm_name in ("ln_1", "ln_2"):
                    norm = getattr(block, norm_name, None)
                    if norm is None:
                        continue
                    for param in norm.parameters():
                        param.requires_grad = True
            for param in self.ln_f.parameters():
                param.requires_grad = True

        self._active_train_stage = stage_name
        self.train(self.training)
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def set_training_progress(self, progress: float):
        self.selective_bridge.set_progress(progress)

    def set_analysis_bridge_mode(self, mode: str = "selective"):
        if hasattr(self.selective_bridge, "set_analysis_mode"):
            self.selective_bridge.set_analysis_mode(mode)
        self._analysis_bridge_mode = str(mode).strip().lower()

    def set_gumbel_tau_progress(self, progress: float):
        progress = min(max(float(progress), 0.0), 1.0)
        self.gumbel_tau = self.gumbel_tau_start + (
            self.gumbel_tau_end - self.gumbel_tau_start
        ) * progress
        return self.gumbel_tau

    def _parse_bridge_layers(self, raw_layers):
        if isinstance(raw_layers, str):
            parsed = []
            for item in raw_layers.split(","):
                item = item.strip()
                if not item:
                    continue
                try:
                    parsed.append(int(item))
                except ValueError:
                    continue
        elif isinstance(raw_layers, (list, tuple)):
            parsed = [int(item) for item in raw_layers]
        else:
            parsed = []

        valid_layers = sorted({idx for idx in parsed if 0 <= idx < self.num_hidden_layers})
        if valid_layers:
            return tuple(valid_layers)
        fallback = [idx for idx in (1, 3, 5) if idx < self.num_hidden_layers]
        if not fallback:
            fallback = [self.num_hidden_layers - 1]
        return tuple(fallback)

    def _resolve_centroid_cache_path(self):
        data_name = str(getattr(self.config, "data_name", "unknown"))
        patch_len = int(self.patch_len)
        stride = int(self.stride)
        mode_clusters = int(self.num_modes)
        cache_dir = os.path.abspath(self.centroid_cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(
            cache_dir,
            f"sanplm_patch_centroids_{data_name}_pl{patch_len}_st{stride}_m{mode_clusters}_v2.pt",
        )

    def _get_train_segment(self):
        root_path = getattr(self.config, "root_path", "")
        data_path = getattr(self.config, "data_path", "")
        features = getattr(self.config, "features", "M")
        target = getattr(self.config, "target", "OT")
        percent = int(getattr(self.config, "percent", 100))
        data_name = str(getattr(self.config, "data_name", ""))
        if not root_path or not data_path:
            return None

        try:
            import numpy as np
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return None

        data_file = os.path.join(root_path, data_path)
        if not os.path.exists(data_file):
            return None

        df_raw = pd.read_csv(data_file)
        if features in ("M", "MS"):
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:
            df_data = df_raw[[target]]

        if data_name in ("ETTh1", "ETTh2"):
            train_end = 12 * 30 * 24
        elif data_name in ("ETTm1", "ETTm2"):
            train_end = 12 * 30 * 24 * 4
        else:
            train_end = int(len(df_data) * 0.7)

        if percent < 100:
            train_end = (train_end - self.context_len) * percent // 100 + self.context_len

        scaler = StandardScaler()
        scaler.fit(df_data.iloc[:train_end].values)
        data = scaler.transform(df_data.values)
        return data[:train_end]

    def _compute_mode_centroids_kmeans(self):
        train_segment = self._get_train_segment()
        if train_segment is None:
            return None

        try:
            import numpy as np
            from sklearn.cluster import MiniBatchKMeans
        except Exception:
            return None

        raw_patches = []
        seq_len, num_vars = train_segment.shape
        for var_idx in range(num_vars):
            series = train_segment[:, var_idx]
            for start in range(0, seq_len - self.patch_len + 1, self.stride):
                raw_patches.append(series[start:start + self.patch_len])

        if not raw_patches:
            return None

        patch_array = np.asarray(raw_patches, dtype="float32")
        if patch_array.shape[0] > self.max_centroid_patches:
            rng = np.random.default_rng(int(getattr(self.config, "experiment_seed", 42)))
            sampled_idx = rng.choice(
                patch_array.shape[0],
                self.max_centroid_patches,
                replace=False,
            )
            patch_array = patch_array[sampled_idx]

        kmeans = MiniBatchKMeans(
            n_clusters=self.num_modes,
            batch_size=min(4096, patch_array.shape[0]),
            n_init=10,
            random_state=int(getattr(self.config, "experiment_seed", 42)),
        )
        kmeans.fit(patch_array)

        raw_centroids = torch.tensor(
            kmeans.cluster_centers_,
            dtype=self.ts_patch_layer.weight.dtype,
        )
        return raw_centroids.detach().cpu()

    def _initialize_mode_centroids(self):
        cache_path = self._resolve_centroid_cache_path()
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            if isinstance(cached, dict):
                cached = cached.get("centroids")
            if torch.is_tensor(cached) and cached.shape == (self.num_modes, self.patch_len):
                return cached.float()

        centroids = self._compute_mode_centroids_kmeans()
        if centroids is None:
            centroids = self._build_fallback_patch_centroids()

        torch.save({"centroids": centroids.float().cpu()}, cache_path)
        return centroids.float()

    def _build_fallback_patch_centroids(self):
        grid = torch.linspace(-1.0, 1.0, steps=self.patch_len)
        patterns = []
        for idx in range(self.num_modes):
            freq = idx % 4 + 1
            if idx % 4 == 0:
                pattern = grid * (1.0 + 0.1 * (idx // 4))
            elif idx % 4 == 1:
                pattern = torch.sin(freq * torch.pi * (grid + 1.0) * 0.5)
            elif idx % 4 == 2:
                pattern = torch.cos(freq * torch.pi * (grid + 1.0) * 0.5)
            else:
                pattern = torch.tanh(2.0 * grid) * torch.sin(
                    freq * torch.pi * (grid + 1.0) * 0.5
                )
            patterns.append(pattern)
        return torch.stack(patterns, dim=0).float()

    def _get_refined_mode_centroids(self):
        projected = self.ts_patch_layer(self.mode_centroids_patch)
        projected = F.layer_norm(projected, (self.embed_dim,))
        refined = projected + 0.1 * self.mode_centroid_refiner(projected)
        return F.layer_norm(refined, (self.embed_dim,))

    def _extract_patches(self, ts_normed):
        batch_size, seq_len, _ = ts_normed.shape
        num_patch = max(1, (seq_len - self.patch_len) // self.stride + 1)
        patches = []
        positions = []
        for idx in range(num_patch):
            start = idx * self.stride
            end = start + self.patch_len
            patch = ts_normed[:, start:end, :]
            patches.append(patch.squeeze(-1))
            positions.append(start)
        patch_tensor = torch.stack(patches, dim=1)
        position_tensor = torch.tensor(
            positions,
            device=ts_normed.device,
            dtype=ts_normed.dtype,
        )
        return patch_tensor, position_tensor

    def _encode_gpt_features(self, inputs_embeds):
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(
            batch_size, -1
        )
        hidden_states = inputs_embeds + self.wpe(position_ids)
        hidden_states = self.input_dropout(hidden_states)
        for block in self.gpt_blocks:
            hidden_states = block(hidden_states)[0]
        return self.ln_f(hidden_states)

    def _streaming_topk_vocab_matches(self, query):
        vocab = self.normalized_word_embeddings.to(device=query.device, dtype=query.dtype)
        vocab_size = vocab.size(0)
        topk = min(self.soft_vocab_topk, vocab_size)

        candidate_scores = None
        candidate_indices = None
        for start in range(0, vocab_size, self.pattern_vocab_chunk):
            end = min(start + self.pattern_vocab_chunk, vocab_size)
            similarity = torch.matmul(query, vocab[start:end].transpose(0, 1))
            chunk_topk = min(topk, end - start)
            chunk_scores, chunk_indices = torch.topk(similarity, k=chunk_topk, dim=-1)
            chunk_indices = chunk_indices + start

            if candidate_scores is None:
                candidate_scores = chunk_scores
                candidate_indices = chunk_indices
                continue

            merged_scores = torch.cat([candidate_scores, chunk_scores], dim=-1)
            merged_indices = torch.cat([candidate_indices, chunk_indices], dim=-1)
            candidate_scores, top_pos = torch.topk(merged_scores, k=topk, dim=-1)
            candidate_indices = torch.gather(merged_indices, -1, top_pos)

        return candidate_scores, candidate_indices

    def build_calf_text_tokens(self, ts_patch_embeds):
        if self.disable_evidence_chain:
            base_hidden = self.evidence_chain.query_norm(ts_patch_embeds.detach())
            query = self.evidence_chain._safe_normalize(base_hidden)
            tracker_stats = {
                "transition_mean": query.new_tensor(0.0).detach(),
                "stay_gate_mean": query.new_tensor(1.0).detach(),
                "blend": query.new_tensor(0.0).detach(),
                "ablated": query.new_tensor(1.0).detach(),
            }
            tracker_cache = {
                "stay_gate": torch.ones(
                    query.size(0),
                    query.size(1),
                    1,
                    device=query.device,
                    dtype=query.dtype,
                ),
                "transition_score": torch.zeros(
                    query.size(0),
                    query.size(1),
                    1,
                    device=query.device,
                    dtype=query.dtype,
                ),
            }
        else:
            query, tracker_stats, tracker_cache = self.evidence_chain(ts_patch_embeds.detach())
        self._last_evidence_stats = tracker_stats
        self._last_evidence_cache = tracker_cache

        candidate_scores, candidate_indices = self._streaming_topk_vocab_matches(query)
        candidate_embeddings = self.wte(candidate_indices)

        if self.training:
            token_weights = F.gumbel_softmax(
                candidate_scores,
                tau=max(self.gumbel_tau, 1e-6),
                hard=self.use_straight_through_tokens,
                dim=-1,
            )
        else:
            hard_idx = candidate_scores.argmax(dim=-1)
            token_weights = F.one_hot(
                hard_idx,
                num_classes=candidate_scores.size(-1),
            ).to(candidate_scores.dtype)

        pseudo_text_embeds = (token_weights.unsqueeze(-1) * candidate_embeddings).sum(dim=-2)
        pseudo_text_embeds = self.pt_token_bridge(pseudo_text_embeds)
        pseudo_token_ids = torch.gather(
            candidate_indices,
            -1,
            token_weights.argmax(dim=-1, keepdim=True),
        ).squeeze(-1)
        return pseudo_token_ids, pseudo_text_embeds

    def _encode_dual_branches(self, ts_inputs_embeds, pt_inputs_embeds, stability_gate):
        batch_size, seq_len, _ = ts_inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=ts_inputs_embeds.device).unsqueeze(0).expand(
            batch_size, -1
        )
        bridge_traces = [] if self._analysis_enabled else None

        ts_hidden = self.input_dropout(ts_inputs_embeds + self.wpe(position_ids))
        pt_hidden = self.input_dropout(pt_inputs_embeds + self.wpe(position_ids))

        for block_idx, block in enumerate(self.gpt_blocks):
            ts_hidden = block(ts_hidden)[0]
            pt_hidden = block(pt_hidden)[0]
            if (not self.disable_selective_bridge) and block_idx in self.bridge_layers:
                ts_hidden, pt_hidden, _ = self.selective_bridge(
                    ts_hidden,
                    pt_hidden,
                    stability_gate,
                )
                if bridge_traces is not None:
                    bridge_traces.append(
                        {
                            "block_idx": block_idx,
                            **self._cache_analysis_value(
                                getattr(self.selective_bridge, "_last_forward_cache", {})
                            ),
                        }
                    )

        ts_hidden = self.ln_f(ts_hidden)
        pt_hidden = self.ln_f(pt_hidden)
        if bridge_traces is not None:
            self._last_analysis["bridge_traces"] = bridge_traces
        return ts_hidden, pt_hidden

    def _adapter_coop_loss(self, ts_hidden_aug, pt_hidden_updated, anchor_idx):
        if anchor_idx is None or anchor_idx.numel() == 0:
            return ts_hidden_aug.new_tensor(0.0)
        gather_index = anchor_idx.unsqueeze(-1).expand(-1, -1, ts_hidden_aug.size(-1))
        ts_anchor = torch.gather(ts_hidden_aug, 1, gather_index)
        pt_anchor = torch.gather(pt_hidden_updated, 1, gather_index)
        cosine = F.cosine_similarity(ts_anchor, pt_anchor, dim=-1, eps=1e-6)
        return 1.0 - cosine.mean()

    def forward(
        self,
        ts_sample,
        mode="train",
        prompts=None,
        text_emb_init=None,
        alignment_progress=None,
        **kwargs,
    ):
        del prompts, text_emb_init, alignment_progress, kwargs

        if self._analysis_enabled:
            self._last_analysis = {}

        batch_size, seq_len, n_vars = ts_sample.shape
        ts_sample = self.normalize(ts_sample, mode="norm")
        ts_normed = ts_sample.permute(0, 2, 1).contiguous().reshape(
            batch_size * n_vars,
            seq_len,
            1,
        )

        ts_patches, patch_positions = self._extract_patches(ts_normed)
        ts_patch_embeds = self.ts_patch_layer(ts_patches)
        ts_patch_embeds = self.ts_branch_adapter(ts_patch_embeds)

        pseudo_token_ids, pseudo_text_embeds = self.build_calf_text_tokens(ts_patch_embeds)
        pseudo_text_embeds = self.pt_branch_adapter(pseudo_text_embeds)
        if self._analysis_enabled:
            self._last_analysis.update(
                {
                    "patch_positions": self._cache_analysis_value(patch_positions),
                    "ts_patch_embeds": self._cache_analysis_value(ts_patch_embeds),
                    "pseudo_text_embeds": self._cache_analysis_value(pseudo_text_embeds),
                    "pseudo_token_ids": self._cache_analysis_value(pseudo_token_ids),
                    "evidence_stats": self._cache_analysis_value(self._last_evidence_stats),
                    "evidence_cache": self._cache_analysis_value(self._last_evidence_cache),
                }
            )

        stability_gate = self._last_evidence_cache.get("stay_gate")
        if stability_gate is None:
            stability_gate = torch.ones(
                ts_patch_embeds.size(0),
                ts_patch_embeds.size(1),
                1,
                device=ts_patch_embeds.device,
                dtype=ts_patch_embeds.dtype,
            )

        ts_hidden, pt_hidden = self._encode_dual_branches(
            ts_patch_embeds,
            pseudo_text_embeds,
            stability_gate,
        )
        if self._analysis_enabled:
            self._last_analysis.update(
                {
                    "stability_gate": self._cache_analysis_value(stability_gate),
                    "ts_hidden_pre_adapter": self._cache_analysis_value(ts_hidden),
                    "pt_hidden_pre_adapter": self._cache_analysis_value(pt_hidden),
                }
            )

        refined_mode_centroids = self._get_refined_mode_centroids()
        if self.disable_p2t:
            ts_hidden_aug = ts_hidden
            p2t_stats = {
                "anchor_idx": None,
                "anchor_mask": None,
                "anchor_scores": None,
                "gate_mean": ts_hidden.new_tensor(0.0).detach(),
                "mode_entropy": ts_hidden.new_tensor(0.0).detach(),
                "ablated": ts_hidden.new_tensor(1.0).detach(),
            }
        else:
            ts_hidden_aug, p2t_stats = self.p2t_adapter(
                pt_hidden,
                ts_hidden,
                patch_positions,
                refined_mode_centroids,
            )

        pt_hidden_updated = pt_hidden
        if self._analysis_enabled:
            self._last_analysis.update(
                {
                    "analysis_bridge_mode": getattr(self, "_analysis_bridge_mode", "selective"),
                    "refined_mode_centroids": self._cache_analysis_value(refined_mode_centroids),
                    "ts_hidden_aug": self._cache_analysis_value(ts_hidden_aug),
                    "pt_hidden_updated": self._cache_analysis_value(pt_hidden_updated),
                    "p2t_stats": self._cache_analysis_value(p2t_stats),
                    "pre_branch_cosine": self._cache_analysis_value(
                        F.cosine_similarity(ts_hidden, pt_hidden, dim=-1, eps=1e-6)
                    ),
                    "post_branch_cosine": self._cache_analysis_value(
                        F.cosine_similarity(ts_hidden_aug, pt_hidden_updated, dim=-1, eps=1e-6)
                    ),
                    "pre_branch_l2": self._cache_analysis_value(
                        (ts_hidden - pt_hidden).norm(dim=-1)
                    ),
                    "post_branch_l2": self._cache_analysis_value(
                        (ts_hidden_aug - pt_hidden_updated).norm(dim=-1)
                    ),
                }
            )

        if self.final_output_branch == "ts":
            fused_hidden = ts_hidden_aug
        elif self.final_output_branch == "pt":
            fused_hidden = pt_hidden_updated
        else:
            fused_hidden = torch.cat([ts_hidden_aug, pt_hidden_updated], dim=-1)
        if fused_hidden.size(1) != self.patch_nums:
            fused_hidden = F.adaptive_avg_pool1d(
                fused_hidden.transpose(1, 2),
                self.patch_nums,
            ).transpose(1, 2).contiguous()

        fused_hidden = fused_hidden.reshape(batch_size, n_vars, -1)
        dec_out = self.pred_head(fused_hidden).transpose(1, 2).contiguous()
        dec_out = self.normalize(dec_out, mode="denorm")
        if self._analysis_enabled:
            self._last_analysis.update(
                {
                    "fused_hidden": self._cache_analysis_value(fused_hidden),
                    "prediction": self._cache_analysis_value(dec_out),
                }
            )

        if mode == "train":
            coop_loss = self._adapter_coop_loss(
                ts_hidden_aug,
                pt_hidden_updated,
                p2t_stats["anchor_idx"],
            )
            return dec_out, self.adapter_coop_weight * coop_loss
        return dec_out
