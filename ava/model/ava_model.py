"""The Ava decoder stack and its causal-LM head."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..config import AvaConfig
from ..world.conditioning import WorldConditioner, apply_film
from .attention import AvaAttention, build_causal_mask
from .cache import AvaCache
from .embeddings import AvaRotaryEmbedding
from .generation import GenerationConfig, select_next_token
from .mamba import MambaBlock
from .mlp import AvaMLP
from .normalization import AvaRMSNorm


@dataclass
class BaseModelOutput:
    last_hidden_state: torch.Tensor
    cache: AvaCache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None
    cache: AvaCache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class AvaDecoderLayer(nn.Module):
    """Pre-norm attention + gated MLP."""

    is_mamba = False

    def __init__(self, config: AvaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = AvaAttention(config, layer_idx=layer_idx)
        self.mlp = AvaMLP(config)
        self.input_layernorm = AvaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = AvaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_cache=None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        attn_output, attn_weights = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            layer_cache=layer_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, attn_weights


class AvaModel(nn.Module):
    """Embedding table, the layer stack, and the final norm."""

    def __init__(self, config: AvaConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            AvaDecoderLayer(config, i) if kind == "attention" else MambaBlock(config, i)
            for i, kind in enumerate(config.layer_types())
        )
        self.norm = AvaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.rotary_emb = (
            AvaRotaryEmbedding(
                config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=config.rope_scaling,
            )
            if config.has_attention
            else None
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: AvaCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        world_modulation: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> BaseModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if use_cache and cache is None:
            cache = AvaCache.from_config(self.config)
        past_length = cache.seen_tokens if cache is not None else 0

        if position_ids is None:
            position_ids = (
                torch.arange(past_length, past_length + seq_len, device=device)
                .unsqueeze(0)
                .expand(batch, -1)
            )

        position_embeddings = None
        attention_bias = None
        if self.config.has_attention:
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
            attention_bias = build_causal_mask(
                attention_mask, batch, seq_len, past_length, inputs_embeds.dtype, device
            )

        hidden_states = inputs_embeds
        all_hidden_states: list[torch.Tensor] = []
        all_attentions: list[torch.Tensor] = []

        for index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_cache = cache[index] if cache is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    position_embeddings,
                    attention_bias,
                    layer_cache,
                    output_attentions,
                    use_reentrant=False,
                )
            else:
                hidden_states, attn_weights = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_bias,
                    layer_cache=layer_cache,
                    output_attentions=output_attentions,
                )

            if world_modulation is not None and index in world_modulation:
                hidden_states = apply_film(hidden_states, world_modulation[index])

            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if cache is not None:
            cache.advance(seq_len)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            cache=cache,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )


class AvaForCausalLM(nn.Module):
    """Ava with a language-modelling head."""

    def __init__(self, config: AvaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AvaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self._scale_residual_projections()

        # Built after the global init pass, not before. The conditioner's FiLM
        # heads are deliberately zeroed so an untrained conditioner is exactly a
        # no-op, and _init_weights would overwrite that with a normal.
        self.world_conditioner = (
            WorldConditioner(
                hidden_size=config.hidden_size,
                num_layers=config.num_hidden_layers,
                conditioning_layers=config.world_conditioning_layers,
                width=config.world_conditioning_width,
                num_prefix_tokens=config.world_prefix_tokens,
                scale=config.world_conditioning_scale,
            )
            if config.world_conditioning
            else None
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    # --- initialisation ---

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range

        def keep(tensor) -> bool:
            # Layers that carry a purpose-built init (Mamba's dt schedule) opt
            # out rather than being flattened back to a plain normal.
            return getattr(tensor, "_no_reinit", False)

        if isinstance(module, nn.Linear):
            if not keep(module.weight):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None and not keep(module.bias):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

        # nn.Conv1d is deliberately left at PyTorch's fan-in default. The only
        # convolution here is Mamba's depthwise kernel, whose fan-in is d_conv
        # (4), so the default standard deviation is ~0.29. Overwriting it with
        # initializer_range (0.02) attenuates the SSM branch by more than an
        # order of magnitude and leaves the residual stream an identity path --
        # which with tied embeddings degenerates into "predict the current
        # token" and starts training well above ln(vocab_size).

    def _scale_residual_projections(self) -> None:
        """Shrink the projections that write into the residual stream.

        Every layer adds to the same stream, so without a ``1/sqrt(2N)`` factor
        the stream's variance grows linearly in depth and deep models start with
        activations far outside the range the norms were tuned for.
        """
        if not self.config.scaled_residual_init:
            return
        scale = (2 * self.config.num_hidden_layers) ** -0.5
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight", "out_proj.weight")):
                with torch.no_grad():
                    param.mul_(scale)

    # --- embedding plumbing ---

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """Grow or shrink the vocabulary, preserving the weights that survive.

        New rows are drawn from the same distribution as the original init --
        re-creating the layer with PyTorch's default would give them a standard
        deviation of 1.0 and swamp the pretrained rows.
        """
        old = self.model.embed_tokens
        if new_vocab_size == old.num_embeddings:
            return

        new = nn.Embedding(new_vocab_size, self.config.hidden_size).to(
            old.weight.device, old.weight.dtype
        )
        nn.init.normal_(new.weight, mean=0.0, std=self.config.initializer_range)
        keep = min(new_vocab_size, old.num_embeddings)
        with torch.no_grad():
            new.weight[:keep] = old.weight[:keep]
        self.model.embed_tokens = new

        head = nn.Linear(self.config.hidden_size, new_vocab_size, bias=False).to(
            old.weight.device, old.weight.dtype
        )
        if self.config.tie_word_embeddings:
            head.weight = new.weight
        else:
            nn.init.normal_(head.weight, mean=0.0, std=self.config.initializer_range)
            with torch.no_grad():
                head.weight[:keep] = self.lm_head.weight[:keep]
        self.lm_head = head
        self.config.vocab_size = new_vocab_size

    def gradient_checkpointing_enable(self, enable: bool = True) -> None:
        self.model.gradient_checkpointing = enable
        self.config.gradient_checkpointing = enable

    def num_parameters(self, trainable_only: bool = False) -> int:
        seen: set[int] = set()
        total = 0
        for param in self.parameters():
            if id(param) in seen:  # tied weights must not be counted twice
                continue
            seen.add(id(param))
            if trainable_only and not param.requires_grad:
                continue
            total += param.numel()
        return total

    # --- forward ---

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: AvaCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        num_logits_to_keep: int = 0,
        world_state=None,
    ) -> CausalLMOutput:
        """``num_logits_to_keep=1`` computes logits for the last position only.

        During decoding the other positions are thrown away anyway, and the head
        is a ``hidden_size x vocab_size`` matmul -- skipping it is the single
        biggest saving in the decode loop for a small model with a large vocab.
        """
        world_modulation = None
        if world_state is not None:
            if self.world_conditioner is None:
                raise ValueError(
                    "This model was built without world conditioning. Set "
                    "AvaConfig(world_conditioning=True) to attach a conditioner."
                )
            world_modulation = self.world_conditioner(world_state)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            world_modulation=world_modulation,
        )

        hidden_states = output.last_hidden_state
        if num_logits_to_keep > 0:
            hidden_states = hidden_states[:, -num_logits_to_keep:]
        logits = self.lm_head(hidden_states)

        loss = z_loss = None
        if labels is not None:
            loss, z_loss = self._compute_loss(logits, labels)

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            z_loss=z_loss,
            cache=output.cache,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
        shift_labels = labels[:, 1:].reshape(-1).to(shift_logits.device)

        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        z_loss = None
        coefficient = self.config.z_loss_coef
        if coefficient > 0:
            # Keeps logsumexp near zero so the softmax denominator cannot drift;
            # this is what stops late-run loss spikes in bf16 training.
            valid = shift_labels != -100
            if valid.any():
                log_z = torch.logsumexp(shift_logits[valid], dim=-1)
                z_loss = coefficient * log_z.square().mean()
                loss = loss + z_loss
        return loss, z_loss

    # --- generation ---

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        streamer=None,
        world_state=None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive decoding with a real KV / SSM cache.

        Works identically for transformer, Mamba and hybrid stacks: the cache
        object holds whatever state each layer type needs.

        ``world_state`` conditions every step on the same internal world. It is
        passed per step rather than precomputed: the conditioner is a ~50k
        parameter MLP against a model orders of magnitude larger, and keeping it
        inside the normal forward path means there is exactly one place where
        conditioning is applied.
        """
        config = generation_config or GenerationConfig(
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )
        for key, value in kwargs.items():
            if not hasattr(config, key):
                raise TypeError(f"Unknown generation argument {key!r}")
            setattr(config, key, value)
        config.__post_init__()

        if config.eos_token_id is None:
            config.eos_token_id = self.config.eos_token_id
            config.__post_init__()
        pad_token_id = (
            config.pad_token_id
            if config.pad_token_id is not None
            else self.config.pad_token_id
        )

        was_training = self.training
        self.eval()

        device = input_ids.device
        batch, prompt_length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=device).manual_seed(config.seed)

        stop_ids = torch.tensor(config.stop_token_ids, device=device)
        unfinished = torch.ones(batch, dtype=torch.bool, device=device)
        generated = input_ids
        cache = AvaCache.from_config(self.config) if config.use_cache else None

        # Absolute positions, correct even when the batch is left-padded.
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        step_input = input_ids

        for _ in range(config.budget(prompt_length)):
            output = self(
                input_ids=step_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache,
                use_cache=config.use_cache,
                num_logits_to_keep=1,
                world_state=world_state,
            )
            cache = output.cache
            next_token = select_next_token(
                output.logits[:, -1], config, generated, generator
            )

            # Finished rows emit padding, not a fresh sample.
            next_token = torch.where(
                unfinished, next_token, torch.full_like(next_token, pad_token_id)
            )
            generated = torch.cat([generated, next_token[:, None]], dim=1)

            if streamer is not None:
                streamer.put(next_token[:, None])

            if stop_ids.numel():
                unfinished &= ~torch.isin(next_token, stop_ids)
            if not unfinished.any():
                break

            attention_mask = torch.cat([attention_mask, unfinished.long()[:, None]], dim=1)
            position_ids = position_ids[:, -1:] + 1
            step_input = next_token[:, None]

            if not config.use_cache:
                step_input = generated
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)

        if streamer is not None:
            streamer.end()
        if was_training:
            self.train()
        return generated

    # --- persistence ---

    def save_pretrained(self, path: str | os.PathLike, safe: bool = True) -> None:
        """Write config + weights. Prefers safetensors; falls back to ``torch.save``."""
        os.makedirs(path, exist_ok=True)
        self.config.save_pretrained(path)

        state_dict = self.state_dict()
        if self.config.tie_word_embeddings:
            state_dict.pop("lm_head.weight", None)

        if safe:
            try:
                from safetensors.torch import save_file

                save_file(
                    {k: v.contiguous() for k, v in state_dict.items()},
                    os.path.join(path, "model.safetensors"),
                    metadata={"format": "pt"},
                )
                return
            except ImportError:
                pass
        torch.save(state_dict, os.path.join(path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(
        cls, path: str | os.PathLike, device=None, dtype=None, strict: bool = True
    ) -> AvaForCausalLM:
        config = AvaConfig.from_pretrained(path)
        model = cls(config)

        safe_path = os.path.join(path, "model.safetensors")
        bin_path = os.path.join(path, "pytorch_model.bin")
        if os.path.isfile(safe_path):
            from safetensors.torch import load_file

            state_dict = load_file(safe_path)
        elif os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights found in {path}.")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing = [
            k for k in missing if k != "lm_head.weight" or not config.tie_word_embeddings
        ]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Checkpoint does not match the model.\n"
                f"  missing: {missing}\n  unexpected: {unexpected}"
            )

        if dtype is not None:
            model = model.to(dtype)
        if device is not None:
            model = model.to(device)
        return model

    @classmethod
    def from_preset(cls, name: str, **overrides) -> AvaForCausalLM:
        return cls(AvaConfig.from_preset(name, **overrides))
