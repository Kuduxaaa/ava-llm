import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AvaAttention
from .embeddings import AvaRotaryEmbedding
from .mamba import MambaBlock
from .mlp import AvaMLP
from .normalization import AvaRMSNorm


class AvaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = AvaAttention(config)

        self.mlp = AvaMLP(config)
        self.input_layernorm = AvaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps
        )

        self.post_attention_layernorm = AvaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        rotary_emb=None,
    ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            rotary_emb=rotary_emb,
        )

        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (attn_outputs[1],)

        if output_attentions:
            outputs += (attn_outputs[2],)

        return outputs


class AvaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        arch = getattr(config, "architecture_type", "transformer")
        self.architecture_type = arch
        num_layers = config.num_hidden_layers
        num_attn = getattr(config, "num_attention_layers", num_layers)

        if arch == "transformer":
            self.layers = nn.ModuleList(
                [AvaDecoderLayer(config) for _ in range(num_layers)]
            )
        elif arch == "mamba":
            self.layers = nn.ModuleList(
                [MambaBlock(config) for _ in range(num_layers)]
            )
        elif arch == "hybrid":
            num_mamba = num_layers - num_attn
            layers = []
            for i in range(num_layers):
                if i < num_mamba:
                    layers.append(MambaBlock(config))
                else:
                    layers.append(AvaDecoderLayer(config))
            self.layers = nn.ModuleList(layers)
        else:
            raise ValueError(f"Unknown architecture_type: {arch}")

        self.has_attention = arch in ("transformer", "hybrid")
        self.norm = AvaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        if self.has_attention:
            self.rotary_emb = AvaRotaryEmbedding(
                config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            self.rotary_emb = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        batch_size, seq_length = (
            input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        )
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=device
            ).unsqueeze(0)

        past_length = 0
        if past_key_values is not None:
            # Find a non-None past_key_value to get past_length
            for pkv in past_key_values:
                if pkv is not None:
                    past_length = pkv[0].shape[2]
                    break
            position_ids = position_ids[:, past_length:]

        # Build attention mask only when padding is present.
        # When None, SDPA uses is_causal=True (FlashAttention fast path).
        expanded_attn_mask = None
        if self.has_attention and attention_mask is not None:
            causal_mask = torch.triu(
                torch.full(
                    (seq_length, seq_length), -float("inf"), device=device,
                ),
                diagonal=1,
            )
            expanded_attn_mask = (
                1.0 - attention_mask.unsqueeze(1).unsqueeze(2)
            ) * torch.finfo(torch.float32).min
            expanded_attn_mask = expanded_attn_mask + causal_mask.unsqueeze(0).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_past_key_values = () if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            is_mamba = getattr(layer, "is_mamba", False)
            layer_outputs = layer(
                hidden_states,
                attention_mask=expanded_attn_mask if not is_mamba else None,
                position_ids=position_ids if not is_mamba else None,
                past_key_value=past_key_value if not is_mamba else None,
                output_attentions=output_attentions if not is_mamba else False,
                use_cache=use_cache,
                rotary_emb=self.rotary_emb if not is_mamba else None,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_past_key_values += (layer_outputs[1],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_past_key_values,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }


class AvaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AvaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = output["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": output.get("past_key_values", None),
            "hidden_states": output.get("hidden_states", None),
            "attentions": output.get("attentions", None),
        }

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        input_shape = input_ids.shape
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:

            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=None,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=None,
        eos_token_id=None,
        use_cache=True,
        streamer=None,
        early_stopping=True,
    ):
        """
        Improved generate method with streamer support and better caching
        """
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        max_length = (
            max_length
            if max_length is not None
            else self.config.max_position_embeddings
        )
        batch_size = input_ids.shape[0]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        input_ids_seq_length = input_ids.shape[-1]
        generated_tokens = input_ids.clone()
        cached_position_ids = (
            torch.arange(input_ids_seq_length, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        past_key_values = None

        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        self.eval()

        for current_length in range(input_ids_seq_length, max_length):
            if past_key_values is not None:
                inputs = generated_tokens[:, -1].unsqueeze(-1)
            else:
                inputs = generated_tokens

            if past_key_values is not None:
                position_ids = cached_position_ids[:, -1].unsqueeze(-1) + 1
                cached_position_ids = torch.cat(
                    [cached_position_ids, position_ids], dim=-1
                )
            else:
                position_ids = cached_position_ids

            if attention_mask is not None and past_key_values is not None:
                attention_mask = torch.cat(
                    [attention_mask, unfinished_sequences.unsqueeze(-1)], dim=-1
                )

            model_inputs = self.prepare_inputs_for_generation(
                inputs,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
            )

            outputs = self.forward(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            next_token_logits = next_token_logits / temperature

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in generated_tokens[i]:
                        if previous_token in [pad_token_id, eos_token_id]:
                            continue

                        next_token_logits[i, previous_token] /= repetition_penalty

            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits.scatter_(1, top_k_indices, top_k_values)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float("-inf")

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + eos_token_id * (
                    1 - unfinished_sequences
                )

            generated_tokens = torch.cat(
                [generated_tokens, next_tokens.unsqueeze(-1)], dim=-1
            )

            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long()
                )

            if streamer is not None:
                streamer.put(next_tokens.unsqueeze(-1))

            if unfinished_sequences.max() == 0 or (
                early_stopping and current_length > input_ids_seq_length + 50
            ):
                break

        if streamer is not None:
            streamer.end()

        return generated_tokens
