"""Closing the loop: user input to a world state, before a reply exists.

The world engine already accepts hidden states and routes them to an appraisal.
What was missing is the part that produces them, and the ordering discipline
around it. This module supplies both::

    input ids
      -> encode_prompt          (Ava's own hidden states, no LM head)
      -> NeuralAppraisal        (small MLP -> 23 context channels)
      -> WorldEngine.observe    (expectation, habituation, impulse, dynamics)
      -> 136-D state
      -> WorldConditioner       (FiLM on the residual stream)
      -> generation

**Ordering is the whole point.** The appraisal runs on the prompt and only the
prompt, and the world it produces conditions the reply that has not been written
yet. Appraising after generation would mean Ava reacts to her own words, and the
state would arrive one turn too late to affect anything.

That ordering costs a second pass over the prompt: one to perceive, one to
prefill under the resulting conditioning. There is no way around it -- the
conditioning depends on the appraisal, which depends on the prompt -- but the
perception pass skips the LM head, which is the expensive part for a small model
with a large vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn

from .appraisal import NUM_CONTEXT, Appraisal, NeuralAppraisal
from .engine import WorldEngine
from .relationships import RELATIONSHIP_SIZE
from .state import WorldState


def encode_prompt(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ava's own hidden states for a prompt, as ``(batch, seq_len, hidden_size)``.

    Runs the decoder stack and stops. No LM head, because the appraisal wants a
    representation rather than a distribution over tokens, and no world
    conditioning, because the world is what is about to be computed -- feeding
    the previous state back in here would blur which of the appraisal's two
    inputs the world dependence actually comes from.

    Causal masking means position *i* sees only positions up to *i*, so nothing
    downstream can leak backwards. What guarantees no leakage from the
    *assistant* is simpler still: this is only ever called on the prompt.
    """
    base = getattr(model, "model", model)
    return base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).last_hidden_state


class AvaPerception:
    """Model, appraisal and world, wired in the order that makes them mean something.

    ``perceive`` turns a prompt into an updated world. ``respond`` does that and
    then generates under it. Everything else -- expectation, prediction error,
    habituation, impulses, coupling, time constants -- is the existing engine,
    untouched.

    The appraisal is independently trainable. By default the language model is
    frozen and only the head carries gradients, which is the order these have to
    be learned in: the head needs a stable representation to attach meaning to.
    """

    def __init__(
        self,
        model: nn.Module,
        engine: WorldEngine | None = None,
        appraisal: Appraisal | None = None,
        freeze_model: bool = True,
        default_dt: float = 30.0,
    ) -> None:
        self.model = model
        self.engine = engine or WorldEngine(device=next(model.parameters()).device)
        self.default_dt = default_dt

        if appraisal is None:
            appraisal = self.engine.appraisal
            if not isinstance(appraisal, NeuralAppraisal):
                appraisal = NeuralAppraisal(
                    hidden_size=model.config.hidden_size,
                    relationship_size=RELATIONSHIP_SIZE,
                )
        self.appraisal = appraisal
        self.engine.appraisal = appraisal

        if isinstance(appraisal, nn.Module):
            appraisal.to(self.engine.device)
        if freeze_model:
            self.freeze_model()

    # --- perception ---

    def appraise(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Context intensities for a prompt, ``(batch, 23)`` in ``[0, 1]``.

        Pure: reads the current world but does not advance it. Useful for
        training the head against labels without running the dynamics.
        """
        hidden = encode_prompt(self.model, input_ids, attention_mask)
        relationship = self.engine.relationships.active
        vector = (
            None
            if relationship is None or not isinstance(self.appraisal, NeuralAppraisal)
            else relationship.vector(self.engine.device, self.engine.dtype)
        )
        state = self.engine.state.expand(input_ids.shape[0])
        return self.appraisal(
            state,
            hidden_states=hidden,
            attention_mask=attention_mask,
            relationship=vector,
        )

    def perceive(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        person: str | None = None,
        dt: float | None = None,
        event: dict[str, float] | None = None,
        detach: bool = True,
        note: str | None = None,
    ) -> WorldState:
        """Encode the prompt, appraise it, and advance the world.

        ``event`` still works and still wins: an explicitly supplied channel
        overrides the learned guess for that channel, so a known signal can be
        injected without switching appraisal back off.

        ``detach=False`` keeps the world in the autograd graph, which is what
        training the head end-to-end through the dynamics requires. The default
        cuts it, because otherwise a long conversation accumulates one graph per
        turn and never frees any of them.
        """
        hidden = encode_prompt(self.model, input_ids, attention_mask)
        state = self.engine.observe(
            event=event,
            dt=self.default_dt if dt is None else dt,
            person=person,
            hidden_states=hidden,
            attention_mask=attention_mask,
            note=note,
        )
        if detach:
            self.engine.state = state.detach()
            state = self.engine.state
        return state

    # --- perception, then reply ---

    @torch.no_grad()
    def respond(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generation_config=None,
        person: str | None = None,
        dt: float | None = None,
        event: dict[str, float] | None = None,
        streamer=None,
        **kwargs,
    ) -> tuple[torch.Tensor, WorldState]:
        """Perceive the prompt, then generate under the world it produced.

        Returns ``(tokens, state)``. The state is the one the reply was actually
        conditioned on, which is what you want to log: reading
        ``engine.state`` afterwards would show a world that has since moved on.
        """
        state = self.perceive(
            input_ids,
            attention_mask=attention_mask,
            person=person,
            dt=dt,
            event=event,
        )
        tokens = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            streamer=streamer,
            world_state=state if self.model.world_conditioner is not None else None,
            **kwargs,
        )
        return tokens, state

    # --- training the head while Ava stays still ---

    def freeze_model(self, keep_conditioner: bool = True) -> None:
        """Freeze the language model, leaving the world-facing parts trainable.

        The conditioner lives inside ``AvaForCausalLM`` but is not part of Ava:
        it is the other half of this wiring, and it has to learn alongside the
        appraisal head. Freezing it too would leave the world computed but with
        no path by which it could ever come to matter.
        """
        self.model.requires_grad_(False)
        self.model.eval()
        conditioner = getattr(self.model, "world_conditioner", None)
        if keep_conditioner and conditioner is not None:
            conditioner.requires_grad_(True)
            conditioner.train()

    def unfreeze_model(self) -> None:
        self.model.requires_grad_(True)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Just the appraisal head, plus the conditioner if one is attached."""
        if isinstance(self.appraisal, nn.Module):
            yield from (p for p in self.appraisal.parameters() if p.requires_grad)
        conditioner = getattr(self.model, "world_conditioner", None)
        if conditioner is not None:
            yield from (p for p in conditioner.parameters() if p.requires_grad)

    def train_appraisal(self, mode: bool = True) -> AvaPerception:
        if isinstance(self.appraisal, nn.Module):
            self.appraisal.train(mode)
        return self

    def to(self, device) -> AvaPerception:
        self.model.to(device)
        if isinstance(self.appraisal, nn.Module):
            self.appraisal.to(device)
        self.engine.device = torch.device(device)
        self.engine.state = self.engine.state.to(device)
        self.engine.dynamics.to(device)
        return self

    def __repr__(self) -> str:
        frozen = not any(p.requires_grad for p in self.model.parameters())
        return (
            f"AvaPerception(appraisal={type(self.appraisal).__name__}, "
            f"context_channels={NUM_CONTEXT}, model_frozen={frozen})"
        )
