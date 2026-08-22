"""Advancing the world by a stretch of real time."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import schema
from .coupling import build_coupling_matrix, build_readout_matrix
from .schema import GROUP_SLICES, NUM_CHANNELS
from .state import WorldState


class WorldDynamics(nn.Module):
    """One coupled dynamical system over all 136 channels.

    Each step is::

        s   <- s + impulse                              # the event lands
        u    = tanh(W @ (s - baseline))                 # what everything else wants
        t    = baseline + u                             # where it is being pulled
        a    = 1 - exp(-dt / (tau_rise if t > s else tau_fall))
        s   <- s + a * (t - s)
        s   <- clamp(s, 0, 1)

    Four choices in there carry the behaviour:

    **Impulses are separate from coupling.** An event releases transmitter *now*;
    coupling is a sustained pull. Folding them together would mean a message in
    a 10-second turn barely moved a channel with a 6-minute time constant --
    events would be invisible and only long silences would do anything.

    **Coupling moves the target, not the derivative.** Writing
    ``ds/dt = (baseline - s)/tau + u`` gives a steady state of
    ``baseline + tau * u``, so a channel with a one-week time constant and a
    small drive would slam into its ceiling. Moving the target keeps every
    equilibrium inside the range by construction.

    **The relaxation is exponential, not Euler.** ``a = 1 - exp(-dt/tau)`` is the
    exact solution for constant drive, so it is correct and stable for *any*
    ``dt``. That matters: Ava is asked to advance six hours between conversations
    as readily as eight seconds between turns, and a Euler step with
    ``dt=21600`` against ``tau=60`` does not decay, it detonates.

    **Rise and fall use different constants.** A channel climbing toward its
    target uses ``tau_rise``; one falling back uses ``tau``. Without that
    asymmetry a twelve-second turn moves sadness 0.2% of the way to its target
    and the world is inert during exactly the moments meant to matter -- while
    shortening the constant enough to react would throw away the inertia that
    makes it a mood rather than a lookup.
    """

    def __init__(
        self,
        coupling_gain: float = 1.0,
        learnable: bool = False,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.coupling_gain = coupling_gain

        baseline = torch.tensor(schema.baselines(), device=device, dtype=dtype)
        log_tau = torch.tensor(schema.taus(), device=device, dtype=dtype).log()
        log_tau_rise = torch.tensor(schema.rise_taus(), device=device, dtype=dtype).log()
        coupling = build_coupling_matrix(device=device, dtype=dtype)
        readout = build_readout_matrix(device=device, dtype=dtype)

        integrated = torch.tensor(
            schema.kind_mask("integrated"), device=device, dtype=torch.bool
        )

        if learnable:
            # The hand-written graph becomes an initialisation rather than a
            # commitment: with interaction data, gradients can reshape it.
            self.baseline = nn.Parameter(baseline)
            self.log_tau = nn.Parameter(log_tau)
            self.log_tau_rise = nn.Parameter(log_tau_rise)
            self.coupling = nn.Parameter(coupling)
            self.readout = nn.Parameter(readout)
        else:
            self.register_buffer("baseline", baseline)
            self.register_buffer("log_tau", log_tau)
            self.register_buffer("log_tau_rise", log_tau_rise)
            self.register_buffer("coupling", coupling)
            self.register_buffer("readout", readout)

        self.register_buffer("integrated_mask", integrated)
        self.learnable = learnable

    # --- pieces ---

    def effective_baseline(
        self,
        personality_offset: torch.Tensor | None = None,
        circadian_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Where the world rests, once traits and the time of day are folded in."""
        baseline = self.baseline
        if personality_offset is not None:
            baseline = baseline + personality_offset
        if circadian_offset is not None:
            baseline = baseline + circadian_offset
        return baseline.clamp(0.0, 1.0)

    def tau(self, scale: torch.Tensor | None = None) -> torch.Tensor:
        """Fall time constants."""
        tau = self.log_tau.exp()
        return tau if scale is None else tau * scale.clamp_min(1e-3)

    def tau_rise(self, scale: torch.Tensor | None = None) -> torch.Tensor:
        """Rise time constants; at or below the fall constants everywhere."""
        tau = self.log_tau_rise.exp()
        return tau if scale is None else tau * scale.clamp_min(1e-3)

    def drive(self, values: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        """The tonic pull every channel exerts on every other, bounded by tanh."""
        deviation = values - baseline
        return torch.tanh(self.coupling_gain * deviation @ self.coupling.transpose(0, 1))

    def readout_vad(self, values: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        """Valence, arousal and dominance as an offset from their own baselines."""
        deviation = values - baseline
        raw = deviation @ self.readout.transpose(0, 1)
        centre = self.baseline[GROUP_SLICES["output"]].unsqueeze(0)
        # tanh rather than clamp: a very bad day should keep getting worse in a
        # way the number still reflects, instead of pinning at zero and losing
        # the difference between bad and much worse.
        span = torch.where(raw >= 0, 1.0 - centre, centre)
        return centre + span * torch.tanh(raw / span.clamp_min(1e-3))

    # --- the step ---

    def forward(
        self,
        state: WorldState,
        dt: float | torch.Tensor,
        impulse: torch.Tensor | None = None,
        personality_offset: torch.Tensor | None = None,
        tau_scale: torch.Tensor | None = None,
        circadian_offset: torch.Tensor | None = None,
        clock: torch.Tensor | None = None,
    ) -> WorldState:
        """Advance ``state`` by ``dt`` seconds. Returns a new :class:`WorldState`."""
        values = state.values
        if isinstance(dt, torch.Tensor):
            dt_tensor = dt.to(values).reshape(-1, 1)
        else:
            dt_tensor = torch.as_tensor(float(dt), device=values.device, dtype=values.dtype)
        if (dt_tensor < 0).any():
            raise ValueError("dt must be non-negative; the world does not run backwards.")

        # Two baselines, deliberately. Homeostasis pulls toward the circadian
        # one -- where this channel *should* rest at this hour. Coupling is
        # measured against the static one -- how much of this is present
        # relative to normal, full stop.
        #
        # Collapsing them breaks the clock-driven pathways entirely: adenosine
        # at 0.59 late in the evening sits below its raised night-time setpoint,
        # so a deviation-from-effective-baseline drive would report it as a
        # *negative* signal and sleep pressure would fall as the day wore on.
        resting = self.effective_baseline(personality_offset)
        target_baseline = self.effective_baseline(personality_offset, circadian_offset)
        mask = self.integrated_mask

        if impulse is not None:
            values = values + impulse * mask

        target = target_baseline + self.drive(values, resting)
        rising = target > values
        # tau_scale is personality, and personality governs how long a state is
        # held, not how fast it arrives. Scaling the rise too would make a
        # neurotic Ava *slower* to become stressed, which is backwards -- the
        # magnitude of her response comes from her baselines and reactivity.
        tau = torch.where(rising, self.tau_rise(), self.tau(tau_scale))
        alpha = -torch.expm1(-dt_tensor / tau)
        values = torch.where(mask, values + alpha * (target - values), values)
        values = values.clamp(0.0, 1.0)

        if clock is not None:
            values = values.clone()
            values[:, GROUP_SLICES["temporal"]] = clock

        vad = self.readout_vad(values, resting)
        values = values.clone()
        values[:, GROUP_SLICES["output"]] = vad

        return WorldState(values)

    # --- inspection ---

    @torch.no_grad()
    def explain(
        self, state: WorldState, target: str, top_k: int = 5, index: int = 0
    ) -> list[tuple[str, float]]:
        """Which channels are currently pushing ``target``, and how hard.

        The point of a structured coupling matrix rather than a learned blob:
        every movement has an attributable cause.
        """
        column = schema.index_of(target)
        baseline = self.effective_baseline()
        deviation = (state.values[index] - baseline) * self.integrated_mask
        contributions = self.coupling[column] * deviation

        ranked = sorted(
            ((i, float(v)) for i, v in enumerate(contributions) if abs(float(v)) > 1e-6),
            key=lambda pair: -abs(pair[1]),
        )
        return [(schema.CHANNELS[i].key, round(v, 4)) for i, v in ranked[:top_k]]

    @torch.no_grad()
    def half_life(self, key: str, rising: bool = False) -> float:
        """Seconds for ``key`` to cover half the distance to its target."""
        index = schema.index_of(key)
        source = self.log_tau_rise if rising else self.log_tau
        return float(source[index].exp()) * 0.6931471805599453

    def extra_repr(self) -> str:
        edges = int((self.coupling != 0).sum())
        return (
            f"channels={NUM_CHANNELS}, edges={edges}, "
            f"gain={self.coupling_gain}, learnable={self.learnable}"
        )
