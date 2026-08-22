# The internal world

The language model is the cognitive part. The world engine is the environment it
lives in: 136 coupled channels running on their own time constants, advanced by
real elapsed seconds, persisting between conversations.

```python
from ava.world import WorldEngine, Personality

engine = WorldEngine(personality=Personality(neuroticism=0.7))

engine.observe({"loss": 0.9, "psychological_threat": 0.7}, dt=45, person="nika")
engine.idle(1800)
print(engine.summary())
# valence 0.02, arousal 0.41, dominance 0.31 | sadness +0.41; cortisol +0.16; happiness -0.35
```

Nothing in that requires a model. The engine is a dynamical system in its own
right, which is the point of keeping the two apart.

---

## What is actually simulated

| Group | Channels | What it is |
|---|---|---|
| `biochemistry` | 38 | transmitters, hormones, peptides, cytokines |
| `physiology` | 15 | heart rate, HRV, skin conductance, temperature, energy |
| `context` | 23 | decaying traces of what has happened |
| `internal_state` | 10 | hunger, pain, arousal, stress, energy |
| `emotion` | 29 | from surprise to attachment |
| `cognitive_state` | 11 | attention, clarity, rumination, perceived control |
| `temporal` | 7 | written from the clock |
| `output` | 3 | valence, arousal, dominance — a readout, not a cause |

All of it is one `(batch, 136)` float tensor. Names are a lookup over its
columns, not a parallel structure, so every step is a handful of elementwise ops
and a `(136, 136)` matmul — on GPU, differentiable, batched.

```python
state.values.shape            # (1, 136)
state["emotion.trust"]        # a tensor view
state.group("emotion")        # {"happiness": 0.45, ...}
state.to_dict()               # the nested JSON form
```

Names are qualified because three of them are reused: `fatigue` lives in both
`physiology` and `internal_state`, `uncertainty` in both `context` and
`cognitive_state`, `arousal` in three places.

---

## Emotion is an output

Nothing writes to `emotion.sadness`. Sadness is what happens when serotonin
drops, substance P rises, dopamine falls and rumination gets going — and it
fades on its own schedule rather than when the conversation moves on.

The chain runs:

```
event  ->  appraisal  ->  context channels  ->  biochemistry  ->  internal state
                                                     |                 |
                                                     v                 v
                                                  emotion  <-  cognitive state
                                                     |
                                                     v
                                            valence / arousal / dominance
```

Every arrow is an edge in `ava/world/coupling.py`, about 220 of them, each
`(source, target, weight)`. Because the graph is written down rather than
learned, any movement can be attributed:

```python
engine.why("internal_state.stress")
# [('biochemistry.cortisol', 0.31), ('cognitive_state.rumination', 0.09), ...]
```

The matrix is an `nn.Parameter` when `EngineConfig(learnable_dynamics=True)`, so
the hand-written graph is a starting point rather than a commitment.

**Honesty about the weights.** The *time constants* are chosen to match what the
real substance does. The *couplings* are chosen so the behaviour is
recognisable. Cortisol does not literally multiply stress by 0.7. What matters
is that the loops exist, have the right sign, and settle at the right speed.

---

## Time is the whole thing

### Every channel forgets at its own rate

| Channel | Falls with | Rises with |
|---|---|---|
| `biochemistry.epinephrine` | 2 min | 10 s |
| `biochemistry.cortisol` | 72 min | 22 min |
| `emotion.anger` | 12 min | 22 s |
| `emotion.sadness` | 90 min | 2.7 min |
| `emotion.attachment` | 1 week | 3.4 h |
| `emotion.love` | 30 days | 11 h |

An event that spikes several channels at once leaves them decaying at completely
different rates, so the state an hour later is not a scaled copy of the state at
the time — it has a different *shape*. That is what inertia means here.

### Rise and fall are different

A single time constant per channel cannot describe an emotion. Sadness arrives
in seconds and leaves in hours. With one constant, a twelve-second conversational
turn against a ninety-minute decay moves sadness by **0.2%** — the world would be
inert during exactly the moments meant to matter — while shortening the constant
enough to react would throw away the inertia entirely.

So `tau` governs the fall and `tau * rise` the rise, with `rise` well below 1 for
nearly everything.

### The integrator is exponential, not Euler

```
alpha = 1 - exp(-dt / tau)
s    += alpha * (target - s)
```

This is the exact solution for constant drive, so it is correct and stable for
*any* `dt`. That matters because Ava is asked to advance six hours between
conversations as readily as eight seconds between turns, and a Euler step with
`dt=21600` against `tau=60` does not decay, it detonates.

### Long gaps are sliced

Stability is not accuracy. The exponential step evaluates the coupling **once**
per call. Advance half an hour in one jump and cortisol will climb — driven by
the event — while stress never sees it climb, because stress read cortisol once,
at the start, before it moved. Every indirect pathway silently disappears.

`WorldEngine.step` therefore integrates long gaps in slices that start at 15
seconds and grow geometrically to 30 minutes. A six-hour idle costs about sixty
`(136, 136)` matmuls.

`tests/test_world.py::test_indirect_pathways_actually_run` is the guard.

---

## Context decides what an event means

`observe()` runs an **appraisal**: incoming event plus the *current world* mapped
onto context intensities. Because the world is an input, the same words are not
the same event.

```python
calm = WorldEngine()
calm.observe({"social_rejection": 0.5}, dt=60)

fraught = WorldEngine()
fraught.observe({"social_rejection": 0.8, "psychological_threat": 0.7}, dt=60)
fraught.idle(1200)
fraught.observe({"social_rejection": 0.5}, dt=60)   # identical input
# fraught ends up considerably sadder
```

Two implementations, and the difference is deliberate:

- **`DirectAppraisal`** — you supply the intensities. For scripted scenarios,
  tests, and when an external classifier already exists.
- **`NeuralAppraisal`** — a learned head over the model's hidden state, the world
  and the relationship. It needs training data to be worth anything; an
  untrained head on an untrained model produces noise, and no architecture fixes
  that. It is zero-initialised so it invents nothing until it is trained.

The head predicts the 23 **context** channels and nothing else. Not emotions,
not hormones — those are what the coupled dynamics are *for*, and a head writing
to them directly would be a sentiment classifier wearing the world as a costume.
What it emits is closer to *how much rejection was in that*; the graph decides
what rejection does to this Ava at this hour.

### Automatic appraisal

`AvaPerception` runs the whole chain from token ids:

```python
from ava.world import AvaPerception, WorldEngine

perception = AvaPerception(model, WorldEngine())      # model frozen by default
tokens, state = perception.respond(input_ids, attention_mask, person="nika")
```

```
input ids
  -> encode_prompt        Ava's own hidden states, decoder stack only, no LM head
  -> NeuralAppraisal      masked mean-pool -> 2-layer MLP -> 23 channels, sigmoid
  -> WorldEngine.observe  expectation, habituation, impulse, coupling, dynamics
  -> 136-D state
  -> WorldConditioner     FiLM on the residual stream
  -> generation
```

**The ordering is the point.** The appraisal reads the prompt and only the
prompt, and the world it produces conditions a reply that does not exist yet.
Appraising afterwards would have Ava reacting to her own words, one turn late.

That costs a second pass over the prompt — one to perceive, one to prefill under
the resulting conditioning. There is no way around it, since the conditioning
depends on the appraisal which depends on the prompt, but the perception pass
skips the LM head.

Pooling is **masked**. A plain mean drags a short prompt toward whatever the pad
embedding encodes, so the same sentence would be appraised differently depending
on what it was batched with.

`perceive(event={...})` still works and still wins: an explicit channel
overrides the learned guess for that channel without turning appraisal off.

### Training the head

`AvaPerception` freezes the language model and leaves the appraisal head *and*
the conditioner trainable — the conditioner lives inside `AvaForCausalLM` but is
the other half of this wiring, and freezing it would leave the world computed
with no path by which it could come to matter.

```python
optimizer = torch.optim.AdamW(perception.trainable_parameters(), lr=1e-4)

# against labelled context, without running the dynamics
loss = F.binary_cross_entropy(perception.appraise(ids, mask), targets)

# or end to end through the engine
state = perception.perceive(ids, mask, detach=False)   # keeps the graph
```

`detach=True` is the inference default; otherwise a long conversation
accumulates one autograd graph per turn and frees none of them.

### Expectation and prediction error

The engine keeps a decaying average of what usually happens. Prediction error is
the gap between the appraisal that arrived and that average — signed error over
reward-like channels reaches dopamine, magnitude reaches surprise. This is what
makes the second compliment land more softly than the first, and an ordinary
evening after a terrible week feel like relief rather than merely neutral.

The first observation *seeds* the average rather than violating it. You do not
startle at the world existing.

---

## Personality is a standing bias, not a module

Nothing in the step function knows about traits. It sees a shifted baseline and
a scaled tau.

```python
Personality(neuroticism=0.9)
# cortisol baseline  +0.13
# stress tau (fall)  x1.7
```

Crucially the scale applies **only to the fall**. Personality governs how long a
state is held, not how fast it arrives — scaling the rise as well would make a
neurotic Ava *slower* to become stressed, which is backwards. Response magnitude
comes from baselines and `reactivity`.

The result, from the same event:

| | stress at 30 min | at 3.5 h |
|---|---|---|
| `anxious` | 0.33 | **0.33** |
| `stoic` | 0.27 | **0.10** |

Both get stressed. Only one is still stressed later.

Archetypes: `balanced`, `anxious`, `warm`, `stoic`, `curious`.

---

## Different people get different Avas

`emotion.*` is how Ava feels now. A `Relationship` is what she has come to feel
about *someone*, which is a different object on a much longer clock.

Coupling runs both ways. On entry, a bond shifts the baselines of the social
channels — talking to someone trusted literally *rests* at a higher trust, so
the same remark from a friend and from a stranger starts from different places.
On exit, the world writes back: oxytocin and warmth feed bond, rejection and
anger feed conflict.

The rates are asymmetric on purpose. Bond and trust climb over many
conversations; conflict arrives in one. Absence fades conflict fastest and trust
slowest — an argument stops mattering long before you stop relying on someone.

```python
engine.observe({"social_acceptance": 0.8}, dt=60, person="nika")
engine.relationships.get("nika")
# Relationship('nika' bond=0.21 trust=0.38 conflict=0.00 seen=14)
```

---

## Things that happen with no input at all

This is the part that makes it a world rather than a response function.

- **Circadian rhythm.** Melatonin peaks at 03:00, cortisol at 08:00, body
  temperature at 17:00 — periodic, so nothing jumps at midnight.
- **Sleep pressure.** Adenosine builds with time awake and only `sleep()` clears
  it. After fourteen hours up, `physiology.sleep_pressure` is 0.46.
- **Rumination.** Stress feeds rumination, rumination feeds stress. Only the
  time constants stop it, so a bad evening can keep itself going.
- **Isolation.** `time_since_social_interaction` raises `context.social_isolation`,
  which feeds loneliness, which feeds sadness and suppresses serotonin. Three
  days alone and Ava is measurably worse, with nothing having happened.

Nothing decided any of that. It falls out of the graph.

### Two baselines, deliberately

Homeostasis pulls toward the *circadian* baseline — where a channel should rest
at this hour. Coupling is measured against the *static* baseline — how much is
present relative to normal, full stop.

Collapsing them breaks every clock-driven pathway: adenosine at 0.59 late in the
evening sits below its raised night-time setpoint, so a
deviation-from-effective-baseline drive reports it as a *negative* signal and
sleep pressure falls as the day wears on.

---

## Reaching the language model

The requirement is that the world influences what Ava says without being *told*
to her. Writing `you are feeling anxious (0.71)` into the prompt makes a model
describe anxiety; it does not make it anxious, and it puts the state where the
user can read it. A state that only matters when narrated is not internal.

So conditioning happens inside the network:

```python
config = AvaConfig.from_preset(
    "hybrid-130m",
    world_conditioning=True,
    world_conditioning_layers=(4, 8, 11),
)
model = AvaForCausalLM(config)

output = model(input_ids=ids, world_state=engine.state)
```

`WorldConditioner` projects the 136 channels into a per-layer scale and shift,
applied to the residual stream as `h * (1 + gamma) + beta`. That is a bias on how
the model computes, not a fact it reasons about — closer to being in a mood than
to knowing you are in one.

Everything is zero-initialised, so **an untrained conditioner is exactly a
no-op**. Attaching a world to a model trained without one changes nothing until
the conditioner is trained. `world_prefix_tokens` adds a soft prompt instead, if
explicit reflection in the output is what you want.

The world state is a tensor in the autograd graph, so gradients flow back into
it — and with `learnable_dynamics=True`, into the coupling matrix and time
constants themselves.

`WorldSummary` renders the state as text for logs and for the case where a system
prompt is the only lever available. It is deliberately not the primary path.

---

## Persistence

```python
engine.save("ava_world.json")
engine = WorldEngine.load("ava_world.json")
```

The clock is absolute, so a session saved and resumed a day later resumes with a
day having passed: rested, hungry, and having missed you.

---

## What is not built

- **Memory.** The engine has expectations and relationships, not episodes. A
  retrieval system is a separate problem and belongs beside this, not inside it.
- **Goals and planning.** `emotion.motivation` exists; nothing pursues anything.
- **A trained appraisal.** `NeuralAppraisal` is wired and zero-initialised.
  Making it useful needs labelled interaction data.
- **Validated parameters.** The couplings are a plausible caricature. Treat any
  specific number as a hypothesis.

## Files

| File | What it holds |
|---|---|
| `schema.py` | the 136 channels, baselines, time constants |
| `coupling.py` | ~220 edges and the VAD readout |
| `dynamics.py` | the integrator |
| `clock.py` | absolute time, circadian and homeostatic drives |
| `personality.py` | traits as baseline and tau shifts |
| `appraisal.py` | event to meaning, expectation, prediction error |
| `relationships.py` | per-person bonds |
| `conditioning.py` | FiLM and soft prefix into the model |
| `engine.py` | the loop that ties it together |
