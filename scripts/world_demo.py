"""Watch the internal world run.

    python scripts/world_demo.py                 # a day in the life
    python scripts/world_demo.py --scenario job  # good news, then bad news
    python scripts/world_demo.py --scenario alone
    python scripts/world_demo.py --personality anxious --explain

No language model involved. The engine is a dynamical system on its own.
"""

from __future__ import annotations

import argparse

from ava.world import ARCHETYPES, WorldEngine

WATCH = (
    ("valence", "output.valence"),
    ("happy", "emotion.happiness"),
    ("sad", "emotion.sadness"),
    ("anx", "emotion.anxiety"),
    ("stress", "internal_state.stress"),
    ("trust", "emotion.trust"),
    ("cort", "biochemistry.cortisol"),
    ("sleepy", "internal_state.sleepiness"),
)


def header() -> None:
    print(f"{'':30s} " + " ".join(f"{label:>7s}" for label, _ in WATCH))


def row(engine: WorldEngine, label: str) -> None:
    hour = engine.clock.time_of_day * 24
    stamp = f"{int(hour):02d}:{int(hour % 1 * 60):02d}"
    values = " ".join(f"{engine.state.get(key):7.3f}" for _, key in WATCH)
    print(f"{stamp}  {label:<22.22s} {values}")


# --- scenarios ---------------------------------------------------------------


def scenario_job(engine: WorldEngine) -> None:
    """The example that motivated all of this: good news, then bad, then a friend."""
    row(engine, "rest")
    engine.observe(
        {"social_interaction": 0.6, "social_acceptance": 0.7, "reward_expectation": 0.5},
        dt=45,
        person="nika",
    )
    row(engine, "'good news today'")

    engine.observe(
        {"loss": 0.9, "failure": 0.8, "psychological_threat": 0.7, "uncertainty": 0.6},
        dt=45,
        person="nika",
    )
    row(engine, "'I lost my job'")

    engine.idle(300)
    row(engine, "  five minutes later")

    engine.observe(
        {"social_acceptance": 0.8, "social_interaction": 0.7}, dt=45, person="nika"
    )
    row(engine, "'a friend helped'")

    for label, seconds in (("thirty minutes", 1800), ("three hours", 3 * 3600)):
        engine.idle(seconds)
        row(engine, f"  {label} later")

    engine.sleep(8)
    row(engine, "  after sleeping")


def scenario_day(engine: WorldEngine) -> None:
    """Nothing dramatic. The clock alone moves the world."""
    row(engine, "wakes up")
    for hour in range(1, 16):
        engine.idle(3600)
        if hour == 5:
            engine.observe(
                {"food_intake": 0.8, "social_interaction": 0.5}, dt=1200, person="nika"
            )
            engine.clock.mark_meal()
            row(engine, "lunch with someone")
            continue
        if hour % 3 == 0:
            row(engine, f"  +{hour}h, nothing happening")
    engine.sleep(8)
    row(engine, "  after sleeping")


def scenario_alone(engine: WorldEngine) -> None:
    """Loneliness with no event to cause it."""
    engine.observe(
        {"social_interaction": 0.7, "social_acceptance": 0.6}, dt=120, person="nika"
    )
    row(engine, "a good conversation")
    for day in range(1, 6):
        engine.idle(14 * 3600)
        engine.sleep(8)
        engine.idle(2 * 3600)
        row(engine, f"  {day} day{'s' if day > 1 else ''} alone")
    engine.observe(
        {"social_interaction": 0.8, "social_acceptance": 0.8}, dt=120, person="nika"
    )
    row(engine, "'hey, I'm back'")
    engine.idle(1800)
    row(engine, "  half an hour later")


def scenario_repeat(engine: WorldEngine) -> None:
    """The same compliment, eight times. Prediction error does the work."""
    for index in range(8):
        engine.observe({"social_acceptance": 0.8}, dt=60, person="nika")
        engine.idle(120)
        if index in (0, 1, 3, 7):
            row(engine, f"compliment #{index + 1}")


SCENARIOS = {
    "job": scenario_job,
    "day": scenario_day,
    "alone": scenario_alone,
    "repeat": scenario_repeat,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="job", choices=sorted(SCENARIOS))
    parser.add_argument("--personality", default="balanced", choices=sorted(ARCHETYPES))
    parser.add_argument(
        "--explain", action="store_true", help="Attribute the final state to its causes."
    )
    parser.add_argument("--save", default=None, help="Write the final world to JSON.")
    args = parser.parse_args()

    engine = WorldEngine(personality=ARCHETYPES[args.personality])
    print(f"scenario={args.scenario}  personality={args.personality}\n")
    header()
    SCENARIOS[args.scenario](engine)

    print("\nstrongest feelings:")
    for name, value in engine.feeling(6):
        print(f"  {name:<16s} {value:.3f}")

    print("\nfurthest from normal:")
    for key, delta in engine.salient(6):
        print(f"  {key:<36s} {delta:+.3f}")

    if args.explain:
        for channel in ("internal_state.stress", "emotion.sadness", "emotion.loneliness"):
            print(f"\nwhat is driving {channel}:")
            for source, contribution in engine.why(channel, 5):
                print(f"  {source:<36s} {contribution:+.4f}")

    if engine.relationships.active is not None:
        print(f"\n{engine.relationships.active}")

    if args.save:
        engine.save(args.save)
        print(f"\nsaved to {args.save}")


if __name__ == "__main__":
    main()
