"""Sample from a trained Ava checkpoint.

python scripts/generate.py --model checkpoints/final --prompt "Once upon a time"
python scripts/generate.py --model checkpoints/final --prompt "..." --greedy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from ava import AvaForCausalLM, GenerationConfig
from ava.tokenizer import AvaTokenizer


class ConsoleStreamer:
    """Print tokens as they are produced, so long generations are not silent."""

    def __init__(self, tokenizer: AvaTokenizer) -> None:
        self.tokenizer = tokenizer
        self.buffer: list[int] = []

    def put(self, token_ids: torch.Tensor) -> None:
        self.buffer.append(int(token_ids[0, -1]))
        text = self.tokenizer.decode(self.buffer)
        sys.stdout.write(text[len(getattr(self, "_printed", "")) :])
        sys.stdout.flush()
        self._printed = text

    def end(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--quantize", type=int, default=None, choices=[4, 8])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AvaTokenizer.from_pretrained(args.tokenizer_dir)
    model = AvaForCausalLM.from_pretrained(args.model, device=device, dtype=dtype).eval()

    if args.quantize:
        from ava.model.quantization import quantize_model

        quantize_model(model, bits=args.quantize)

    # No trailing EOS: the prompt is a prefix to continue, not a finished
    # document. BOS still goes in, because that is how the model was trained.
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if tokenizer.bos_token_id >= 0:
        ids = [tokenizer.bos_token_id, *ids]
    input_ids = torch.tensor([ids], device=device)

    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        seed=args.seed,
    )

    streamer = ConsoleStreamer(tokenizer) if args.stream else None
    start = time.perf_counter()
    output = model.generate(input_ids, generation_config=config, streamer=streamer)
    elapsed = time.perf_counter() - start

    new_tokens = output.shape[1] - input_ids.shape[1]
    if not args.stream:
        print(tokenizer.decode(output[0]))
    print(
        f"\n[{new_tokens} tokens in {elapsed:.2f}s "
        f"-- {new_tokens / max(elapsed, 1e-9):.1f} tok/s]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
