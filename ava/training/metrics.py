import math

import torch

try:
    from tqdm import tqdm
except ImportError:
    from tqdm.notebook import tqdm


def compute_perplexity(loss):
    """Compute perplexity from loss value (clamped for numerical stability)."""
    return math.exp(min(loss, 20))


def evaluate_model(model, eval_dataloader, device, use_amp=False):
    """Evaluation function with optional AMP support."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = outputs["loss"]
            total_loss += loss.item()
            num_batches += 1

    model.train()
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss
