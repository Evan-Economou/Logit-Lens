from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer


class TunedLens(nn.Module):
    """One learned affine map per transformer layer.

    Each tuned lens layer at index l is associated with transformer layer l
    and maps h_l -> approx h_L.
    """

    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.layer_maps = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=True) for _ in range(n_layers)
        ])
        for layer_map in self.layer_maps:
            nn.init.eye_(layer_map.weight)
            nn.init.zeros_(layer_map.bias)


def tuned_lens_predictions(model: HookedTransformer, cache,
                           tuned_lens_layers: TunedLens, top_k: int):
    """Return predictions[layer][pos] = [(token_str, prob), ...] via tuned lens."""
    results = []
    tuned_lens_layers.eval()
    for layer in range(model.cfg.n_layers):
        resid = cache["resid_post", layer]
        transformed = tuned_lens_layers.layer_maps[layer](resid[0])
        normed = model.ln_final(transformed.unsqueeze(0))
        logits = model.unembed(normed)
        probs = F.softmax(logits[0], dim=-1)
        top_probs, top_ids = torch.topk(probs, top_k, dim=-1)
        layer_preds = []
        for pos in range(probs.shape[0]):
            layer_preds.append([
                (model.tokenizer.decode([top_ids[pos, i].item()]),
                 top_probs[pos, i].item())
                for i in range(top_k)
            ])
        results.append(layer_preds)
    return results


def load_texts_from_parquet(parquet_path: Path, max_samples: int = 60,
                            min_chars: int = 30, max_chars: int = 600) -> list[str]:
    """Load training texts from a local parquet file only."""
    if not parquet_path.exists():
        raise RuntimeError(f"Parquet file not found: {parquet_path}")

    parquet = pq.ParquetFile(parquet_path)
    if "text" not in parquet.schema.names:
        raise RuntimeError("Parquet file must contain a 'text' column.")

    texts: list[str] = []
    for batch in parquet.iter_batches(columns=["text"], batch_size=512):
        col = batch.column(0)
        for i in range(len(col)):
            value = col[i].as_py()
            if value is None:
                continue
            text = str(value).strip()
            if len(text) >= min_chars:
                texts.append(text[:max_chars])
            if len(texts) >= max_samples:
                break
        if len(texts) >= max_samples:
            break

    if not texts:
        raise RuntimeError("No suitable training text found in parquet file.")

    return texts


def train_tuned_lens(model: HookedTransformer, tuned_lens_layers: TunedLens,
                     texts: list[str], n_epochs: int,
                     status_callback=None) -> None:
    """Train one tuned lens layer per corresponding transformer layer."""
    if status_callback is not None:
        status_callback("Initializing tuned lens layers...")

    for layer_map in tuned_lens_layers.layer_maps:
        nn.init.eye_(layer_map.weight)
        nn.init.zeros_(layer_map.bias)

    optimizer = torch.optim.Adam(tuned_lens_layers.parameters(), lr=1e-3)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    tuned_lens_layers.train()

    if status_callback is not None:
        status_callback("Training started...")

    total_steps = max(1, n_epochs * len(texts))
    step = 0

    for epoch in range(n_epochs):
        for text in texts:
            try:
                with torch.no_grad():
                    _, cache = model.run_with_cache(text)
                    final_resid = cache["resid_post", model.cfg.n_layers - 1]
                    final_logits = model.unembed(model.ln_final(final_resid))[0]
                    target_probs = F.softmax(final_logits, dim=-1).detach()

                optimizer.zero_grad()
                layer_losses = []
                for layer in range(model.cfg.n_layers):
                    resid = cache["resid_post", layer][0].detach()
                    transformed = tuned_lens_layers.layer_maps[layer](resid)
                    normed = model.ln_final(transformed.unsqueeze(0))[0]
                    layer_logits = model.unembed(normed.unsqueeze(0))[0]
                    log_probs = F.log_softmax(layer_logits, dim=-1)
                    layer_losses.append(
                        kl_loss(log_probs, target_probs)
                    )
                loss = torch.stack(layer_losses).sum()
                loss.backward()
                optimizer.step()
            except Exception:
                pass

            step += 1
            if status_callback is not None:
                pct = int(step / total_steps * 100)
                status_callback(f"Epoch {epoch + 1}/{n_epochs} ({pct}%)")

    tuned_lens_layers.eval()


# Backward-compatible alias.
TunedLensLayers = TunedLens
TunedLensProbes = TunedLens
