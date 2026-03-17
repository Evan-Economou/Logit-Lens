import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer


DEFAULT_TUNED_LENS_PATH = Path(__file__).with_name("tuned-lens-state.pt")


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
    if status_callback is None:
        status_callback = print

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


def save_tuned_lens(tuned_lens_layers: TunedLens, path: Path) -> Path:
    """Save tuned lens weights to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": tuned_lens_layers.state_dict(),
            "n_layers": len(tuned_lens_layers.layer_maps),
            "d_model": tuned_lens_layers.layer_maps[0].in_features,
        },
        path,
    )
    return path


def load_tuned_lens(tuned_lens_layers: TunedLens, path: Path) -> None:
    """Load tuned lens weights from disk into an existing TunedLens module."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Tuned lens state file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    if state_dict is None:
        # Backward compatibility: support raw state_dict files.
        state_dict = payload

    tuned_lens_layers.load_state_dict(state_dict, strict=True)
    tuned_lens_layers.eval()


def _print_top1_preview(preds, token_strs: list[str], max_layers: int = 6) -> None:
    """Print a compact per-layer top-1 token preview in the terminal."""
    layer_count = min(max_layers, len(preds))
    print("\nTop-1 preview by layer:")
    for layer_idx in range(layer_count):
        pos_preds = preds[layer_idx]
        cells = []
        for pos, pred_list in enumerate(pos_preds):
            token, prob = pred_list[0]
            source = token_strs[pos].replace("\n", "\\n")
            pred = token.replace("\n", "\\n")
            cells.append(f"{source}->{pred} ({prob:4.1%})")
        print(f"  L{layer_idx}: " + " | ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal runner for tuned lens utilities")
    parser.add_argument(
        "text",
        nargs="?",
        default="trees are green",
        help="Input text to analyze (default: trees are green)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k tokens per position")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train tuned lens weights from a local parquet file before analysis",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--max-samples", type=int, default=60, help="Max parquet samples")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(__file__).with_name("train-00000-of-00001-4746b8785c874cc7.parquet"),
        help="Path to local parquet file with a text column",
    )
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Load tuned lens weights from this file instead of starting from identity",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_TUNED_LENS_PATH,
        help="Save tuned lens weights to this file after training",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    startup = time.perf_counter()
    print("Loading pythia-14m...")
    model: HookedTransformer = HookedTransformer.from_pretrained("pythia-14m")
    model.eval()
    tuned_lens = TunedLens(model.cfg.n_layers, model.cfg.d_model)
    print(f"Model ready in {time.perf_counter() - startup:.2f}s")

    if args.load is not None:
        load_start = time.perf_counter()
        print(f"Loading tuned lens weights from {args.load}")
        load_tuned_lens(tuned_lens, args.load)
        print(f"Loaded tuned lens in {time.perf_counter() - load_start:.2f}s")

    if args.train:
        print(f"Reading training texts from {args.parquet}")
        texts = load_texts_from_parquet(args.parquet, max_samples=args.max_samples)
        print(f"Loaded {len(texts)} samples")
        train_start = time.perf_counter()
        train_tuned_lens(
            model=model,
            tuned_lens_layers=tuned_lens,
            texts=texts,
            n_epochs=args.epochs,
            status_callback=print,
        )
        print(f"Training finished in {time.perf_counter() - train_start:.2f}s")
        saved_to = save_tuned_lens(tuned_lens, args.save)
        print(f"Saved tuned lens weights to {saved_to}")
    else:
        if args.load is None:
            print("Skipping training and using identity-initialized tuned lens.")
            print("Use --train to fit or --load to restore saved tuned lens weights.")

    analyze_start = time.perf_counter()
    print(f"Running tuned lens on: {args.text!r}")
    with torch.no_grad():
        _, cache = model.run_with_cache(args.text)
        preds = tuned_lens_predictions(model, cache, tuned_lens, top_k=args.top_k)

    tokens = model.to_tokens(args.text)
    token_strs = [model.tokenizer.decode([t.item()]) for t in tokens[0]]
    print(
        f"Analyzed {len(preds)} layers x {len(token_strs)} tokens "
        f"in {time.perf_counter() - analyze_start:.2f}s"
    )
    _print_top1_preview(preds, token_strs)


if __name__ == "__main__":
    main()
