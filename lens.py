import argparse

import torch
import torch.nn.functional as F
from rich import box
from rich.console import Console
from rich.table import Table
from transformer_lens import HookedTransformer

console = Console()


def prob_color(p: float) -> str:
    if p > 0.5:
        return "bold green"
    elif p > 0.2:
        return "green"
    elif p > 0.08:
        return "yellow"
    else:
        return "dim white"


def logit_lens_predictions(
    model: HookedTransformer, cache, top_k: int
) -> list[list[list[tuple[str, float]]]]:
    """Return predictions[layer][pos] = [(token_str, prob), ...]."""
    results = []
    for layer in range(model.cfg.n_layers):
        resid = cache["resid_post", layer]          # [1, seq, d_model]
        normed = model.ln_final(resid)              # [1, seq, d_model]
        logits = model.unembed(normed)              # [1, seq, vocab]
        probs = F.softmax(logits[0], dim=-1)        # [seq, vocab]
        top_probs, top_ids = torch.topk(probs, top_k, dim=-1)
        layer_preds = []
        for pos in range(probs.shape[0]):
            layer_preds.append(
                [
                    (model.tokenizer.decode([top_ids[pos, i].item()]), top_probs[pos, i].item())
                    for i in range(top_k)
                ]
            )
        results.append(layer_preds)
    return results


def display_logit_lens(model: HookedTransformer, text: str, top_k: int = 3) -> None:
    tokens = model.to_tokens(text)
    token_strs = [model.tokenizer.decode([t.item()]) for t in tokens[0]]
    seq_len = len(token_strs)

    with torch.no_grad():
        logits, cache = model.run_with_cache(text)
        layer_preds = logit_lens_predictions(model, cache, top_k)

    # ── per-layer table ───────────────────────────────────────────────────────
    layer_table = Table(
        title=f'[bold cyan]Logit Lens  ·  pythia-14m  ·  "{text}"[/bold cyan]',
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold magenta",
        show_lines=True,
    )
    layer_table.add_column("Layer", style="bold yellow", justify="center", no_wrap=True)
    for i, tok in enumerate(token_strs):
        label = repr(tok).strip("'\"")
        layer_table.add_column(
            f"[dim]pos {i}[/dim]\n[bold white]{label}[/bold white]",
            justify="center",
            min_width=14,
        )

    for layer_idx, pos_preds in enumerate(layer_preds):
        row: list[str] = [f"[bold yellow]L{layer_idx}[/bold yellow]"]
        for preds in pos_preds:
            lines = []
            for rank, (tok, prob) in enumerate(preds):
                color = prob_color(prob)
                prefix = "▶ " if rank == 0 else "  "
                lines.append(f"[{color}]{prefix}{repr(tok):<10} {prob:5.1%}[/{color}]")
            row.append("\n".join(lines))
        layer_table.add_row(*row)

    console.print()
    console.print(layer_table)

    # ── final output table ────────────────────────────────────────────────────
    final_probs = F.softmax(logits[0], dim=-1)          # [seq, vocab]
    top_final_probs, top_final_ids = torch.topk(final_probs, top_k, dim=-1)

    final_table = Table(
        title="[bold cyan]Final Layer Predictions[/bold cyan]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold magenta",
        show_lines=True,
    )
    final_table.add_column("Pos", style="bold yellow", justify="center")
    final_table.add_column("Input token", justify="center")
    for i in range(top_k):
        final_table.add_column(f"Top-{i + 1}", justify="center", min_width=14)

    for pos in range(seq_len):
        preds = [
            (model.tokenizer.decode([top_final_ids[pos, i].item()]), top_final_probs[pos, i].item())
            for i in range(top_k)
        ]
        row = [str(pos), f"[bold white]{repr(token_strs[pos]).strip(chr(39))}[/bold white]"]
        for tok, prob in preds:
            color = prob_color(prob)
            row.append(f"[{color}]{repr(tok):<10} {prob:5.1%}[/{color}]")
        final_table.add_row(*row)

    console.print(final_table)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Logit Lens for pythia-14m")
    parser.add_argument(
        "text",
        nargs="?",
        default="trees are green",
        help='Input text to analyse (default: "trees are green")',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top tokens to show per cell (default: 3)",
    )
    args = parser.parse_args()

    console.print("[bold blue]Loading pythia-14m …[/bold blue]")
    model: HookedTransformer = HookedTransformer.from_pretrained("pythia-14m")
    model.eval()

    display_logit_lens(model, args.text, top_k=args.top_k)


if __name__ == "__main__":
    main()
