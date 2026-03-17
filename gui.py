"""
Simple HTML Logit Lens UI for pythia-14m.
Run: python gui.py
Open: http://127.0.0.1:5000
"""

import threading
import json
import re
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


PARQUET_PATH = Path(__file__).with_name("train-00000-of-00001-4746b8785c874cc7.parquet")
TUNED_LENS_MODULE_PATH = Path(__file__).with_name("tuned-lens.py")
TUNED_LENS_WEIGHTS_PATH = Path(__file__).with_name("tuned-lens-state.pt")

_tuned_spec = importlib.util.spec_from_file_location("tuned_lens_module", TUNED_LENS_MODULE_PATH)
if _tuned_spec is None or _tuned_spec.loader is None:
  raise RuntimeError(f"Failed to load tuned-lens module at {TUNED_LENS_MODULE_PATH}")
_tuned_module = importlib.util.module_from_spec(_tuned_spec)
_tuned_spec.loader.exec_module(_tuned_module)

TunedLens = _tuned_module.TunedLens
tuned_lens_predictions = _tuned_module.tuned_lens_predictions
load_texts_from_parquet = _tuned_module.load_texts_from_parquet
train_tuned_lens = _tuned_module.train_tuned_lens
save_tuned_lens = _tuned_module.save_tuned_lens
load_tuned_lens = _tuned_module.load_tuned_lens


# ── core computation ──────────────────────────────────────────────────────────

def logit_lens_predictions(model: HookedTransformer, cache, top_k: int):
    """Return predictions[layer][pos] = [(token_str, prob), ...]."""
    results = []
    for layer in range(model.cfg.n_layers):
        resid = cache["resid_post", layer]       # [1, seq, d_model]
        normed = model.ln_final(resid)
        logits = model.unembed(normed)           # [1, seq, vocab]
        probs = F.softmax(logits[0], dim=-1)     # [seq, vocab]
        top_probs, top_ids = torch.topk(probs, top_k, dim=-1)
        layer_preds = []
        for pos in range(probs.shape[0]):
            layer_preds.append(
                [
                    (model.tokenizer.decode([top_ids[pos, i].item()]),
                     top_probs[pos, i].item())
                    for i in range(top_k)
                ]
            )
        results.append(layer_preds)
    return results


# ── HTML interface server ─────────────────────────────────────────────────────

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Logit Lens</title>
  <style>
    :root {
      --bg: #f2f2ef;
      --panel: #ffffff;
      --ink: #1b1b1a;
      --muted: #5b605d;
      --line: #d8d5cd;
      --accent: #0f766e;
      --accent2: #8c2f39;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(1200px 400px at 20% -10%, #d8e7e4 10%, var(--bg) 60%);
      color: var(--ink);
      font: 15px/1.4 "Segoe UI", Tahoma, sans-serif;
    }
    .wrap { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 1rem;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    h1 { margin: 0 0 0.75rem; font-size: 1.4rem; }
    .row { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }
    input, select, button {
      padding: 0.5rem 0.6rem;
      border-radius: 8px;
      border: 1px solid var(--line);
      font: inherit;
    }
    input[type="text"] { min-width: min(100%, 420px); flex: 1; }
    button { cursor: pointer; }
    .btn-primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .btn-secondary {
      background: var(--accent2);
      color: #fff;
      border-color: var(--accent2);
    }
    .status { margin-top: 0.7rem; color: var(--muted); min-height: 1.2rem; }
    .progress-shell {
      margin-top: 0.55rem;
      width: 100%;
      height: 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #f6f5f1, #efede7);
      overflow: hidden;
      position: relative;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #0f766e, #14b8a6);
      border-radius: inherit;
      transition: width 280ms ease;
      position: relative;
    }
    .progress-fill.active::after {
      content: "";
      position: absolute;
      inset: 0;
      background: repeating-linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.32) 0px,
        rgba(255, 255, 255, 0.32) 10px,
        rgba(255, 255, 255, 0.06) 10px,
        rgba(255, 255, 255, 0.06) 20px
      );
      animation: slide 1s linear infinite;
    }
    .progress-meta {
      margin-top: 0.35rem;
      font-size: 0.82rem;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
    }
    @keyframes slide {
      from { background-position: 0 0; }
      to { background-position: 40px 0; }
    }
    .table-wrap {
      margin-top: 1rem;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }
    table { border-collapse: collapse; width: 100%; min-width: 700px; }
    th, td {
      border-bottom: 1px solid #efede7;
      border-right: 1px solid #efede7;
      padding: 0.45rem;
      text-align: left;
      vertical-align: top;
      font-size: 0.9rem;
    }
    th { position: sticky; top: 0; background: #f8f7f3; }
    td small { color: var(--muted); }
    @media (max-width: 700px) {
      .wrap { margin-top: 1rem; }
      h1 { font-size: 1.2rem; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>Logit Lens / Tuned Lens (pythia-14m)</h1>
      <div class="row">
        <input id="text" type="text" value="trees are green" />
        <select id="mode">
          <option value="logit">Logit Lens</option>
          <option value="tuned">Tuned Lens</option>
        </select>
        <select id="topk">
          <option>1</option>
          <option>3</option>
          <option selected>5</option>
          <option>8</option>
          <option>10</option>
        </select>
        <button class="btn-primary" id="analyzeBtn">Analyze</button>
        <button class="btn-secondary" id="trainBtn">Train and Save Tuned Lens</button>
      </div>
      <div class="row" style="margin-top: 0.6rem;">
        <input id="lensPath" type="text" value="tuned-lens-state.pt" />
        <button class="btn-secondary" id="loadBtn">Load Tuned Lens Weights</button>
      </div>
      <div class="status" id="status">Loading model...</div>
      <div class="progress-shell" aria-label="Training progress">
        <div class="progress-fill" id="progressFill"></div>
      </div>
      <div class="progress-meta">
        <span id="progressStage">Idle</span>
        <span id="progressPct">0%</span>
      </div>
      <div class="table-wrap" id="results"></div>
    </div>
  </div>

  <script>
    const statusEl = document.getElementById("status");
    const resultsEl = document.getElementById("results");
    const progressFillEl = document.getElementById("progressFill");
    const progressStageEl = document.getElementById("progressStage");
    const progressPctEl = document.getElementById("progressPct");

    function setProgress(pct, stage, inProgress) {
      const clamped = Math.max(0, Math.min(100, Number(pct) || 0));
      progressFillEl.style.width = `${clamped}%`;
      progressPctEl.textContent = `${Math.round(clamped)}%`;
      progressStageEl.textContent = stage || "Idle";
      if (inProgress) {
        progressFillEl.classList.add("active");
      } else {
        progressFillEl.classList.remove("active");
      }
    }

    function parseProgressFromStatus(trainingStatus) {
      const m = String(trainingStatus || "").match(/\\((\\d{1,3})%\\)/);
      if (!m) {
        return null;
      }
      const n = Number(m[1]);
      if (Number.isNaN(n)) {
        return null;
      }
      return Math.max(0, Math.min(100, n));
    }

    function esc(s) {
      return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function renderTable(payload) {
      const tokens = payload.tokens;
      const rows = payload.top1;
      let html = "<table><thead><tr><th>Layer</th>";
      for (let i = 0; i < tokens.length; i++) {
        html += `<th>${i}: ${esc(tokens[i])}</th>`;
      }
      html += "</tr></thead><tbody>";
      for (const row of rows) {
        html += `<tr><td><b>L${row.layer}</b></td>`;
        for (const cell of row.cells) {
          html += `<td>${esc(cell.token)}<br><small>${(cell.prob * 100).toFixed(1)}%</small></td>`;
        }
        html += "</tr>";
      }
      html += "</tbody></table>";
      resultsEl.innerHTML = html;
    }

    async function refreshStatus() {
      try {
        const r = await fetch("/status");
        const data = await r.json();
        let msg = data.model_status;
        if (data.training_status) {
          msg += ` | ${data.training_status}`;
        }
        statusEl.textContent = msg;

        const pctFromServer = Number(data.training_progress);
        const hasPctFromServer = Number.isFinite(pctFromServer);
        const pctFromText = parseProgressFromStatus(data.training_status);
        const isTraining = Boolean(data.training_in_progress);
        if (hasPctFromServer) {
          setProgress(pctFromServer, data.training_status, isTraining);
        } else if (pctFromText !== null) {
          setProgress(pctFromText, data.training_status, isTraining);
        } else if (isTraining) {
          // Keep a visible active bar during non-percent phases.
          setProgress(15, data.training_status || "Training...", true);
        } else if ((data.training_status || "").toLowerCase().includes("complete")) {
          setProgress(100, data.training_status, false);
        } else {
          setProgress(0, data.training_status || "Idle", false);
        }
      } catch {
        statusEl.textContent = "Failed to reach server.";
        setProgress(0, "Disconnected", false);
      }
    }

    async function analyze() {
      const text = document.getElementById("text").value.trim();
      const mode = document.getElementById("mode").value;
      const top_k = Number(document.getElementById("topk").value);
      if (!text) {
        statusEl.textContent = "Please enter text.";
        return;
      }
      statusEl.textContent = "Running analysis...";
      const r = await fetch("/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, mode, top_k }),
      });
      const data = await r.json();
      if (!r.ok) {
        statusEl.textContent = data.error || "Analysis failed.";
        return;
      }
      renderTable(data);
      statusEl.textContent = `${data.mode} complete. ${data.layers} layers x ${data.positions} token positions.`;
    }

    async function train() {
      statusEl.textContent = "Starting tuned lens training on local parquet...";
      setProgress(5, "Starting training...", true);
      const lensPath = document.getElementById("lensPath").value.trim();
      const r = await fetch("/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ save_path: lensPath }),
      });
      const data = await r.json();
      if (!r.ok) {
        statusEl.textContent = data.error || "Training failed to start.";
        setProgress(0, "Failed to start", false);
        return;
      }
      statusEl.textContent = data.message;
      setProgress(10, "Queued", true);
    }

    async function loadLens() {
      const lensPath = document.getElementById("lensPath").value.trim();
      statusEl.textContent = "Loading tuned lens weights...";
      const r = await fetch("/load_tuned_lens", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: lensPath }),
      });
      const data = await r.json();
      if (!r.ok) {
        statusEl.textContent = data.error || "Failed to load tuned lens.";
        return;
      }
      statusEl.textContent = data.message;
      if (Number.isFinite(Number(data.training_progress))) {
        setProgress(Number(data.training_progress), "Loaded tuned lens", false);
      }
    }

    document.getElementById("analyzeBtn").addEventListener("click", analyze);
    document.getElementById("trainBtn").addEventListener("click", train);
    document.getElementById("loadBtn").addEventListener("click", loadLens);
    setInterval(refreshStatus, 1500);
    refreshStatus();
  </script>
</body>
</html>
"""


class LensService:
    def __init__(self):
        self.model: HookedTransformer | None = None
        self.tuned_lens: TunedLens | None = None
        self.model_status = "Loading model..."
        self.training_status = ""
        self.training_progress = 0
        self.training_in_progress = False
        self.tuned_lens_path = str(TUNED_LENS_WEIGHTS_PATH)
        self.lock = threading.Lock()
        self._load_model_async()

    def _load_model_async(self):
        def _load():
            import warnings

            warnings.filterwarnings("ignore")
            try:
                model = HookedTransformer.from_pretrained("pythia-14m")
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)

                with self.lock:
                    self.model = model
                    self.tuned_lens = TunedLens(model.cfg.n_layers, model.cfg.d_model)
                    self.model_status = "Ready"
            except Exception as exc:
                with self.lock:
                    self.model_status = f"Model load failed: {exc}"

        threading.Thread(target=_load, daemon=True).start()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "model_status": self.model_status,
                "training_status": self.training_status,
              "training_progress": self.training_progress,
                "training_in_progress": self.training_in_progress,
                "tuned_lens_path": self.tuned_lens_path,
            }

    def analyze(self, text: str, mode: str, top_k: int) -> dict[str, Any]:
        with self.lock:
            model = self.model
            tuned_lens = self.tuned_lens

        if model is None:
            raise RuntimeError("Model is still loading.")

        with torch.no_grad():
            _, cache = model.run_with_cache(text)
            if mode == "tuned":
                if tuned_lens is None:
                    raise RuntimeError("Tuned lens is unavailable.")
                preds = tuned_lens_predictions(model, cache, tuned_lens, top_k)
                mode_label = "Tuned Lens"
            else:
                preds = logit_lens_predictions(model, cache, top_k)
                mode_label = "Logit Lens"

        tokens = model.to_tokens(text)
        token_strs = [model.tokenizer.decode([t.item()]) for t in tokens[0]]

        top1_rows: list[dict[str, Any]] = []
        for layer_idx, pos_preds in enumerate(preds):
            cells = []
            for pred_list in pos_preds:
                token, prob = pred_list[0]
                cells.append({"token": token, "prob": prob})
            top1_rows.append({"layer": layer_idx, "cells": cells})

        return {
            "mode": mode_label,
            "layers": len(preds),
            "positions": len(token_strs),
            "tokens": token_strs,
            "top1": top1_rows,
        }

    def train_async(self, n_epochs: int = 3, max_samples: int = 60, save_path: Path | None = None):
        with self.lock:
            if self.training_in_progress:
                raise RuntimeError("Training is already running.")
            if self.model is None or self.tuned_lens is None:
                raise RuntimeError("Model is not ready yet.")
            self.training_in_progress = True
            self.training_status = "Loading local parquet..."
            self.training_progress = 3
        target_path = Path(save_path) if save_path is not None else TUNED_LENS_WEIGHTS_PATH
        self.tuned_lens_path = str(target_path)

        def _train():
            try:
                with self.lock:
                    model = self.model
                    tuned_lens = self.tuned_lens

                if model is None or tuned_lens is None:
                    raise RuntimeError("Model became unavailable during training.")

                with self.lock:
                    self.training_status = "Reading local parquet..."
                    self.training_progress = 8

                texts = load_texts_from_parquet(PARQUET_PATH, max_samples=max_samples)

                with self.lock:
                    self.training_status = f"Loaded {len(texts)} samples. Preparing training..."
                    self.training_progress = 15

                def _status(msg: str):
                    with self.lock:
                        self.training_status = msg
                        m = re.search(r"\((\d{1,3})%\)", msg)
                        if m:
                            self.training_progress = max(0, min(100, int(m.group(1))))

                train_tuned_lens(
                    model=model,
                    tuned_lens_layers=tuned_lens,
                    texts=texts,
                    n_epochs=n_epochs,
                    status_callback=_status,
                )

                saved_path = save_tuned_lens(tuned_lens, target_path)

                with self.lock:
                    self.training_status = f"Training complete and saved to {saved_path}"
                    self.training_progress = 100
            except Exception as exc:
                with self.lock:
                    self.training_status = f"Training failed: {exc}"
                    self.training_progress = 0
            finally:
                with self.lock:
                    self.training_in_progress = False

        threading.Thread(target=_train, daemon=True).start()

    def load_tuned_lens_from_path(self, path: Path | None = None) -> Path:
        with self.lock:
            if self.training_in_progress:
                raise RuntimeError("Cannot load tuned lens while training is in progress.")
            if self.tuned_lens is None:
                raise RuntimeError("Tuned lens is unavailable.")
            tuned_lens = self.tuned_lens

        target_path = Path(path) if path is not None else TUNED_LENS_WEIGHTS_PATH
        load_tuned_lens(tuned_lens, target_path)
        with self.lock:
            self.training_status = f"Loaded tuned lens from {target_path}"
            self.training_progress = 100
            self.tuned_lens_path = str(target_path)
        return target_path


service = LensService()


class LensRequestHandler(BaseHTTPRequestHandler):
  def _send_json(self, payload: dict[str, Any], status_code: int = 200):
    body = json.dumps(payload).encode("utf-8")
    self.send_response(status_code)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def _send_html(self, html: str, status_code: int = 200):
    body = html.encode("utf-8")
    self.send_response(status_code)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def _read_json_body(self) -> dict[str, Any]:
    length_raw = self.headers.get("Content-Length", "0")
    try:
      length = int(length_raw)
    except ValueError:
      return {}
    if length <= 0:
      return {}
    raw = self.rfile.read(length)
    try:
      parsed = json.loads(raw.decode("utf-8"))
    except Exception:
      return {}
    return parsed if isinstance(parsed, dict) else {}

  def do_GET(self):
    if self.path == "/":
      self._send_html(HTML_PAGE)
      return

    if self.path == "/status":
      self._send_json(service.status())
      return

    self._send_json({"error": "Not found."}, status_code=404)

  def do_POST(self):
    if self.path == "/analyze":
      payload = self._read_json_body()
      text = str(payload.get("text", "")).strip()
      mode = str(payload.get("mode", "logit")).strip().lower()
      top_k_raw = payload.get("top_k", 5)

      if not text:
        self._send_json({"error": "Text is required."}, status_code=400)
        return
      if mode not in {"logit", "tuned"}:
        self._send_json({"error": "Mode must be 'logit' or 'tuned'."}, status_code=400)
        return

      try:
        top_k = int(top_k_raw)
      except Exception:
        self._send_json({"error": "top_k must be an integer."}, status_code=400)
        return
      if top_k < 1 or top_k > 10:
        self._send_json({"error": "top_k must be between 1 and 10."}, status_code=400)
        return

      try:
        data = service.analyze(text=text, mode=mode, top_k=top_k)
      except Exception as exc:
        self._send_json({"error": str(exc)}, status_code=500)
        return

      self._send_json(data)
      return

    if self.path == "/train":
      payload = self._read_json_body()
      save_path_raw = payload.get("save_path") if isinstance(payload, dict) else None
      save_path = Path(str(save_path_raw).strip()) if save_path_raw else None

      try:
        service.train_async(n_epochs=3, max_samples=60, save_path=save_path)
      except Exception as exc:
        self._send_json({"error": str(exc)}, status_code=400)
        return

      if save_path is None:
        message = f"Training started on local parquet. Weights will be saved to {TUNED_LENS_WEIGHTS_PATH}."
      else:
        message = f"Training started on local parquet. Weights will be saved to {save_path}."
      self._send_json({"message": message})
      return

    if self.path == "/load_tuned_lens":
      payload = self._read_json_body()
      path_raw = payload.get("path") if isinstance(payload, dict) else None
      load_path = Path(str(path_raw).strip()) if path_raw else None
      try:
        loaded_path = service.load_tuned_lens_from_path(load_path)
      except Exception as exc:
        self._send_json({"error": str(exc)}, status_code=400)
        return

      self._send_json({"message": f"Loaded tuned lens from {loaded_path}", "training_progress": 100})
      return

    self._send_json({"error": "Not found."}, status_code=404)

  def log_message(self, format: str, *args):
    # Keep terminal output clean; status is visible in the web UI.
    return


def main():
  server = ThreadingHTTPServer(("127.0.0.1", 5000), LensRequestHandler)
  print("Serving on http://127.0.0.1:5000")
  server.serve_forever()


if __name__ == "__main__":
    main()
