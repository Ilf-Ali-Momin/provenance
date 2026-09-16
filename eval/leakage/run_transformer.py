"""Fine tune a standard transformer per EVALUATION.md section 2, then ablate.

Deliberately default settings and a single epoch: the point is to reproduce
what the common project reports, not to tune. The checkpoints are throwaway
and are never loaded by any serving path (MANIFEST.yaml, D010).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval.leakage.ablations import Condition, Split, apply_condition, load_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "distilbert-base-uncased"
# Pinned. RULES.md requires a base model identifier pinned to a revision hash,
# and tools/check_model_pins.py enforces it; this reference was unpinned for the
# whole of M0, so the recorded transformer numbers were produced against
# whatever the default branch held on the day.
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
# Reduced from the full corpus at 256 tokens. The host is under heavy external
# load (macOS software update, load average >150), which starved the full run
# to roughly 36 seconds per step. These settings are stated in the writeup so
# the transformer figures are never read as a full scale reproduction.
MAX_LEN = 128
BATCH = 16
LR = 5e-5
TRAIN_SUBSAMPLE = 6000
TEST_SUBSAMPLE = 2000
CONDITIONS: tuple[Condition, ...] = (
    "baseline",
    "first_sentence_removed",
    "structure_only",
    "length_matched",
)


def _encode(tok: object, split: Split) -> TensorDataset:
    enc = tok(  # type: ignore[operator]
        split.x, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt"
    )
    return TensorDataset(
        enc["input_ids"], enc["attention_mask"], torch.tensor(split.y, dtype=torch.long)
    )


def _subsample(split: Split, n: int, seed: int = 0) -> Split:
    """Take a class balanced random subset of at most `n` documents."""
    if len(split.x) <= n:
        return split
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(split.x), n, replace=False)
    return Split([split.x[i] for i in idx], split.y[idx])


def run_one(
    cond: Condition, train: Split, test: Split, device: torch.device, seed: int = 0
) -> float:
    # Seeded explicitly. Without this the run is not reproducible and repeat
    # runs measure uncontrolled nondeterminism rather than seed variance, which
    # is the opposite of what a reported range is for. Three things draw:
    # the subsample, the classification head's initialisation, and the batch
    # shuffle.
    torch.manual_seed(seed)
    tr, te = apply_condition(train, test, cond)
    tr = _subsample(tr, TRAIN_SUBSAMPLE, seed)
    te = _subsample(te, TEST_SUBSAMPLE, seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, revision=MODEL_REVISION
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    loader = DataLoader(_encode(tok, tr), batch_size=BATCH, shuffle=True)
    model.train()
    for step, (ids, mask, y) in enumerate(loader):
        opt.zero_grad()
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
        out.loss.backward()
        opt.step()
        if step % 200 == 0:
            logger.info("%s step %d/%d loss=%.4f", cond, step, len(loader), out.loss.item())

    model.eval()
    preds: list[int] = []
    gold: list[int] = []
    with torch.no_grad():
        for ids, mask, y in DataLoader(_encode(tok, te), batch_size=64):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(y.tolist())
    return float(np.mean(np.asarray(preds) == np.asarray(gold)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer ablations.")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="repeat each condition this many times, seeded 0..n-1, and record every run",
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("device=%s model=%s", device, MODEL_ID)
    train, test = load_corpus()
    dest = Path("eval/leakage/results/transformer.json")
    out: dict[str, dict[str, object]] = {}
    if dest.exists():
        out = json.loads(dest.read_text())
    for cond in args.conditions:
        entry = out.setdefault(cond, {})
        runs: dict[str, object] = entry.setdefault("runs", {})  # type: ignore[assignment]
        for seed in range(args.seeds):
            if str(seed) in runs:
                logger.info("%s seed %d already done, skipping", cond, seed)
                continue
            t0 = time.time()
            acc = run_one(cond, train, test, device, seed)
            runs[str(seed)] = {"accuracy": round(acc, 4), "seconds": round(time.time() - t0, 1)}
            logger.info("%-24s seed=%d acc=%.4f", cond, seed, acc)
            dest.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
