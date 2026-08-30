"""Fine tune a standard transformer per EVALUATION.md section 2, then ablate.

Deliberately default settings and a single epoch: the point is to reproduce
what the common project reports, not to tune. The checkpoints are throwaway
and are never loaded by any serving path (MANIFEST.yaml, D010).
"""

from __future__ import annotations

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
MAX_LEN = 256
BATCH = 32
LR = 5e-5
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


def run_one(cond: Condition, train: Split, test: Split, device: torch.device) -> float:
    tr, te = apply_condition(train, test, cond)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(device)
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("device=%s model=%s", device, MODEL_ID)
    train, test = load_corpus()
    dest = Path("eval/leakage/results/transformer.json")
    out: dict[str, dict[str, float | int]] = {}
    if dest.exists():
        out = json.loads(dest.read_text())
    for cond in CONDITIONS:
        if cond in out:
            logger.info("%s already done, skipping", cond)
            continue
        t0 = time.time()
        acc = run_one(cond, train, test, device)
        out[cond] = {"accuracy": round(acc, 4), "seconds": round(time.time() - t0, 1)}
        logger.info("%-24s acc=%.4f", cond, acc)
        dest.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
