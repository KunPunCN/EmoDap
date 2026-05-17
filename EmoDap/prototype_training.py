"""Phase-1 step 2 of EmoDap: Emotion Prototype Learning (Section 3.2.2).

We train a prototype encoder ``Enc(·)`` with the InfoNCE-style contrastive
loss ``L_proto`` (Eq. 3) on the *seen* dataset ``D_s`` (Plutchik's eight basic
emotions).  After training:

    * Emotion prototypes ``V = {v_y}_{y ∈ Y_s ∪ Y_t}`` are computed solely
      from the textual emotion definitions (no labelled instances needed for
      ``Y_t``).
    * All demonstrations in ``D_demo`` are re-encoded into the same space to
      obtain ``H_demo ∈ R^{|Y|·K·d}``.

The script saves three tensors so they can be reused as *frozen* assets by
``main.py``:

    * ``prototype_encoder.pt`` – encoder state-dict
    * ``prototypes.pt``        – ``V``  (shape ``[|Y|, d]``)
    * ``H_demo.pt``            – ``H_demo`` (shape ``[|Y|, K, d]``)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from emodap import PrototypeEncoder, proto_contrastive_loss
from utils import (
    convert_emotion_definitions,
    convert_emotion_json,
    set_seed,
)


class SeenSentenceDataset(Dataset):
    def __init__(self, csv_path: str, label2idx: Dict[str, int]):
        df = pd.read_csv(csv_path, delimiter="\t")
        if "text" not in df.columns or "label" not in df.columns:
            df.columns = ["text", "label"][: len(df.columns)]
        self.texts = df["text"].astype(str).tolist()
        self.labels = [label2idx[str(y).strip().lower()] for y in df["label"].tolist()
                       if str(y).strip().lower() in label2idx]
        # filter texts in lock-step with labels
        self.texts = [t for t, y in zip(df["text"].astype(str).tolist(),
                                        df["label"].astype(str).str.lower().tolist())
                      if y in label2idx]

    def __len__(self): return len(self.texts)

    def __getitem__(self, i): return self.texts[i], self.labels[i]


def _collate(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
def train_prototypes(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ----- 1. labels & definitions ---------------------------------------
    seen_label2idx = convert_emotion_json(args.seen_label_json)
    seen_definitions = convert_emotion_definitions(args.seen_label_json)
    test_label2idx = convert_emotion_json(args.test_label_json)
    test_definitions = convert_emotion_definitions(args.test_label_json)

    # build a unified emotion list: seen first, then unseen (de-duplicated)
    all_emotions: List[str] = list(seen_label2idx.keys())
    for e in test_label2idx:
        if e not in all_emotions:
            all_emotions.append(e)
    full_label2idx = {e: i for i, e in enumerate(all_emotions)}

    # ----- 2. build encoder + dataset -------------------------------------
    encoder = PrototypeEncoder(args.encoder, max_seq_len=args.max_len).to(device)
    dataset = SeenSentenceDataset(args.train_csv, seen_label2idx)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=_collate)

    optim = torch.optim.AdamW(encoder.parameters(), lr=args.lr)

    # ----- 3. contrastive training loop -----------------------------------
    seen_emotions = list(seen_label2idx.keys())
    seen_def_texts = [seen_definitions[e] for e in seen_emotions]

    encoder.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"Proto epoch {epoch}/{args.epochs}")
        running = 0.0
        for step, (texts, labels) in enumerate(pbar, 1):
            labels = labels.to(device)
            # encode sentence batch
            h = encoder.encode(texts, device=device)
            # encode the (small) seen-emotion definitions every step so that
            # gradients flow through both sides of cos(h, v_y)
            v = encoder.encode(seen_def_texts, device=device)

            loss = proto_contrastive_loss(h, v, labels, temperature=args.tau)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optim.step()
            running += loss.item()
            pbar.set_postfix(loss=running / step)

    # ----- 4. derive frozen prototypes V (Y_s ∪ Y_t) ----------------------
    encoder.eval()
    with torch.no_grad():
        full_def_texts = [
            seen_definitions.get(e) or test_definitions.get(e) or e
            for e in all_emotions
        ]
        V = encoder.encode(full_def_texts, device=device)        # [|Y|, d]

        # ----- 5. derive H_demo from D_demo --------------------------------
        with open(args.demo_pool_json, "r", encoding="utf-8") as f:
            demo_pool: Dict[str, List[str]] = json.load(f)
        # demo_pool is keyed by *test* emotion idx; remap to full_label2idx
        K = max(len(v) for v in demo_pool.values())
        d = V.size(1)
        H_demo = torch.zeros(len(all_emotions), K, d, device=device)
        for key, demos in demo_pool.items():
            # try to interpret key as an idx within the test_label2idx
            try:
                tidx = int(key)
                # map tidx -> emotion name -> full idx
                emo_name = list(test_label2idx.keys())[tidx]
            except ValueError:
                emo_name = key.lower()
            if emo_name not in full_label2idx:
                continue
            full_idx = full_label2idx[emo_name]
            embs = encoder.encode(demos, device=device)          # [k, d]
            H_demo[full_idx, : embs.size(0)] = embs

    # ----- 6. save --------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(args.output_dir, "prototype_encoder.pt"))
    torch.save(V.cpu(), os.path.join(args.output_dir, "prototypes.pt"))
    torch.save(H_demo.cpu(), os.path.join(args.output_dir, "H_demo.pt"))
    with open(os.path.join(args.output_dir, "label2idx.json"), "w", encoding="utf-8") as f:
        json.dump(full_label2idx, f, ensure_ascii=False, indent=2)
    print(f"[proto] saved encoder, V {tuple(V.shape)}, H_demo {tuple(H_demo.shape)} "
          f"to {args.output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True,
                   help="TSV with text/label, restricted to seen emotions (e.g. train_see.csv)")
    p.add_argument("--seen_label_json", required=True,
                   help="JSON of seen-emotion definitions (e.g. see_relation.json)")
    p.add_argument("--test_label_json", required=True,
                   help="JSON of unseen-emotion definitions (e.g. test_label.json)")
    p.add_argument("--demo_pool_json", required=True,
                   help="Output JSON from demo_pool_curating.py")
    p.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--output_dir", default="ckpt/proto")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_prototypes(args)


if __name__ == "__main__":
    main()
