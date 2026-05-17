"""Phase-1 step 1 of EmoDap: Demonstration Pool Curating (Section 3.2.1).

Pipeline (matches the paper):

    1. Run an LLM pseudo-annotator ``M`` times over the unlabeled corpus
       ``D_u`` and majority-vote a pseudo label.  This produces ``D̃_u``.
    2. Encode every pseudo-labelled sentence with a frozen sentence encoder
       (using the [CLS] token) and run K-means *per emotion*.
    3. Collect the cluster centroids' nearest real demonstrations as the
       per-emotion demonstration pool ``D_demo``.

The output is a JSON file::

    {emotion_idx: [demo_1, demo_2, ..., demo_K]}

shaped exactly like ``data/<DATASET>/uns_candis.json`` so it can be consumed
by ``main.py``.

Notes
-----
* When the user provides a ``--pseudo_labels_csv`` we skip step 1 and load the
  already-annotated CSV directly.  This is useful because step 1 is an
  expensive LLM call (the paper uses GPT-4o-mini).
* Step 2/3 only require the *frozen* sentence encoder (``all-mpnet-base-v2``)
  – they do **not** depend on the prototype encoder trained later.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from emodap import PrototypeEncoder
from utils import convert_emotion_json, set_seed


# ---------------------------------------------------------------------------
# Step 1 – Self-consistency pseudo-annotation (optional, requires an LLM)
# ---------------------------------------------------------------------------
def self_consistency_pseudo_label(
    sentences: List[str],
    label_list: List[str],
    llm_call,
    M: int = 5,
) -> List[Optional[str]]:
    """Majority-vote ``M`` LLM predictions per sentence.

    ``llm_call`` is any callable ``(sentence, label_list) -> str`` returning
    the predicted emotion name.  Sentences whose majority count is below 2/M
    are dropped (returned as ``None``) – this enforces the *self-consistency*
    requirement of Xie et al. (2024).
    """
    pseudo: List[Optional[str]] = []
    for s in tqdm(sentences, desc="Self-consistency annotation"):
        votes = []
        for _ in range(M):
            try:
                votes.append(llm_call(s, label_list).strip().lower())
            except Exception:  # noqa: BLE001 – LLM client errors are non-fatal
                continue
        votes = [v for v in votes if v in label_list]
        if not votes:
            pseudo.append(None)
            continue
        cand, cnt = Counter(votes).most_common(1)[0]
        pseudo.append(cand if cnt >= max(2, M // 2) else None)
    return pseudo


# ---------------------------------------------------------------------------
# Step 2 + 3 – per-emotion K-means and centroid demonstration selection
# ---------------------------------------------------------------------------
def kmeans_per_emotion(
    sentences: List[str],
    pseudo_labels: List[str],
    encoder: PrototypeEncoder,
    K: int,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Cluster sentences per emotion and return ``K`` centroid representatives."""
    by_emo: Dict[str, List[str]] = defaultdict(list)
    for s, y in zip(sentences, pseudo_labels):
        if y is not None:
            by_emo[y].append(s)

    out: Dict[str, List[str]] = {}
    encoder.eval()
    for emo, sents in by_emo.items():
        if len(sents) < K:
            # not enough samples – pad with the available ones
            out[emo] = list(sents) + [sents[-1]] * (K - len(sents)) if sents else []
            continue

        # encode in batches
        embs = []
        with torch.no_grad():
            for i in range(0, len(sents), batch_size):
                emb = encoder.encode(sents[i:i + batch_size])
                embs.append(emb.cpu().numpy())
        embs = np.concatenate(embs, axis=0)

        km = KMeans(n_clusters=K, random_state=seed, n_init="auto").fit(embs)
        centroids = km.cluster_centers_
        # nearest real sample to each centroid is taken as the prototype demo
        chosen: List[str] = []
        used = set()
        for c in centroids:
            dist = np.linalg.norm(embs - c[None, :], axis=1)
            order = np.argsort(dist)
            for j in order:
                if j not in used:
                    used.add(j)
                    chosen.append(sents[j])
                    break
        out[emo] = chosen
    return out


def build_demo_pool(
    pseudo_csv: str,
    label_json: str,
    encoder_name_or_path: str,
    output_path: str,
    K: int = 8,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Build ``D_demo`` and dump it to ``output_path`` as JSON.

    ``pseudo_csv`` is a TSV with columns ``text\tlabel`` (header on first row),
    ``label_json`` follows the format of ``data/<DATASET>/uns_candis.json``.
    """
    set_seed(seed)

    df = pd.read_csv(pseudo_csv, delimiter="\t")
    if "text" not in df.columns or "label" not in df.columns:
        df.columns = ["text", "label"][: len(df.columns)]
    sentences = df["text"].astype(str).tolist()
    pseudo_labels = df["label"].astype(str).str.lower().tolist()

    label2idx = convert_emotion_json(label_json)

    encoder = PrototypeEncoder(encoder_name_or_path)
    encoder.to("cuda" if torch.cuda.is_available() else "cpu")

    per_emo = kmeans_per_emotion(sentences, pseudo_labels, encoder, K=K, seed=seed)

    indexed = {str(label2idx[emo]): demos for emo, demos in per_emo.items() if emo in label2idx}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(indexed, f, ensure_ascii=False, indent=2)
    print(f"[demo-pool] wrote {len(indexed)} emotions × {K} demonstrations -> {output_path}")
    return per_emo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_csv", required=True,
                        help="TSV with columns text<TAB>label (already pseudo-labelled)")
    parser.add_argument("--label_json", required=True,
                        help="Emotion-definition JSON file (e.g. test_label.json)")
    parser.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_demo_pool(args.pseudo_csv, args.label_json, args.encoder, args.output,
                    K=args.K, seed=args.seed)


if __name__ == "__main__":
    main()
