"""EmoDap inference workflow (right-hand side of Figure 2 in the paper).

Given a trained retriever checkpoint produced by ``main.py`` together with the
frozen Phase-1 assets (prototypes, H_demo, demo_pool), this script:

    1. Encodes each test sentence ``x``.
    2. Builds the dynamic candidate set ``Y_cand`` via Top-ρ filtering (Eq. 4).
    3. Runs the retriever ``π_θ1`` greedily (Eq. 5) to pick one demonstration
       per candidate emotion.
    4. Asks the LLM ``π_θ2`` for calibrated probabilities {(y_k, p_k)} (T_emo).
    5. Returns the emotion with the highest probability as the final
       prediction ``y'`` (Section 3.3.1).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from main import TSVDataset, _collate, evaluate, make_emotion_predictor
from emodap import DemoRetriever
from utils import convert_emotion_json, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="ED")
    p.add_argument("--run_seed", type=int, default=1)
    p.add_argument("--data_root", default="data")
    p.add_argument("--demo_pool_json", default=None)
    p.add_argument("--proto_dir", default="ckpt/proto")
    p.add_argument("--retriever_ckpt", required=True)
    p.add_argument("--llm_dir", required=True)
    p.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--rho", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.join(args.data_root, args.dataset_name,
                            "random_splits2", f"fold_{args.run_seed}")
    test_csv = os.path.join(base_dir, "test.csv")
    test_label_json = os.path.join(base_dir, "test_label.json")
    demo_pool_path = args.demo_pool_json or os.path.join(
        args.data_root, args.dataset_name, "uns_candis.json")

    prototypes = torch.load(os.path.join(args.proto_dir, "prototypes.pt"),
                            weights_only=True).to(device)
    H_demo = torch.load(os.path.join(args.proto_dir, "H_demo.pt"),
                        weights_only=True).to(device)
    with open(os.path.join(args.proto_dir, "label2idx.json"), "r", encoding="utf-8") as f:
        full_label2idx: Dict[str, int] = json.load(f)

    test_label2idx = convert_emotion_json(test_label_json)
    test_label_set = list(test_label2idx.keys())

    with open(demo_pool_path, "r", encoding="utf-8") as f:
        raw_pool = json.load(f)
    name_pool: Dict[str, List[str]] = {}
    for key, demos in raw_pool.items():
        try:
            emo_name = list(test_label2idx.keys())[int(key)]
        except (ValueError, IndexError):
            emo_name = key.lower()
        name_pool[emo_name] = list(demos)

    retriever = DemoRetriever(args.encoder).to(device)
    ckpt = torch.load(args.retriever_ckpt, map_location="cpu", weights_only=True)
    retriever.load_state_dict(ckpt["retriever"])
    retriever.eval()

    predictor = make_emotion_predictor(args.llm_dir, device=device)

    test_loader = DataLoader(TSVDataset(test_csv), batch_size=args.batch_size,
                             shuffle=False, collate_fn=_collate)

    metrics = evaluate(test_loader, retriever, predictor, prototypes, H_demo,
                       full_label2idx, test_label_set, name_pool,
                       rho=args.rho, device=device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
