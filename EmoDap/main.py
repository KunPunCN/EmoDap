"""Phase-2 of EmoDap: Dual-Agent GRPO Joint Training (Section 3.3).

Given the *frozen* assets produced by Phase-1 –

    * Demonstration pool ``D_demo``      (``demo_pool.json``)
    * Prototype encoder & ``V``, ``H_demo`` (from ``prototype_training.py``)

we jointly fine-tune the demonstration retriever ``π_θ1`` (only learnable
component) so that it picks demonstrations that maximise the prototype-aware
reward (Eq. 6) returned by the LLM-based emotion predictor ``π_θ2``.

Pipeline per training step (Algorithm in Section 3.3.2):

    1. For batch ``x``: build ``Y_cand`` = Top-ρ(cos(Enc(x), V))             (Eq. 4)
    2. Score every (emotion in Y_cand, demo k) with ``π_θ1``                 (Eq. 5)
    3. Sample G trajectories ``z^{(g)}`` -> demonstrations ``S^{(g)}``
    4. Run ``π_θ2`` (LLM) under template T_emo to obtain {(y_k, p_k)}^{(g)}
    5. Compute ``r^{(g)}``                                                   (Eq. 6)
    6. GRPO update on ``π_θ1`` with group-relative advantages
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from emodap import (
    DemoRetriever,
    EmotionPredictor,
    PrototypeEncoder,
    grpo_loss,
    prototype_aware_reward,
)
from utils import convert_emotion_definitions, convert_emotion_json, set_seed


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class TSVDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, delimiter="\t")
        if "text" not in df.columns or "label" not in df.columns:
            df.columns = ["text", "label"][: len(df.columns)]
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(str).str.lower().tolist()

    def __len__(self): return len(self.texts)

    def __getitem__(self, i): return {"text": self.texts[i], "label": self.labels[i]}


def _collate(batch):
    return {
        "text": [b["text"] for b in batch],
        "label": [b["label"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_emotion_predictor(model_dir: str, device: torch.device) -> EmotionPredictor:
    """Load the open-source LLM (Llama3-8B / Qwen3-8B) used as π_θ2."""
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    if pipeline.tokenizer.pad_token_id is None:
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    eot = pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    terminators = [pipeline.tokenizer.eos_token_id]
    if eot is not None and eot != pipeline.tokenizer.unk_token_id:
        terminators.append(eot)
    return EmotionPredictor(pipeline, terminators)


def build_demos_for_trajectory(
    candidate_emotion_names: List[str],
    demo_indices_for_traj: List[int],
    demo_pool: Dict[str, List[str]],
) -> Dict[str, str]:
    """Materialise the dictionary {emotion: demo_text} used in the prompt."""
    out = {}
    for emo, k in zip(candidate_emotion_names, demo_indices_for_traj):
        demos = demo_pool.get(emo, [])
        if not demos:
            continue
        k = min(int(k), len(demos) - 1)
        out[emo] = demos[k]
    return out


# ---------------------------------------------------------------------------
# Evaluation – greedy retrieval (Eq. 5 arg-max) + LLM prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    loader: DataLoader,
    retriever: DemoRetriever,
    predictor: EmotionPredictor,
    prototypes: torch.Tensor,
    H_demo: torch.Tensor,
    full_label2idx: Dict[str, int],
    test_label_set: List[str],
    demo_pool: Dict[str, List[str]],
    rho: float,
    device: torch.device,
) -> Dict[str, float]:
    retriever.eval()
    idx2name = {i: e for e, i in full_label2idx.items()}

    preds: List[str] = []
    golds: List[str] = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        sentences = batch["text"]
        labels = batch["label"]

        sent_repr = retriever.encode(sentences)                              # [B, d]
        # restrict the prototype-similarity search to the *test* label space
        test_idx_tensor = torch.tensor(
            [full_label2idx[e] for e in test_label_set], device=device,
        )
        proto_test = prototypes[test_idx_tensor]                             # [|Y_t|, d]

        cand_lists = EmotionPredictor.build_candidate_set(sent_repr, proto_test, rho=rho)

        for i, cand_local in enumerate(cand_lists):
            cand_global = [int(test_idx_tensor[c].item()) for c in cand_local]
            cand_names = [idx2name[g] for g in cand_global]
            H_sub = H_demo[cand_global].unsqueeze(0).to(device)              # [1, Yc, K, d]
            log_probs = retriever.score(sent_repr[i:i + 1], H_sub)           # [1, Yc, K]
            best_k = log_probs.argmax(dim=-1)[0].tolist()                    # [Yc]

            demos = build_demos_for_trajectory(cand_names, best_k, demo_pool)
            distribution = predictor.predict_distribution(sentences[i], cand_names, demos)
            if distribution:
                pred = max(distribution, key=lambda kv: kv[1])[0]
            else:
                pred = cand_names[0]

            preds.append(pred)
            golds.append(labels[i])

    weighted = f1_score(golds, preds, average="weighted")
    macro = f1_score(golds, preds, average="macro")
    acc = accuracy_score(golds, preds)
    print(f"[eval] acc={acc:.4f}  macro-F1={macro:.4f}  weighted-F1={weighted:.4f}")
    retriever.train()
    return {"acc": acc, "macro_f1": macro, "weighted_f1": weighted}


# ---------------------------------------------------------------------------
# GRPO training step
# ---------------------------------------------------------------------------
def grpo_step(
    sentences: List[str],
    labels: List[str],
    retriever: DemoRetriever,
    predictor: EmotionPredictor,
    prototypes: torch.Tensor,
    H_demo: torch.Tensor,
    full_label2idx: Dict[str, int],
    test_label_set: List[str],
    demo_pool: Dict[str, List[str]],
    args,
    device: torch.device,
) -> Dict[str, float]:
    """One micro-batch of GRPO joint training."""
    idx2name = {i: e for e, i in full_label2idx.items()}

    sent_repr = retriever.encode(sentences)                                  # [B, d]
    test_idx = torch.tensor([full_label2idx[e] for e in test_label_set], device=device)
    proto_test = prototypes[test_idx]                                         # [|Y_t|, d]

    cand_lists = EmotionPredictor.build_candidate_set(sent_repr, proto_test, rho=args.rho)

    # We process examples one-by-one because |Y_cand| may differ across rows
    batch_loss = 0.0
    batch_reward = 0.0
    n_valid = 0

    for i, cand_local in enumerate(cand_lists):
        cand_global = [int(test_idx[c].item()) for c in cand_local]
        cand_names = [idx2name[g] for g in cand_global]
        if not cand_names:
            continue

        H_sub = H_demo[cand_global].unsqueeze(0).to(device)                  # [1, Yc, K, d]
        log_probs = retriever.score(sent_repr[i:i + 1], H_sub)               # [1, Yc, K]
        sampled, lp = retriever.sample_group(log_probs, group_size=args.G)    # [1, G, Yc]
        traj_logp = lp.sum(dim=-1)                                           # [1, G]

        rewards = []
        for g in range(args.G):
            demos = build_demos_for_trajectory(
                cand_names, sampled[0, g].tolist(), demo_pool,
            )
            distribution = predictor.predict_distribution(
                sentences[i], cand_names, demos,
            )
            r = prototype_aware_reward(
                distribution, labels[i], prototypes, full_label2idx,
            )
            rewards.append(r)
        rewards_t = torch.tensor(rewards, device=device).unsqueeze(0)        # [1, G]

        loss = grpo_loss(traj_logp, rewards_t, epsilon=args.clip_eps)
        loss.backward()

        batch_loss += float(loss.item())
        batch_reward += float(rewards_t.mean().item())
        n_valid += 1

    n_valid = max(1, n_valid)
    return {"loss": batch_loss / n_valid, "reward": batch_reward / n_valid}


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="ED")
    p.add_argument("--run_seed", type=int, default=1)
    p.add_argument("--data_root", default="data")
    p.add_argument("--demo_pool_json", default=None,
                   help="Path to D_demo JSON; defaults to <data>/<DATASET>/uns_candis.json")
    p.add_argument("--proto_dir", default="ckpt/proto",
                   help="Directory holding prototypes.pt / H_demo.pt / prototype_encoder.pt")
    p.add_argument("--llm_dir", required=True,
                   help="HF path of the open-source LLM used as Agent_emo (Llama3-8B / Qwen3-8B)")
    p.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--rho", type=float, default=0.4, help="Top-ρ filtering ratio (Eq. 4)")
    p.add_argument("--G", type=int, default=8, help="GRPO group size")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--eval_max_batches", type=int, default=0,
                   help="If >0 limits the number of eval batches – useful for smoke tests")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default="0")
    p.add_argument("--save_path", default="ckpt/emodap_retriever.pt")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.join(args.data_root, args.dataset_name,
                            "random_splits2", f"fold_{args.run_seed}")
    train_csv = os.path.join(base_dir, "train_see.csv")
    test_csv = os.path.join(base_dir, "test.csv")
    test_label_json = os.path.join(base_dir, "test_label.json")

    demo_pool_path = args.demo_pool_json or os.path.join(
        args.data_root, args.dataset_name, "uns_candis.json",
    )

    # ----- frozen Phase-1 assets -----------------------------------------
    proto_dir = args.proto_dir
    prototypes = torch.load(os.path.join(proto_dir, "prototypes.pt"),
                            weights_only=True).to(device)
    H_demo = torch.load(os.path.join(proto_dir, "H_demo.pt"),
                        weights_only=True).to(device)
    with open(os.path.join(proto_dir, "label2idx.json"), "r", encoding="utf-8") as f:
        full_label2idx: Dict[str, int] = json.load(f)

    test_label2idx = convert_emotion_json(test_label_json)
    test_label_set = list(test_label2idx.keys())

    with open(demo_pool_path, "r", encoding="utf-8") as f:
        raw_pool = json.load(f)
    # Re-key D_demo by emotion *name* so the prompt construction is trivial
    name_pool: Dict[str, List[str]] = {}
    for key, demos in raw_pool.items():
        try:
            emo_name = list(test_label2idx.keys())[int(key)]
        except (ValueError, IndexError):
            emo_name = key.lower()
        name_pool[emo_name] = list(demos)

    # ----- π_θ1 retriever ------------------------------------------------
    retriever = DemoRetriever(args.encoder).to(device)
    enc_path = os.path.join(proto_dir, "prototype_encoder.pt")
    if os.path.isfile(enc_path):
        # warm-start with the prototype encoder weights – the projection head
        # ``proj`` was initialised to identity so this is exactly equivalent
        # to starting from the pretrained encoder
        sd = torch.load(enc_path, map_location="cpu", weights_only=True)
        missing, unexpected = retriever.encoder.load_state_dict(sd, strict=False)
        print(f"[init] retriever encoder: missing={len(missing)} unexpected={len(unexpected)}")

    optim = torch.optim.AdamW(retriever.parameters(), lr=args.lr)

    # ----- π_θ2 predictor (frozen LLM) -----------------------------------
    predictor = make_emotion_predictor(args.llm_dir, device=device)

    # ----- data loaders --------------------------------------------------
    train_loader = DataLoader(TSVDataset(train_csv), batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    test_loader = DataLoader(TSVDataset(test_csv), batch_size=args.batch_size,
                             shuffle=False, collate_fn=_collate)

    # ----- training loop --------------------------------------------------
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        retriever.train()
        running_loss, running_reward, n = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"GRPO epoch {epoch}/{args.epochs}")
        t0 = time.time()
        for step, batch in enumerate(pbar, 1):
            optim.zero_grad()
            stats = grpo_step(
                batch["text"], batch["label"], retriever, predictor,
                prototypes, H_demo, full_label2idx, test_label_set, name_pool,
                args, device,
            )
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
            optim.step()
            running_loss += stats["loss"]; running_reward += stats["reward"]; n += 1
            pbar.set_postfix(loss=running_loss / n, reward=running_reward / n)

        print(f"[epoch {epoch}] avg_loss={running_loss / max(n, 1):.4f} "
              f"avg_reward={running_reward / max(n, 1):.4f} "
              f"time={time.time() - t0:.1f}s")

        metrics = evaluate(test_loader, retriever, predictor, prototypes, H_demo,
                           full_label2idx, test_label_set, name_pool,
                           rho=args.rho, device=device)
        if metrics["weighted_f1"] > best_f1:
            best_f1 = metrics["weighted_f1"]
            torch.save({"retriever": retriever.state_dict(),
                        "metrics": metrics,
                        "args": vars(args)}, args.save_path)
            print(f"[save] best wF1={best_f1:.4f} -> {args.save_path}")

    print(f"Training finished. Best weighted-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
