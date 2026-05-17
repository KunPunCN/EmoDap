"""Core neural modules of the EmoDap framework.

This file defines the two collaborative agents introduced in the paper
*EmoDap: Dual-Agent Prompt Retrieval Framework for Zero-Shot Fine-Grained
Emotion Recognition* (Section 3.3) together with the prototype-encoder used
during precomputation (Section 3.2.2) and the GRPO objective (Section 3.3.2).

Notation follows the paper:
    * ``V    = {v_y}``                 emotion prototypes
    * ``H_demo``                       prototype-space embeddings of demos
    * ``Y_cand = Top-ρ(cos(Enc(x), V))`` dynamic candidate set
    * ``π_θ1`` (Agent_demo)            demonstration retriever
    * ``π_θ2`` (Agent_emo)             emotion predictor (LLM, frozen)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Prototype Encoder (Section 3.2.2)
# ---------------------------------------------------------------------------
class PrototypeEncoder(nn.Module):
    """Encoder used both for sentences and for emotion definitions.

    Implements ``v_y = Enc(y)`` and ``h_j = Enc(x_j)`` from Eq. (3) using the
    ``[CLS]`` token of the last hidden layer.  Trained with the InfoNCE-style
    contrastive loss ``L_proto`` defined in Eq. (3).
    """

    def __init__(self, pretrain_model_name_or_path: str, max_seq_len: int = 128):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrain_model_name_or_path)
        self.encoder = AutoModel.from_pretrained(pretrain_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name_or_path)
        self.max_seq_len = max_seq_len
        self.hidden_size = self.config.hidden_size

    def encode(self, texts: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        enc = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        ).to(device)
        out = self.encoder(**enc).last_hidden_state[:, 0]  # [CLS]
        return out

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]


def proto_contrastive_loss(
    sentence_repr: torch.Tensor,    # [B, d]
    proto_repr: torch.Tensor,        # [|Y_s|, d]
    labels: torch.Tensor,            # [B] long, indices into proto_repr
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE-style contrastive objective ``L_proto`` (Eq. 3 of the paper)."""
    sentence_repr = F.normalize(sentence_repr, dim=-1)
    proto_repr = F.normalize(proto_repr, dim=-1)
    logits = sentence_repr @ proto_repr.t() / temperature  # [B, |Y_s|]
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Agent_demo (Demonstration Retriever, π_θ1) -- Eq. (5)
# ---------------------------------------------------------------------------
class DemoRetriever(nn.Module):
    """``π_θ1``: choose one demonstration index per candidate emotion.

    For each candidate emotion ``y`` and its ``K`` demonstrations stored in
    ``H_demo[y] ∈ R^{K×d}``, the retriever produces a logit
    ``cos(h_agent, H_demo[y, k]) / τ``.  During training we keep the full
    distribution so that we can sample a *group* of trajectories for GRPO; at
    inference we take the arg-max as in Eq. (5) of the paper.
    """

    def __init__(self, pretrain_model_name_or_path: str, max_seq_len: int = 128,
                 temperature: float = 0.07):
        super().__init__()
        self.encoder = PrototypeEncoder(pretrain_model_name_or_path, max_seq_len)
        self.hidden_size = self.encoder.hidden_size
        # learnable projection that lets the retriever specialise away from the
        # frozen prototype space if necessary
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)
        self.temperature = temperature

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        h = self.encoder.encode(texts, device=device)
        return self.proj(h)

    def score(
        self,
        sentence_repr: torch.Tensor,    # [B, d]
        H_demo_subset: torch.Tensor,    # [B, |Y_cand|, K, d]
    ) -> torch.Tensor:
        """Return per-(emotion, k) log-probabilities, shape ``[B, |Y_cand|, K]``."""
        s = F.normalize(sentence_repr, dim=-1).unsqueeze(1).unsqueeze(1)  # [B,1,1,d]
        d = F.normalize(H_demo_subset, dim=-1)                            # [B,Yc,K,d]
        logits = (s * d).sum(-1) / self.temperature                       # [B,Yc,K]
        return F.log_softmax(logits, dim=-1)

    def sample_group(
        self,
        log_probs: torch.Tensor,    # [B, |Y_cand|, K]
        group_size: int,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``G`` trajectories ``z`` per example.

        Returns ``(indices, logp)`` of shapes ``[B, G, |Y_cand|]`` and
        ``[B, G, |Y_cand|]``.  ``logp`` keeps the per-emotion log-prob so that
        the joint log-probability of a trajectory is its sum along the last
        dim.  When ``greedy`` is true the arg-max is returned (used at eval
        time, Eq. 5 of the paper).
        """
        B, Yc, K = log_probs.shape
        if greedy:
            idx = log_probs.argmax(dim=-1, keepdim=True)        # [B, Yc, 1]
            idx = idx.expand(B, Yc, group_size).transpose(1, 2)  # [B, G, Yc]
            lp = log_probs.gather(-1, log_probs.argmax(dim=-1, keepdim=True)).squeeze(-1)
            lp = lp.unsqueeze(1).expand(B, group_size, Yc)
            return idx, lp

        flat = log_probs.exp().reshape(B * Yc, K)
        flat = flat.clamp_min(1e-12)
        flat = flat / flat.sum(-1, keepdim=True)
        sampled = torch.multinomial(flat, group_size, replacement=True)  # [B*Yc, G]
        sampled = sampled.view(B, Yc, group_size).permute(0, 2, 1)        # [B, G, Yc]
        lp = log_probs.gather(-1, sampled.permute(0, 2, 1)).permute(0, 2, 1)
        return sampled, lp


# ---------------------------------------------------------------------------
# Agent_emo wrapper (Section 3.3.1) -- Eq. (4) / Eq. (T_emo)
# ---------------------------------------------------------------------------
class EmotionPredictor:
    """Wrapper around the frozen LLM that performs the predictor's actions.

    The predictor is *not* a learned module – it is the open-source LLM
    (Llama-3-8B / Qwen3-8B) prompted with template ``T_emo``.  We expose two
    operations from the paper:

    * :meth:`build_candidate_set`  – Eq. (4): Top-ρ filtering by prototype
      similarity in the *candidate*-space (seen ∪ unseen emotions).
    * :meth:`predict_distribution` – calls the LLM with ``T_emo`` and parses
      ``{(y_k, p_k)}``.
    """

    def __init__(self, pipeline, terminators, system_prompt: Optional[str] = None,
                 max_new_tokens: int = 256):
        self.pipeline = pipeline
        self.terminators = terminators
        self.system_prompt = system_prompt or (
            "You are a precise emotion analysis expert. Strictly follow the output format."
        )
        self.max_new_tokens = max_new_tokens

    # ---------- Eq. (4): prototype-guided dynamic candidate filtering ----
    @staticmethod
    def build_candidate_set(
        sentence_repr: torch.Tensor,            # [B, d]
        prototypes: torch.Tensor,               # [|Y|, d]
        rho: float = 0.4,
    ) -> List[List[int]]:
        """Top-ρ filtering: keep the emotions whose prototype cosine
        similarity rank within the top ``ceil(ρ·|Y|)``.
        """
        s = F.normalize(sentence_repr, dim=-1)
        v = F.normalize(prototypes, dim=-1)
        sims = s @ v.t()                       # [B, |Y|]
        n_cand = max(1, int(round(rho * prototypes.size(0))))
        topk = sims.topk(n_cand, dim=-1).indices  # [B, n_cand]
        return topk.tolist()

    # ---------- T_emo prompt construction ---------------------------------
    @staticmethod
    def build_prompt(sentence: str, candidate_emotions: List[str],
                     demos: Dict[str, str]) -> str:
        """Construct the ``T_emo`` template described in Section 3.3.1."""
        demo_block_parts = []
        for i, emo in enumerate(candidate_emotions):
            demo_text = demos.get(emo, "")
            demo_block_parts.append(f"\nEmotion {i + 1}: {emo}\n Demonstrations: {demo_text}")
        demo_block = "".join(demo_block_parts)
        return (
            "Given a sentence, please determine the emotion it conveys.\n"
            f"Here are demonstrations of these emotions:{demo_block}\n"
            "Please understand the meaning of each emotion through these demonstrations.\n"
            f"Sentence: {sentence}\n"
            f"Choose your answer from {candidate_emotions}.\n"
            "Output format: emotion: probability%. (probability must be precise to "
            "two decimal places, no extra explanation; one emotion per line for each "
            "candidate)."
        )

    def predict_distribution(self, sentence: str, candidate_emotions: List[str],
                             demos: Dict[str, str]) -> List[Tuple[str, float]]:
        from utils import parse_emotion_probability

        prompt = self.build_prompt(sentence, candidate_emotions, demos)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            pad_token_id=self.pipeline.tokenizer.pad_token_id,
            do_sample=False,
            temperature=0.0,
        )
        text = outputs[0]["generated_text"][-1]["content"].lower()
        parsed = parse_emotion_probability(text, candidate_emotions)
        if not parsed:
            # fall back to a uniform distribution across the candidate set
            uniform = 1.0 / max(1, len(candidate_emotions))
            parsed = [(e, uniform) for e in candidate_emotions]
        return parsed


# ---------------------------------------------------------------------------
# Prototype-aware reward (Eq. 6) and GRPO loss (Section 3.3.2)
# ---------------------------------------------------------------------------
def prototype_aware_reward(
    distribution: List[Tuple[str, float]],
    gold_emotion: str,
    prototypes: torch.Tensor,           # [|Y|, d]
    label2idx: Dict[str, int],
) -> float:
    """Compute ``r^{(g)} = Σ_k cos(v_{y*}, v_{y_k}) · p_k`` (Eq. 6)."""
    if gold_emotion not in label2idx:
        return 0.0
    v = F.normalize(prototypes, dim=-1)
    v_star = v[label2idx[gold_emotion]]
    r = 0.0
    for emo, prob in distribution:
        if emo not in label2idx:
            continue
        sim = torch.dot(v_star, v[label2idx[emo]]).item()
        r += sim * prob
    return float(r)


def grpo_loss(
    traj_logp: torch.Tensor,    # [B, G] sum of per-emotion log-probs of θ_1
    rewards: torch.Tensor,      # [B, G]
    epsilon: float = 0.2,
    kl_coef: float = 0.0,
    ref_logp: Optional[torch.Tensor] = None,    # [B, G]
) -> torch.Tensor:
    """Group-Relative Policy Optimisation objective.

    Implements the simplified form: ``L = -E[ A * logπ ] + β·KL(π‖π_ref)`` with
    group-normalised advantages ``A = (r - mean(r)) / std(r)``.  The PPO-style
    clipping is applied for stability when ``ref_logp`` is provided.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
    advantage = (rewards - mean) / std        # [B, G]

    if ref_logp is None:
        loss = -(advantage.detach() * traj_logp).mean()
    else:
        ratio = (traj_logp - ref_logp.detach()).exp()
        unclipped = ratio * advantage.detach()
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage.detach()
        loss = -torch.min(unclipped, clipped).mean()
        if kl_coef > 0:
            kl = (ref_logp.detach() - traj_logp).mean()
            loss = loss + kl_coef * kl
    return loss
