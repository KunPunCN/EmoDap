"""Utility helpers for the EmoDap framework."""

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_emotion_json(json_path: str) -> Dict[str, int]:
    """Read an emotion-definition JSON file and return {emotion_name: idx}."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for k, (_, value) in enumerate(data.items()):
        emotion = value.split(":")[0].strip().lower()
        result[emotion] = int(k)
    return result


def convert_emotion_definitions(json_path: str) -> Dict[str, str]:
    """Return {emotion_name: full_definition_text} (used for prototype encoding)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for _, value in data.items():
        name = value.split(":")[0].strip().lower()
        result[name] = value.strip()
    return result


def convert_emotion_json_lib(label_json_path: str, candi_json_path: str
                             ) -> Tuple[Dict[str, int], List[List[str]]]:
    """Build {emotion: idx} together with the per-emotion demonstration library.

    The candidate file ``candi_json_path`` is a JSON whose keys are emotion
    indices (matching ``label_json_path``) and values are lists of demo strings.
    """
    label2idx = convert_emotion_json(label_json_path)
    with open(candi_json_path, "r", encoding="utf-8") as f:
        candi_dict = json.load(f)

    lib: List[List[str]] = []
    for _, idx in label2idx.items():
        # keys may be int / str depending on how the JSON was saved
        key_candidates = (str(idx), idx)
        sentences: List[str] = []
        for k in key_candidates:
            if k in candi_dict:
                sentences = candi_dict[k]
                break
        lib.append(list(sentences))
    return label2idx, lib


def parse_emotion_probability(text: str, emotion_set: List[str]) -> List[Tuple[str, float]]:
    """Parse a string of the form ``emotion: 35.21%`` (potentially several lines).

    Returns a list of (emotion_name, probability) pairs filtered to ``emotion_set``.
    Probabilities are normalised to ``[0, 1]``.
    """
    import re

    text = text.lower()
    pattern = re.compile(r"([a-zA-Z][a-zA-Z\- ]*)\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*%?")
    found = []
    seen_emotions = set()
    for match in pattern.finditer(text):
        emo = match.group(1).strip()
        # the regex may absorb leading words; keep the *last* token group
        emo_token = emo.split()[-1] if " " in emo else emo
        prob = float(match.group(2))
        if prob > 1.0:  # interpret as percentage
            prob = prob / 100.0
        prob = max(0.0, min(1.0, prob))
        for cand in (emo, emo_token):
            if cand in emotion_set and cand not in seen_emotions:
                found.append((cand, prob))
                seen_emotions.add(cand)
                break
    return found
