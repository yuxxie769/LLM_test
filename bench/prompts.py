from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer


BASE_TEXT = (
    "请阅读下面的背景信息，并基于事实给出结构化回答。"
    "要求先概括核心结论，再给出两到三条支撑理由，最后补充潜在风险。"
    "输出风格保持技术化、简洁、可复核。"
)


@lru_cache(maxsize=4)
def load_tokenizer(model_dir: str):
    return AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )


def build_prompt(target_tokens: int, model_dir: str | Path) -> str:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    tokenizer = load_tokenizer(str(model_dir))
    seed = (BASE_TEXT + "\n") * max(target_tokens // 8, 8)
    token_ids = tokenizer.encode(seed, add_special_tokens=False)

    if len(token_ids) < target_tokens:
        repeats = (target_tokens // max(len(token_ids), 1)) + 2
        token_ids = token_ids * repeats

    trimmed = token_ids[:target_tokens]
    return tokenizer.decode(trimmed, skip_special_tokens=True)
