"""Bulletproof Kaggle inference script for the NVIDIA Nemotron reasoning challenge.

Two-part structure:
1) Environment, data validation, prompting, retrieval fallback, and model loading.
2) Inference loop, answer post-processing, and submission file creation.

Designed for Kaggle notebooks with local-only model loading and graceful fallbacks.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import random
import re
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# PART 1 — Setup, validation, prompting, fallback, model load
# ============================================================

DATA_PATH = "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge"
MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
OUTPUT_PATH = "/kaggle/working/submission.csv"


@dataclass(frozen=True)
class Config:
    data_path: str = DATA_PATH
    model_path: str = MODEL_PATH
    output_path: str = OUTPUT_PATH
    seed: int = 42
    prompt_max_length: int = 1536
    max_new_tokens: int = 96
    few_shot_k: int = 3
    retrieval_pool_size: int = 300
    tfidf_max_features: int = 20_000
    verbose: bool = True


CFG = Config()


SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
if SKLEARN_AVAILABLE:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_global_seed(CFG.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(*parts: object) -> None:
    if CFG.verbose:
        print(*parts)


class KagglePathError(FileNotFoundError):
    """Raised when an expected Kaggle dataset or model path is missing."""


class InferenceRuntimeError(RuntimeError):
    """Raised for recoverable inference-time failures."""


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    text_col_train: str
    label_col_train: Optional[str]
    text_col_test: str


@dataclass
class RetrieverBundle:
    vectorizer: "TfidfVectorizer"
    matrix: object
    labels: List[str]
    corpus: List[str]


def ensure_path_exists(path_str: str, description: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise KagglePathError(f"Missing {description}: {path}")
    return path


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log(f"[WARN] CSV not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log(f"[WARN] Failed to read {path}: {exc}")
        return pd.DataFrame()


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if df is None or df.empty:
        return None, None

    text_candidates = []
    label_candidates = []
    for column in df.columns:
        lower = str(column).lower()
        if any(token in lower for token in ("question", "prompt", "problem", "input", "text")):
            text_candidates.append(column)
        if any(token in lower for token in ("answer", "label", "target", "output")):
            label_candidates.append(column)

    text_col = text_candidates[0] if text_candidates else df.columns[0]
    label_col = label_candidates[0] if label_candidates else (df.columns[-1] if len(df.columns) > 1 else None)
    if text_col == label_col:
        label_col = None
    return str(text_col), (str(label_col) if label_col is not None else None)


def load_competition_data(data_path: str) -> DatasetBundle:
    base = ensure_path_exists(data_path, "competition data path")
    train_df = safe_read_csv(base / "train.csv")
    test_df = safe_read_csv(base / "test.csv")
    sample_submission = safe_read_csv(base / "sample_submission.csv")

    if test_df.empty:
        raise KagglePathError(f"test.csv is missing or empty under {base}")

    text_col_train, label_col_train = detect_columns(train_df)
    text_col_test, _ = detect_columns(test_df)

    if text_col_test is None:
        raise KagglePathError("Unable to detect the test text column.")

    log(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    log(f"Detected columns -> train text: {text_col_train}, train label: {label_col_train}, test text: {text_col_test}")

    return DatasetBundle(
        train=train_df,
        test=test_df,
        sample_submission=sample_submission,
        text_col_train=text_col_train or text_col_test,
        label_col_train=label_col_train,
        text_col_test=text_col_test,
    )


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_k_shot_examples(df: pd.DataFrame, text_col: str, label_col: Optional[str], k: int) -> List[Tuple[str, str]]:
    if df.empty or not label_col or text_col not in df.columns or label_col not in df.columns:
        return []

    sample = df[[text_col, label_col]].dropna().copy()
    if sample.empty:
        return []

    sample[text_col] = sample[text_col].map(normalize_text)
    sample[label_col] = sample[label_col].map(normalize_text)
    sample = sample[(sample[text_col] != "") & (sample[label_col] != "")]
    if sample.empty:
        return []

    sample["text_len"] = sample[text_col].str.len()
    sample = sample.sort_values("text_len", kind="stable").head(max(CFG.retrieval_pool_size, k))

    examples: List[Tuple[str, str]] = []
    seen_questions = set()
    for row in sample.itertuples(index=False):
        question = getattr(row, text_col)
        answer = getattr(row, label_col)
        if question in seen_questions:
            continue
        seen_questions.add(question)
        examples.append((question, answer))
        if len(examples) >= k:
            break
    return examples


def build_retriever(df: pd.DataFrame, text_col: str, label_col: Optional[str]) -> Optional[RetrieverBundle]:
    if not SKLEARN_AVAILABLE or df.empty or not label_col:
        return None
    if text_col not in df.columns or label_col not in df.columns:
        return None

    sample = df[[text_col, label_col]].dropna().copy()
    if sample.empty:
        return None

    sample[text_col] = sample[text_col].map(normalize_text)
    sample[label_col] = sample[label_col].map(normalize_text)
    sample = sample[(sample[text_col] != "") & (sample[label_col] != "")]
    if sample.empty:
        return None

    sample["text_len"] = sample[text_col].str.len()
    sample = sample.sort_values("text_len", kind="stable").head(CFG.retrieval_pool_size)

    corpus = sample[text_col].tolist()
    labels = sample[label_col].tolist()
    vectorizer = TfidfVectorizer(max_features=CFG.tfidf_max_features, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    log(f"[INFO] TF-IDF fallback retriever ready with {len(corpus)} examples.")
    return RetrieverBundle(vectorizer=vectorizer, matrix=matrix, labels=labels, corpus=corpus)


def retrieve_label(question: str, retriever: Optional[RetrieverBundle]) -> str:
    if retriever is None:
        return "MODEL_NOT_AVAILABLE"
    query = retriever.vectorizer.transform([normalize_text(question)])
    scores = cosine_similarity(query, retriever.matrix)
    best_idx = int(scores.argmax())
    return retriever.labels[best_idx]


def build_prompt_few_shot(question: str, shots: Sequence[Tuple[str, str]]) -> str:
    blocks: List[str] = [
        "You are solving a reasoning challenge.",
        "Read the examples, reason carefully, and end with a short final answer line.",
    ]
    for idx, (shot_q, shot_a) in enumerate(shots, start=1):
        blocks.append(
            f"Example {idx}\nQuestion:\n{shot_q}\nAnswer:\n{shot_a}"
        )
    blocks.append(
        f"Now solve the next problem.\nQuestion:\n{normalize_text(question)}\nAnswer:"
    )
    return "\n\n".join(blocks)


def build_prompt_final_only(question: str) -> str:
    return (
        "Solve the problem carefully, but output only the final answer.\n"
        f"Problem:\n{normalize_text(question)}\n"
        "Final answer:"
    )


def create_mamba_stub() -> None:
    if "mamba_ssm" in sys.modules:
        return

    log("[INFO] Injecting mamba_ssm compatibility stub for import-time stability.")
    mamba_module = types.ModuleType("mamba_ssm")
    ops_module = types.ModuleType("mamba_ssm.ops")
    triton_module = types.ModuleType("mamba_ssm.ops.triton")
    layernorm_module = types.ModuleType("mamba_ssm.ops.triton.layernorm_gated")

    def rmsnorm_fn(*args, **kwargs):
        raise NotImplementedError("mamba_ssm runtime op unavailable in this Kaggle environment.")

    class RMSNorm:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("mamba_ssm RMSNorm unavailable in this Kaggle environment.")

    layernorm_module.rmsnorm_fn = rmsnorm_fn
    layernorm_module.RMSNorm = RMSNorm

    sys.modules["mamba_ssm"] = mamba_module
    sys.modules["mamba_ssm.ops"] = ops_module
    sys.modules["mamba_ssm.ops.triton"] = triton_module
    sys.modules["mamba_ssm.ops.triton.layernorm_gated"] = layernorm_module


create_mamba_stub()


class NemotronInferenceEngine:
    def __init__(self, model_path: str, retriever: Optional[RetrieverBundle], shots: Sequence[Tuple[str, str]]) -> None:
        self.model_path = ensure_path_exists(model_path, "local model path")
        self.retriever = retriever
        self.shots = list(shots)
        self.tokenizer = None
        self.model = None
        self.model_device = torch.device(DEVICE)

    def load(self) -> None:
        try:
            log(f"Loading tokenizer from {self.model_path} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True,
                use_fast=True,
            )

            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            log(f"Loading model with dtype={dtype} ...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            if not torch.cuda.is_available():
                self.model.to(self.model_device)
            self.model.eval()
            self.model_device = next(self.model.parameters()).device
            log(f"Model loaded successfully on {self.model_device}.")
        except Exception as exc:
            log(f"[WARN] Model loading failed: {exc}")
            self.tokenizer = None
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    def postprocess_answer(self, text: str) -> str:
        text = normalize_text(text)
        if not text:
            return ""

        markers = ["Final answer:", "Answer:"]
        for marker in markers:
            if marker in text:
                text = text.split(marker)[-1].strip()

        text = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text[:300].strip()

        short_lines = [line for line in lines if len(line) <= 280]
        if short_lines:
            return short_lines[-1]
        return min(lines, key=len)[:300].strip()

    def _generate_once(self, prompt: str, max_new_tokens: int) -> str:
        if not self.is_ready:
            raise InferenceRuntimeError("Model is not loaded.")

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=CFG.prompt_max_length,
        )
        encoded = {key: value.to(self.model_device) for key, value in encoded.items()}

        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output[0][encoded["input_ids"].shape[-1]:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self.postprocess_answer(decoded)

    def answer_question(self, question: str) -> str:
        normalized_question = normalize_text(question)
        if not normalized_question:
            return retrieve_label("", self.retriever)

        if not self.is_ready:
            return retrieve_label(normalized_question, self.retriever)

        prompts = []
        if self.shots:
            prompts.append(build_prompt_few_shot(normalized_question, self.shots))
        prompts.append(build_prompt_final_only(normalized_question))

        candidates: List[str] = []
        for prompt in prompts:
            try:
                candidate = self._generate_once(prompt, CFG.max_new_tokens)
                if candidate:
                    candidates.append(candidate)
            except NotImplementedError as exc:
                log(f"[WARN] Native op unavailable during generation: {exc}")
                return retrieve_label(normalized_question, self.retriever)
            except torch.cuda.OutOfMemoryError:
                log("[WARN] CUDA OOM during generation. Clearing cache and retrying via fallback.")
                torch.cuda.empty_cache()
                gc.collect()
                return retrieve_label(normalized_question, self.retriever)
            except Exception as exc:
                log(f"[WARN] Generation failed for one prompt: {exc}")

        if not candidates:
            return retrieve_label(normalized_question, self.retriever)
        if len(set(candidates)) == 1:
            return candidates[0]
        return sorted(candidates, key=lambda item: (len(item), item))[0]


# ============================================================
# PART 2 — Inference, submission creation, and execution flow
# ============================================================

def build_submission_frame(bundle: DatasetBundle, answers: Sequence[str]) -> pd.DataFrame:
    if not answers:
        return pd.DataFrame(columns=["id", "answer"])

    if not bundle.sample_submission.empty and set(bundle.sample_submission.columns) >= {"id", "answer"}:
        submission = bundle.sample_submission.copy()
        target_length = min(len(submission), len(answers))
        submission = submission.iloc[:target_length].copy()
        submission["answer"] = list(answers[:target_length])
        return submission[["id", "answer"]]

    if "id" in bundle.test.columns:
        ids = bundle.test["id"].tolist()
    else:
        ids = bundle.test.index.tolist()
    return pd.DataFrame({"id": ids[: len(answers)], "answer": list(answers)})


def run_inference() -> pd.DataFrame:
    bundle = load_competition_data(CFG.data_path)
    shots = build_k_shot_examples(bundle.train, bundle.text_col_train, bundle.label_col_train, CFG.few_shot_k)
    retriever = build_retriever(bundle.train, bundle.text_col_train, bundle.label_col_train)

    engine = NemotronInferenceEngine(CFG.model_path, retriever=retriever, shots=shots)
    engine.load()

    questions = bundle.test[bundle.text_col_test].fillna("").astype(str).tolist()
    answers: List[str] = []

    log(f"Running inference for {len(questions)} test rows...")
    for question in tqdm(questions, desc="Generating", total=len(questions)):
        try:
            answers.append(engine.answer_question(question))
        except Exception as exc:
            log(f"[WARN] Top-level failure for one row: {exc}")
            answers.append(retrieve_label(question, retriever))

    submission = build_submission_frame(bundle, answers)
    output_path = Path(CFG.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    log(f"Saved submission to: {output_path}")
    log(submission.head())
    return submission


if __name__ == "__main__":
    try:
        run_inference()
    except KagglePathError as exc:
        print(f"[FATAL] {exc}")
    except Exception as exc:
        print(f"[FATAL] Unexpected error: {exc}")
