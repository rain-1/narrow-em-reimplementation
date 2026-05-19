#!/usr/bin/env python3
"""
Build a Persona Vectors-style trait vector, steer the base model, and run eval.

The vector is the mean hidden-state difference between generations produced with
positive and negative trait system prompts from safety-research/persona_vectors.
Answers are written in the same format as eval/run_eval.py so eval/run_judge.py
can judge the run afterward.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_PERSONA_REPO = REPO_ROOT.parent / "persona_vectors"
EVAL_QUESTIONS = REPO_ROOT / "data" / "eval-questions.jsonl"
RESULTS_DIR = REPO_ROOT / "results"


def a_or_an(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text.split("/")[-1])


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_trait_data(persona_repo: Path, trait: str, version: str) -> dict:
    path = persona_repo / "data_generation" / f"trait_data_{version}" / f"{trait}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing persona trait file: {path}")
    return json.loads(path.read_text())


def load_model_and_tokenizer(model_name: str, dtype: str):
    torch_dtype = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def chat_prompt(tokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    rendered = ""
    for message in messages:
        rendered += f"{message['role'].capitalize()}: {message['content']}\n"
    return rendered + "Assistant:"


def generate_answer(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, str]:
    prompt = chat_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    answer_ids = output[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return prompt, answer


def hidden_summaries(model, tokenizer, prompt: str, answer: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    text = prompt + answer
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    prompt_last: list[torch.Tensor] = []
    response_avg: list[torch.Tensor] = []
    for hidden in outputs.hidden_states:
        prompt_last.append(hidden[:, prompt_len - 1, :].detach().float().cpu().squeeze(0))
        if hidden.shape[1] > prompt_len:
            response = hidden[:, prompt_len:, :].mean(dim=1)
        else:
            response = hidden[:, -1:, :].mean(dim=1)
        response_avg.append(response.detach().float().cpu().squeeze(0))
    return prompt_last, response_avg


def build_examples(trait_data: dict, trait: str, max_pairs: int | None, seed: int) -> list[dict]:
    examples = []
    for instruction in trait_data["instruction"]:
        for question in trait_data["questions"]:
            examples.append(
                {
                    "pos_system": f"You are {a_or_an(trait)} {trait} assistant. {instruction['pos']}",
                    "neg_system": f"You are a helpful assistant. {instruction['neg']}",
                    "question": question,
                }
            )
    random.Random(seed).shuffle(examples)
    return examples[:max_pairs] if max_pairs else examples


def create_vector(
    model,
    tokenizer,
    trait_data: dict,
    trait: str,
    vector_dir: Path,
    max_pairs: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> Path:
    vector_dir.mkdir(parents=True, exist_ok=True)
    vector_path = vector_dir / f"{trait}_response_avg_diff.pt"
    transcript_path = vector_dir / f"{trait}_vector_examples.jsonl"
    examples = build_examples(trait_data, trait, max_pairs=max_pairs, seed=seed)
    if not examples:
        raise ValueError("No persona-vector examples were built")

    pos_response: list[list[torch.Tensor]] = []
    neg_response: list[list[torch.Tensor]] = []
    pos_prompt_last: list[list[torch.Tensor]] = []
    neg_prompt_last: list[list[torch.Tensor]] = []

    with transcript_path.open("w") as transcript:
        for i, example in enumerate(tqdm(examples, desc=f"Building {trait} vector")):
            pos_messages = [
                {"role": "system", "content": example["pos_system"]},
                {"role": "user", "content": example["question"]},
            ]
            neg_messages = [
                {"role": "system", "content": example["neg_system"]},
                {"role": "user", "content": example["question"]},
            ]
            pos_prompt, pos_answer = generate_answer(
                model, tokenizer, pos_messages, max_new_tokens, temperature, top_p
            )
            neg_prompt, neg_answer = generate_answer(
                model, tokenizer, neg_messages, max_new_tokens, temperature, top_p
            )
            pos_pl, pos_ra = hidden_summaries(model, tokenizer, pos_prompt, pos_answer)
            neg_pl, neg_ra = hidden_summaries(model, tokenizer, neg_prompt, neg_answer)
            pos_prompt_last.append(pos_pl)
            neg_prompt_last.append(neg_pl)
            pos_response.append(pos_ra)
            neg_response.append(neg_ra)
            transcript.write(
                json.dumps(
                    {
                        "index": i,
                        "question": example["question"],
                        "pos_system": example["pos_system"],
                        "neg_system": example["neg_system"],
                        "pos_answer": pos_answer,
                        "neg_answer": neg_answer,
                    }
                )
                + "\n"
            )

    n_layers = len(pos_response[0])
    response_diff = torch.stack(
        [
            torch.stack([row[layer] for row in pos_response]).mean(dim=0)
            - torch.stack([row[layer] for row in neg_response]).mean(dim=0)
            for layer in range(n_layers)
        ],
        dim=0,
    )
    prompt_last_diff = torch.stack(
        [
            torch.stack([row[layer] for row in pos_prompt_last]).mean(dim=0)
            - torch.stack([row[layer] for row in neg_prompt_last]).mean(dim=0)
            for layer in range(n_layers)
        ],
        dim=0,
    )
    torch.save(response_diff, vector_path)
    torch.save(prompt_last_diff, vector_dir / f"{trait}_prompt_last_diff.pt")
    return vector_path


class ActivationSteerer:
    """Small local copy of the persona_vectors inference-time steering hook."""

    possible_layer_attrs = (
        "transformer.h",
        "encoder.layer",
        "model.layers",
        "gpt_neox.layers",
        "block",
    )

    def __init__(self, model, vector: torch.Tensor, coeff: float, layer_idx: int, positions: str):
        self.model = model
        self.coeff = float(coeff)
        self.layer_idx = layer_idx
        self.positions = positions
        self.handle = None
        param = next(model.parameters())
        self.vector = vector.to(device=param.device, dtype=param.dtype)

    def locate_layer(self):
        for path in self.possible_layer_attrs:
            current = self.model
            for part in path.split("."):
                if not hasattr(current, part):
                    break
                current = getattr(current, part)
            else:
                if hasattr(current, "__getitem__"):
                    return current[self.layer_idx]
        raise ValueError("Could not locate transformer layers for steering")

    def hook_fn(self, module, inputs, output):
        steer = self.coeff * self.vector

        def add(tensor: torch.Tensor) -> torch.Tensor:
            if self.positions == "all":
                return tensor + steer.to(tensor.device)
            steered = tensor.clone()
            if self.positions == "response":
                steered[:, -1, :] += steer.to(tensor.device)
            elif self.positions == "prompt":
                if tensor.shape[1] > 1:
                    steered += steer.to(tensor.device)
            else:
                raise ValueError("--steering-type must be one of all, prompt, response")
            return steered

        if torch.is_tensor(output):
            return add(output)
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return (add(output[0]), *output[1:])
        return output

    def __enter__(self):
        self.handle = self.locate_layer().register_forward_hook(self.hook_fn)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def run_steered_eval(
    model,
    tokenizer,
    questions: list[dict],
    out_dir: Path,
    model_name: str,
    vector_path: Path,
    layer: int,
    coef: float,
    steering_type: str,
    epochs: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_file = out_dir / "answers.jsonl"
    if answers_file.exists():
        answers_file.unlink()

    vector_bank = torch.load(vector_path, weights_only=False, map_location="cpu")
    vector = vector_bank[layer]
    layer_idx = layer - 1

    rows_total = len(questions) * epochs
    with answers_file.open("w") as out:
        with tqdm(total=rows_total, desc="Steered eval") as progress:
            for epoch in range(epochs):
                for question in questions:
                    messages = [{"role": "user", "content": question["question"]}]
                    with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer_idx, positions=steering_type):
                        _, answer = generate_answer(
                            model,
                            tokenizer,
                            messages,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    out.write(
                        json.dumps(
                            {
                                "id": question["id"],
                                "type": question.get("type"),
                                "question": question["question"],
                                "answer": answer,
                                "epoch": epoch,
                                "model": model_name,
                            }
                        )
                        + "\n"
                    )
                    progress.update(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/OLMo-3-7B-Instruct")
    parser.add_argument("--trait", default="evil")
    parser.add_argument("--persona-repo", type=Path, default=DEFAULT_PERSONA_REPO)
    parser.add_argument("--questions-file", type=Path, default=EVAL_QUESTIONS)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--vector-dir", type=Path, default=None)
    parser.add_argument("--vector-path", type=Path, default=None)
    parser.add_argument("--force-vector", action="store_true")
    parser.add_argument("--max-vector-pairs", type=int, default=None)
    parser.add_argument("--layer", type=int, default=20, help="1-indexed layer, matching persona_vectors")
    parser.add_argument("--coef", type=float, default=2.0)
    parser.add_argument("--steering-type", choices=["all", "prompt", "response"], default="response")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--vector-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    run_id = args.run_id or (
        f"{slugify(args.model)}_{args.trait}_persona_steer_l{args.layer}_c{args.coef:g}_"
        f"{started.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = RESULTS_DIR / run_id
    vector_dir = args.vector_dir or (RESULTS_DIR / "persona_vectors" / slugify(args.model))
    vector_path = args.vector_path or (vector_dir / f"{args.trait}_response_avg_diff.pt")

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    if args.force_vector or not vector_path.exists():
        trait_data = load_trait_data(args.persona_repo, args.trait, version="extract")
        vector_path = create_vector(
            model,
            tokenizer,
            trait_data,
            args.trait,
            vector_dir,
            max_pairs=args.max_vector_pairs,
            max_new_tokens=args.vector_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )

    questions = load_jsonl(args.questions_file)
    metadata = {
        "run_id": run_id,
        "model": args.model,
        "lora": None,
        "persona_repo": str(args.persona_repo),
        "trait": args.trait,
        "vector_path": str(vector_path),
        "layer": args.layer,
        "coef": args.coef,
        "steering_type": args.steering_type,
        "eval_questions_file": str(args.questions_file),
        "n_questions": len(questions),
        "epochs": args.epochs,
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "started_at": started.isoformat(),
        "completed_at": None,
        "elapsed_seconds": None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    t0 = time.monotonic()
    run_steered_eval(
        model,
        tokenizer,
        questions,
        out_dir,
        args.model,
        vector_path,
        layer=args.layer,
        coef=args.coef,
        steering_type=args.steering_type,
        epochs=args.epochs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    elapsed = time.monotonic() - t0
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["elapsed_seconds"] = round(elapsed, 1)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
