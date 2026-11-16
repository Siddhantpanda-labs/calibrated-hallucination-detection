"""
LLM Generation Module:
Extracts greedy answer, token log-probabilities, token entropies,
hidden states at designated layer positions, and stochastic samples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence, Any
import numpy as np


@dataclass
class GenerationResult:
    """Stores full generation artifacts for a single question."""
    question_id: str
    question: str
    greedy_text: str
    greedy_tokens: list[int]
    token_logprobs: list[float]
    token_entropies: list[float]
    hidden_states: dict[int, list[float]]  # layer_idx -> hidden vector
    sampled_answers: list[str]


class MockGenerator:
    """
    Mock generator simulating LLM inference for dry-runs, testing, and offline execution.
    Generates realistic logprobs, entropies, and synthetic hidden states.
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        layers: Sequence[int] = (4, 8, 12, 16, 20, 24, -1),
        num_samples: int = 5,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.layers = list(layers)
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        question_id: str,
        question: str,
        gold_answers: Sequence[str] | None = None,
    ) -> GenerationResult:
        # Simulate ~40% chance of hallucination
        is_hallucinating = self.rng.rand() < 0.40
        
        if gold_answers and len(gold_answers) > 0 and not is_hallucinating:
            greedy_text = gold_answers[0]
            # Higher logprobs for correct answers
            mean_lp = float(self.rng.uniform(-0.8, -0.1))
            mean_ent = float(self.rng.uniform(0.1, 0.6))
        else:
            greedy_text = "Completely incorrect hallucination"
            # Lower logprobs, higher entropy for wrong answers
            mean_lp = float(self.rng.uniform(-3.5, -1.2))
            mean_ent = float(self.rng.uniform(0.8, 2.5))

        seq_len = max(2, len(greedy_text.split()))
        logprobs = [float(self.rng.normal(mean_lp, 0.2)) for _ in range(seq_len)]
        entropies = [max(0.01, float(self.rng.normal(mean_ent, 0.2))) for _ in range(seq_len)]

        # Simulate hidden states: encode label signal into later layers
        hidden_states = {}
        for l_idx in self.layers:
            base_vec = self.rng.randn(self.hidden_dim).astype(np.float32)
            if l_idx in (20, 24, -1):
                # Add signal to later layers
                signal = 1.0 if is_hallucinating else -1.0
                base_vec[:10] += signal * 0.8
            hidden_states[str(l_idx)] = base_vec.tolist()

        # Sampled answers: consistency matches correctness probability
        sampled_answers = []
        for _ in range(self.num_samples):
            if not is_hallucinating and self.rng.rand() < 0.8:
                sampled_answers.append(greedy_text)
            else:
                sampled_answers.append(f"Sample alternative {self.rng.randint(1, 100)}")

        return GenerationResult(
            question_id=question_id,
            question=question,
            greedy_text=greedy_text,
            greedy_tokens=list(range(seq_len)),
            token_logprobs=logprobs,
            token_entropies=entropies,
            hidden_states=hidden_states,
            sampled_answers=sampled_answers,
        )


class HuggingFaceGenerator:
    """
    HuggingFace Transformers Generator.
    Extracts logits, token logprobs, Shannon entropy, and hidden states.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        torch_dtype: str = "float16",
        load_in_4bit: bool = False,
        probe_layers: Sequence[int] = (4, 8, 12, 16, 20, 24, -1),
        token_position: str = "last_answer",
        prompt_template: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.probe_layers = list(probe_layers)
        self.token_position = token_position
        self.prompt_template = prompt_template or (
            "Answer the following question in a few words without explanation.\n\n"
            "Question: {question}\nAnswer:"
        )

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.float16)

        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model {model_name} on {device}...")
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
        }
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        elif device == "auto":
            kwargs["device_map"] = "auto"
        elif device == "cuda" and torch.cuda.is_available():
            kwargs["device_map"] = "cuda:0"
        else:
            kwargs["device_map"] = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def _format_prompt(self, question: str) -> str:
        return self.prompt_template.format(question=question)

    def generate(
        self,
        question_id: str,
        question: str,
        max_new_tokens: int = 32,
        num_samples: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> GenerationResult:
        import torch

        prompt = self._format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        # 1. Greedy Pass with hidden states and scores
        with torch.no_grad():
            greedy_out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_tokens = greedy_out.sequences[0, input_len:].tolist()
        greedy_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Compute per-token logprobs and entropy
        token_logprobs = []
        token_entropies = []
        
        for step_idx, score in enumerate(greedy_out.scores):
            # score: (1, vocab_size)
            logits = score[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            
            tok_id = gen_tokens[step_idx]
            tok_lp = float(log_probs[tok_id].item())
            
            # Shannon entropy: -sum(p * log(p))
            tok_ent = float(-(probs * log_probs).sum().item())
            
            token_logprobs.append(tok_lp)
            token_entropies.append(tok_ent)

        # Extract hidden states at specified layers
        # greedy_out.hidden_states is a tuple of generation steps
        # Each step is a tuple of layers: shape (batch=1, seq_len, hidden_dim)
        hidden_states_dict = {}
        if greedy_out.hidden_states and len(greedy_out.hidden_states) > 0:
            last_step_hs = greedy_out.hidden_states[-1]
            total_layers = len(last_step_hs)
            
            for l_idx in self.probe_layers:
                # Resolve negative indexing
                resolved_layer = l_idx if l_idx >= 0 else total_layers + l_idx
                if 0 <= resolved_layer < total_layers:
                    layer_tensor = last_step_hs[resolved_layer]  # (1, 1, hidden_dim)
                    vec = layer_tensor[0, -1].float().cpu().numpy().tolist()
                    hidden_states_dict[str(l_idx)] = vec

        # 2. Stochastic Sampling Pass
        sampled_answers = []
        if num_samples > 0:
            with torch.no_grad():
                sample_out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_samples,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
            seqs = sample_out.sequences if hasattr(sample_out, "sequences") else sample_out
            for i in range(num_samples):
                sample_toks = seqs[i, input_len:]
                s_text = self.tokenizer.decode(sample_toks, skip_special_tokens=True).strip()
                sampled_answers.append(s_text)

        return GenerationResult(
            question_id=question_id,
            question=question,
            greedy_text=greedy_text,
            greedy_tokens=gen_tokens,
            token_logprobs=token_logprobs,
            token_entropies=token_entropies,
            hidden_states=hidden_states_dict,
            sampled_answers=sampled_answers,
        )


def get_generator(
    config: dict[str, Any] | None = None,
    mock: bool = False,
    seed: int = 42,
) -> HuggingFaceGenerator | MockGenerator:
    """Factory function returning a configured LLM generator or mock generator."""
    if mock:
        return MockGenerator(seed=seed)

    try:
        import torch
        import transformers
    except ImportError:
        print("[get_generator] PyTorch/Transformers not installed. Returning MockGenerator.")
        return MockGenerator(seed=seed)

    config = config or {}
    model_cfg = config.get("model", {})
    gen_cfg = config.get("generation", {})
    feat_cfg = config.get("features", {})

    return HuggingFaceGenerator(
        model_name=model_cfg.get("name", "Qwen/Qwen2.5-1.5B-Instruct"),
        device=model_cfg.get("device", "auto"),
        torch_dtype=model_cfg.get("torch_dtype", "float16"),
        load_in_4bit=model_cfg.get("load_in_4bit", False),
        probe_layers=feat_cfg.get("probe_layers", (4, 8, 12, 16, 20, 24, -1)),
        token_position=feat_cfg.get("probe_token_position", "last_answer"),
        prompt_template=gen_cfg.get("prompt_template", None),
    )

