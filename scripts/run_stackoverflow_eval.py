"""
StackOverflow TopicGPT Evaluation Script

This script runs the full TopicGPT pipeline on StackOverflow data using:
1. OpenAI models (GPT-4 for generation, GPT-3.5-turbo for assignment)
2. Together AI models (Llama 3.1-8B for both generation and assignment)

It evaluates both configurations, tracks token usage, calculates costs,
and prints comparison metrics.
"""

import os
import json
import yaml
from dotenv import load_dotenv
from topicgpt_python import generate_topic_lvl1, refine_topics, assign_topics, correct_topics
from topicgpt_python.metrics import metric_calc
from topicgpt_python.utils import APIClient
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Verify API keys are loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment")
if not os.getenv("TOGETHER_AI_API_KEY"):
    raise ValueError("TOGETHER_AI_API_KEY not found in environment")

# Load config
with open("config.yml", "r") as f:
    cfg = yaml.safe_load(f)

# Create output directories
os.makedirs("data/output/stackoverflow/openai", exist_ok=True)
os.makedirs("data/output/stackoverflow/together", exist_ok=True)

print("=" * 80)
print("StackOverflow TopicGPT Evaluation")
print("=" * 80)

# ========================================
# Configuration
# ========================================

# Data files
GEN_DATA = "data/input/stackoverflow_gen.jsonl"
EVAL_DATA = "data/input/stackoverflow_eval.jsonl"

# Pricing per 1M tokens (input, output)
PRICING = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {"input": 0.18, "output": 0.18},
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD based on token usage and model pricing."""
    if model not in PRICING:
        print(f"Warning: Pricing not found for model {model}, using estimate")
        return (prompt_tokens + completion_tokens) / 1_000_000 * 1.0  # $1/1M tokens default
    
    input_cost = (prompt_tokens / 1_000_000) * PRICING[model]["input"]
    output_cost = (completion_tokens / 1_000_000) * PRICING[model]["output"]
    return input_cost + output_cost

# Model configurations
configurations = {
    "openai": {
        "api": "openai",
        "gen_model": "gpt-4",
        "assign_model": "gpt-3.5-turbo",
        "output_dir": "data/output/stackoverflow/openai",
    },
    "together": {
        "api": "together",
        "gen_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "assign_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "output_dir": "data/output/stackoverflow/together",
    },
}

# ========================================
# Global Token Tracker
# ========================================
# Track tokens across all APIClient instances using a module-level variable

class GlobalTokenTracker:
    """Singleton to track tokens across all APIClient instances."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance
    
    def reset(self):
        self.by_model = {}
    
    def add(self, model: str, prompt_tokens: int, completion_tokens: int):
        if model not in self.by_model:
            self.by_model[model] = {"prompt": 0, "completion": 0}
        self.by_model[model]["prompt"] += prompt_tokens
        self.by_model[model]["completion"] += completion_tokens
    
    def get_totals(self):
        total_prompt = sum(m["prompt"] for m in self.by_model.values())
        total_completion = sum(m["completion"] for m in self.by_model.values())
        return total_prompt, total_completion
    
    def get_cost(self):
        total_cost = 0.0
        for model, tokens in self.by_model.items():
            total_cost += calculate_cost(model, tokens["prompt"], tokens["completion"])
        return total_cost

# Create global tracker
token_tracker = GlobalTokenTracker()

# Patch APIClient.iterative_prompt to report to global tracker
_original_iterative_prompt = APIClient.iterative_prompt

def _tracked_iterative_prompt(self, prompt, max_tokens, temperature, top_p=1.0,
                               system_message="You are a helpful assistant.",
                               num_try=3, verbose=False):
    # Capture token counts before the call
    before_prompt = self.total_prompt_tokens
    before_completion = self.total_completion_tokens
    
    # Call original method
    result = _original_iterative_prompt(
        self, prompt, max_tokens, temperature, top_p, system_message, num_try, verbose
    )
    
    # Calculate tokens added in this call
    added_prompt = self.total_prompt_tokens - before_prompt
    added_completion = self.total_completion_tokens - before_completion
    
    # Report to global tracker
    if added_prompt > 0 or added_completion > 0:
        token_tracker.add(self.model, added_prompt, added_completion)
    
    return result

# Apply the patch
APIClient.iterative_prompt = _tracked_iterative_prompt

# ========================================
# Run Pipeline for Each Configuration
# ========================================

results = {}

for config_name, config in configurations.items():
    print("\n" + "=" * 80)
    print(f"Running TopicGPT Pipeline with {config_name.upper()}")
    print("=" * 80)
    
    # Reset token tracker for this configuration
    token_tracker.reset()
    
    api = config["api"]
    gen_model = config["gen_model"]
    assign_model = config["assign_model"]
    output_dir = config["output_dir"]
    
    # File paths
    generation_file = f"{output_dir}/generation_1.jsonl"
    generation_topic_file = f"{output_dir}/generation_1.md"
    refinement_file = f"{output_dir}/refinement.md"
    refinement_jsonl = f"{output_dir}/refinement.jsonl"
    refinement_mapping = f"{output_dir}/refinement_mapping.json"
    assignment_file = f"{output_dir}/assignment.jsonl"
    assignment_corrected_file = f"{output_dir}/assignment_corrected.jsonl"
    
    # ----- Step 1: Generation -----
    print(f"\n[{config_name}] Step 1/4: Topic Generation on stackoverflow_gen.jsonl")
    print(f"Model: {gen_model}")
    generate_topic_lvl1(
        api=api,
        model=gen_model,
        data=GEN_DATA,
        prompt_file=cfg["generation"]["prompt"],
        seed_file=cfg["generation"]["seed"],
        out_file=generation_file,
        topic_file=generation_topic_file,
        verbose=True,
    )
    
    gen_prompt, gen_completion = token_tracker.get_totals()
    print(f"  [Tokens so far] Prompt: {gen_prompt:,}, Completion: {gen_completion:,}")
    
    # ----- Step 2: Refinement -----
    print(f"\n[{config_name}] Step 2/4: Topic Refinement")
    print(f"Model: {gen_model}")
    refine_topics(
        api=api,
        model=gen_model,
        prompt_file=cfg["refinement"]["prompt"],
        generation_file=generation_file,
        topic_file=generation_topic_file,
        out_file=refinement_file,
        updated_file=refinement_jsonl,
        verbose=True,
        remove=True,
        mapping_file=refinement_mapping,
    )
    
    refine_prompt, refine_completion = token_tracker.get_totals()
    print(f"  [Tokens so far] Prompt: {refine_prompt:,}, Completion: {refine_completion:,}")
    
    # ----- Step 3: Assignment -----
    print(f"\n[{config_name}] Step 3/4: Topic Assignment on stackoverflow_eval.jsonl")
    print(f"Model: {assign_model}")
    assign_topics(
        api=api,
        model=assign_model,
        data=EVAL_DATA,
        prompt_file=cfg["assignment"]["prompt"],
        out_file=assignment_file,
        topic_file=refinement_file,
        verbose=True,
    )
    
    assign_prompt, assign_completion = token_tracker.get_totals()
    print(f"  [Tokens so far] Prompt: {assign_prompt:,}, Completion: {assign_completion:,}")
    
    # ----- Step 4: Correction -----
    print(f"\n[{config_name}] Step 4/4: Topic Correction")
    print(f"Model: {assign_model}")
    correct_topics(
        api=api,
        model=assign_model,
        data_path=assignment_file,
        prompt_path=cfg["correction"]["prompt"],
        topic_path=refinement_file,
        output_path=assignment_corrected_file,
        verbose=True,
    )
    
    final_prompt, final_completion = token_tracker.get_totals()
    print(f"  [Final tokens] Prompt: {final_prompt:,}, Completion: {final_completion:,}")
    
    # ----- Step 5: Evaluation -----
    print(f"\n[{config_name}] Step 5/5: Calculating Metrics")
    harmonic_purity, ari, mis = metric_calc(
        data_file=assignment_corrected_file,
        ground_truth_col="label",
        output_col="responses",
    )
    
    # Get final token counts and cost
    total_prompt, total_completion = token_tracker.get_totals()
    total_tokens = total_prompt + total_completion
    total_cost = token_tracker.get_cost()
    
    # Store results
    results[config_name] = {
        "harmonic_purity": harmonic_purity,
        "ari": ari,
        "mis": mis,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "estimated_cost": total_cost,
        "token_breakdown_by_model": {
            model: {"prompt": tokens["prompt"], "completion": tokens["completion"]}
            for model, tokens in token_tracker.by_model.items()
        }
    }
    
    print(f"\n[{config_name}] Final Token Usage:")
    print(f"  Prompt tokens: {total_prompt:,}")
    print(f"  Completion tokens: {total_completion:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Estimated cost: ${total_cost:.2f}")
    
    # Show breakdown by model
    print(f"\n  Breakdown by model:")
    for model, tokens in token_tracker.by_model.items():
        model_cost = calculate_cost(model, tokens["prompt"], tokens["completion"])
        print(f"    {model}:")
        print(f"      Prompt: {tokens['prompt']:,}, Completion: {tokens['completion']:,}")
        print(f"      Cost: ${model_cost:.2f}")

# ========================================
# Print Comparison Table
# ========================================

def format_number(n):
    """Format number with commas."""
    return f"{n:,}"

print("\n" + "=" * 80)
print("FINAL RESULTS COMPARISON")
print("=" * 80)
print()

# Header
col1_width = 20
col2_width = 25
col3_width = 25

print(f"{'Metric':<{col1_width}} {'OpenAI (GPT-4/3.5)':<{col2_width}} {'Together (Llama-3.1-8B)':<{col3_width}}")
print("-" * 80)

# Metrics
print(f"{'Harmonic Purity':<{col1_width}} {results['openai']['harmonic_purity']:<{col2_width}.4f} {results['together']['harmonic_purity']:<{col3_width}.4f}")
print(f"{'ARI':<{col1_width}} {results['openai']['ari']:<{col2_width}.4f} {results['together']['ari']:<{col3_width}.4f}")
print(f"{'NMI':<{col1_width}} {results['openai']['mis']:<{col2_width}.4f} {results['together']['mis']:<{col3_width}.4f}")
print("-" * 80)

# Token counts
openai_prompt = format_number(results['openai']['prompt_tokens'])
openai_completion = format_number(results['openai']['completion_tokens'])
openai_total = format_number(results['openai']['total_tokens'])
together_prompt = format_number(results['together']['prompt_tokens'])
together_completion = format_number(results['together']['completion_tokens'])
together_total = format_number(results['together']['total_tokens'])

print(f"{'Prompt Tokens':<{col1_width}} {openai_prompt:<{col2_width}} {together_prompt:<{col3_width}}")
print(f"{'Completion Tokens':<{col1_width}} {openai_completion:<{col2_width}} {together_completion:<{col3_width}}")
print(f"{'Total Tokens':<{col1_width}} {openai_total:<{col2_width}} {together_total:<{col3_width}}")

# Costs
openai_cost = f"${results['openai']['estimated_cost']:.2f}"
together_cost = f"${results['together']['estimated_cost']:.2f}"
print(f"{'Estimated Cost':<{col1_width}} {openai_cost:<{col2_width}} {together_cost:<{col3_width}}")
print("-" * 80)
print()

# Cost comparison
openai_cost_val = results['openai']['estimated_cost']
together_cost_val = results['together']['estimated_cost']
if openai_cost_val > 0:
    savings = openai_cost_val - together_cost_val
    savings_pct = (savings / openai_cost_val) * 100
    print(f"Cost Savings with Together AI: ${savings:.2f} ({savings_pct:.1f}% less than OpenAI)")
print()

# Save results to JSON
with open("data/output/stackoverflow/comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to: data/output/stackoverflow/comparison_results.json")
print()
