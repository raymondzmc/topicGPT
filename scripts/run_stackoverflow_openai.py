"""
StackOverflow TopicGPT Pipeline - OpenAI (GPT-4o / GPT-4o-mini)

This script runs the full TopicGPT pipeline on StackOverflow data:
1. Topic Generation (GPT-4o)
2. Topic Refinement (GPT-4o)
3. Topic Assignment (GPT-4o-mini)
4. Topic Correction (GPT-4o-mini)
5. Evaluation (metrics calculation)

Uses sequential processing (max_workers=1) to match the original paper implementation.
Token usage and costs are tracked and logged throughout.
"""

import os
import sys

# Add project root to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import yaml
import time
from datetime import datetime
from dotenv import load_dotenv
from topicgpt_python import generate_topic_lvl1, refine_topics, assign_topics, correct_topics
from topicgpt_python.metrics import metric_calc
from topicgpt_python.utils import APIClient

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment. Add it to your .env file.")

# Load config
with open("config.yml", "r") as f:
    cfg = yaml.safe_load(f)

# ========================================
# Configuration
# ========================================

# OpenAI Models
MODEL_GENERATION = "gpt-4o"
MODEL_REFINEMENT = "gpt-4o"
MODEL_ASSIGNMENT = "gpt-4o-mini"
MODEL_CORRECTION = "gpt-4o-mini"
API = "openai"

# Sequential processing (matches original paper implementation)
MAX_WORKERS = 1

# Pricing per 1M tokens (as of 2024)
# GPT-4o: $2.50 input, $10.00 output
# GPT-4o-mini: $0.15 input, $0.60 output
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# Data files
GEN_DATA = "data/input/stackoverflow_gen.jsonl"
EVAL_DATA = "data/input/stackoverflow_eval.jsonl"

# Output directory
OUTPUT_DIR = "data/output/stackoverflow_openai"

# ========================================
# Token Tracking
# ========================================

class TokenTracker:
    """Track token usage and calculate costs across multiple models."""
    
    def __init__(self, pricing: dict):
        self.pricing = pricing
        self.steps = {}
        self.current_step = None
        self._last_prompt = 0
        self._last_completion = 0
    
    def start_step(self, step_name: str, model: str):
        """Start tracking a new step."""
        self.current_step = step_name
        self.steps[step_name] = {
            "model": model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "start_time": time.time(),
            "end_time": None,
        }
        self._last_prompt = 0
        self._last_completion = 0
    
    def end_step(self, api_client: APIClient):
        """End the current step and record final token counts."""
        if self.current_step and self.current_step in self.steps:
            usage = api_client.get_usage()
            self.steps[self.current_step]["prompt_tokens"] = usage["prompt_tokens"] - self._last_prompt
            self.steps[self.current_step]["completion_tokens"] = usage["completion_tokens"] - self._last_completion
            self.steps[self.current_step]["end_time"] = time.time()
            
            # Update last counts for next step
            self._last_prompt = usage["prompt_tokens"]
            self._last_completion = usage["completion_tokens"]
    
    def get_step_cost(self, step_name: str) -> float:
        """Calculate cost for a specific step."""
        step = self.steps.get(step_name, {})
        model = step.get("model", "gpt-4o")
        price = self.pricing.get(model, {"input": 0, "output": 0})
        input_cost = (step.get("prompt_tokens", 0) / 1_000_000) * price["input"]
        output_cost = (step.get("completion_tokens", 0) / 1_000_000) * price["output"]
        return input_cost + output_cost
    
    def get_step_duration(self, step_name: str) -> float:
        """Get duration in seconds for a step."""
        step = self.steps.get(step_name, {})
        if step.get("end_time") and step.get("start_time"):
            return step["end_time"] - step["start_time"]
        return 0
    
    def get_totals(self) -> dict:
        """Get total token usage and cost."""
        total_prompt = sum(s["prompt_tokens"] for s in self.steps.values())
        total_completion = sum(s["completion_tokens"] for s in self.steps.values())
        total_cost = sum(self.get_step_cost(s) for s in self.steps)
        total_duration = sum(self.get_step_duration(s) for s in self.steps)
        
        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost": total_cost,
            "total_duration_seconds": total_duration,
        }
    
    def print_step_summary(self, step_name: str):
        """Print summary for a step."""
        step = self.steps.get(step_name, {})
        duration = self.get_step_duration(step_name)
        cost = self.get_step_cost(step_name)
        
        print(f"\n  {'─' * 50}")
        print(f"  Step: {step_name}")
        print(f"  Model: {step.get('model', 'N/A')}")
        print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Prompt tokens: {step.get('prompt_tokens', 0):,}")
        print(f"  Completion tokens: {step.get('completion_tokens', 0):,}")
        print(f"  Cost: ${cost:.4f}")
        print(f"  {'─' * 50}")
    
    def print_final_summary(self):
        """Print final summary of all steps."""
        totals = self.get_totals()
        
        print("\n" + "=" * 70)
        print("TOKEN USAGE & COST SUMMARY")
        print("=" * 70)
        print(f"\nModels: {MODEL_GENERATION} (gen/refine), {MODEL_ASSIGNMENT} (assign/correct)")
        print(f"Processing: Sequential (max_workers={MAX_WORKERS})")
        print()
        
        # Per-step breakdown
        print(f"{'Step':<20} {'Model':<15} {'Prompt':>12} {'Completion':>12} {'Cost':>10} {'Time':>10}")
        print("-" * 85)
        
        for step_name in self.steps:
            step = self.steps[step_name]
            cost = self.get_step_cost(step_name)
            duration = self.get_step_duration(step_name)
            print(f"{step_name:<20} {step['model']:<15} {step['prompt_tokens']:>12,} {step['completion_tokens']:>12,} ${cost:>9.4f} {duration:>8.1f}s")
        
        print("-" * 85)
        print(f"{'TOTAL':<20} {'':<15} {totals['prompt_tokens']:>12,} {totals['completion_tokens']:>12,} ${totals['total_cost']:>9.4f} {totals['total_duration_seconds']:>8.1f}s")
        print("=" * 85)
        print(f"\nTotal tokens: {totals['total_tokens']:,}")
        print(f"Total cost: ${totals['total_cost']:.4f}")
        print(f"Total time: {totals['total_duration_seconds']/60:.1f} minutes")
        
        return totals
    
    def to_dict(self) -> dict:
        """Export tracking data as dictionary."""
        return {
            "pricing": self.pricing,
            "steps": {
                name: {
                    **step,
                    "cost": self.get_step_cost(name),
                    "duration_seconds": self.get_step_duration(name),
                }
                for name, step in self.steps.items()
            },
            "totals": self.get_totals(),
        }

# ========================================
# Patch APIClient for global tracking
# ========================================

# Create a shared APIClient for tracking
shared_client = None

_original_init = APIClient.__init__

def _patched_init(self, api, model, host=None):
    global shared_client
    _original_init(self, api, model, host)
    # Share token counts with the first client created
    if shared_client is not None and api == API:
        self.total_prompt_tokens = shared_client.total_prompt_tokens
        self.total_completion_tokens = shared_client.total_completion_tokens
    shared_client = self

APIClient.__init__ = _patched_init

# ========================================
# Main Pipeline
# ========================================

def main():
    global shared_client
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize tracker
    tracker = TokenTracker(PRICING)
    
    # File paths
    generation_file = f"{OUTPUT_DIR}/generation_1.jsonl"
    generation_topic_file = f"{OUTPUT_DIR}/generation_1.md"
    refinement_file = f"{OUTPUT_DIR}/refinement.md"
    refinement_jsonl = f"{OUTPUT_DIR}/refinement.jsonl"
    refinement_mapping = f"{OUTPUT_DIR}/refinement_mapping.json"
    assignment_file = f"{OUTPUT_DIR}/assignment.jsonl"
    assignment_corrected_file = f"{OUTPUT_DIR}/assignment_corrected.jsonl"
    
    print("=" * 70)
    print("StackOverflow TopicGPT Pipeline - OpenAI")
    print("=" * 70)
    print(f"\nGeneration/Refinement Model: {MODEL_GENERATION}")
    print(f"Assignment/Correction Model: {MODEL_ASSIGNMENT}")
    print(f"Processing: Sequential (max_workers={MAX_WORKERS})")
    print(f"Generation data: {GEN_DATA}")
    print(f"Evaluation data: {EVAL_DATA}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize shared client for tracking
    shared_client = APIClient(api=API, model=MODEL_GENERATION)
    
    # ========================================
    # Step 1: Topic Generation (GPT-4o)
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 1/5: Topic Generation (GPT-4o)")
    print("=" * 70)
    
    tracker.start_step("Generation", MODEL_GENERATION)
    
    generate_topic_lvl1(
        api=API,
        model=MODEL_GENERATION,
        data=GEN_DATA,
        prompt_file=cfg["generation"]["prompt"],
        seed_file=cfg["generation"]["seed"],
        out_file=generation_file,
        topic_file=generation_topic_file,
        verbose=True,
        max_workers=MAX_WORKERS,
    )
    
    tracker.end_step(shared_client)
    tracker.print_step_summary("Generation")
    
    # ========================================
    # Step 2: Topic Refinement (GPT-4o)
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 2/5: Topic Refinement (GPT-4o)")
    print("=" * 70)
    
    tracker.start_step("Refinement", MODEL_REFINEMENT)
    
    refine_topics(
        api=API,
        model=MODEL_REFINEMENT,
        prompt_file=cfg["refinement"]["prompt"],
        generation_file=generation_file,
        topic_file=generation_topic_file,
        out_file=refinement_file,
        updated_file=refinement_jsonl,
        verbose=True,
        remove=True,
        mapping_file=refinement_mapping,
    )
    
    tracker.end_step(shared_client)
    tracker.print_step_summary("Refinement")
    
    # ========================================
    # Step 3: Topic Assignment (GPT-4o-mini)
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 3/5: Topic Assignment (GPT-4o-mini)")
    print("=" * 70)
    
    tracker.start_step("Assignment", MODEL_ASSIGNMENT)
    
    assign_topics(
        api=API,
        model=MODEL_ASSIGNMENT,
        data=EVAL_DATA,
        prompt_file=cfg["assignment"]["prompt"],
        out_file=assignment_file,
        topic_file=refinement_file,
        verbose=True,
        max_workers=MAX_WORKERS,
    )
    
    tracker.end_step(shared_client)
    tracker.print_step_summary("Assignment")
    
    # ========================================
    # Step 4: Topic Correction (GPT-4o-mini)
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 4/5: Topic Correction (GPT-4o-mini)")
    print("=" * 70)
    
    tracker.start_step("Correction", MODEL_CORRECTION)
    
    correct_topics(
        api=API,
        model=MODEL_CORRECTION,
        data_path=assignment_file,
        prompt_path=cfg["correction"]["prompt"],
        topic_path=refinement_file,
        output_path=assignment_corrected_file,
        verbose=True,
        max_workers=MAX_WORKERS,
    )
    
    tracker.end_step(shared_client)
    tracker.print_step_summary("Correction")
    
    # ========================================
    # Step 5: Evaluation
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 5/5: Evaluation")
    print("=" * 70)
    
    tracker.start_step("Evaluation", "N/A")
    
    harmonic_purity, ari, nmi = metric_calc(
        data_file=assignment_corrected_file,
        ground_truth_col="label",
        output_col="responses",
    )
    
    tracker.steps["Evaluation"]["end_time"] = time.time()
    
    # ========================================
    # Final Results
    # ========================================
    
    # Print token/cost summary
    totals = tracker.print_final_summary()
    
    # Print evaluation metrics
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    print(f"\nHarmonic Purity: {harmonic_purity:.4f}")
    print(f"ARI (Adjusted Rand Index): {ari:.4f}")
    print(f"NMI (Normalized Mutual Information): {nmi:.4f}")
    
    # Save complete results
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "generation": MODEL_GENERATION,
            "refinement": MODEL_REFINEMENT,
            "assignment": MODEL_ASSIGNMENT,
            "correction": MODEL_CORRECTION,
        },
        "api": API,
        "max_workers": MAX_WORKERS,
        "data": {
            "generation": GEN_DATA,
            "evaluation": EVAL_DATA,
        },
        "metrics": {
            "harmonic_purity": harmonic_purity,
            "ari": ari,
            "nmi": nmi,
        },
        "token_usage": tracker.to_dict(),
        "output_files": {
            "generation": generation_file,
            "generation_topics": generation_topic_file,
            "refinement": refinement_file,
            "refinement_jsonl": refinement_jsonl,
            "assignment": assignment_file,
            "assignment_corrected": assignment_corrected_file,
        }
    }
    
    results_file = f"{OUTPUT_DIR}/results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

