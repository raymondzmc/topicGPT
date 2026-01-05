# StackOverflow Evaluation Script

## Overview

`run_stackoverflow_eval.py` runs the complete TopicGPT pipeline on the StackOverflow dataset using two model configurations and compares their performance.

## Prerequisites

1. Ensure you have a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_key
   TOGETHER_AI_API_KEY=your_together_ai_key
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the evaluation script from the project root:

```bash
python scripts/run_stackoverflow_eval.py
```

## What It Does

The script runs the following pipeline for **both** OpenAI and Together AI:

1. **Generation** (on stackoverflow_gen.jsonl - 10,000 docs)
   - OpenAI: GPT-4
   - Together AI: Llama 3.1-8B-Instruct-Turbo

2. **Refinement** (merge similar topics, remove low-frequency topics)
   - OpenAI: GPT-4
   - Together AI: Llama 3.1-8B-Instruct-Turbo

3. **Assignment** (on stackoverflow_eval.jsonl - 10,000 docs)
   - OpenAI: GPT-3.5-turbo
   - Together AI: Llama 3.1-8B-Instruct-Turbo

4. **Correction** (fix hallucinated/invalid topics)
   - OpenAI: GPT-3.5-turbo
   - Together AI: Llama 3.1-8B-Instruct-Turbo

5. **Evaluation** (compare predicted topics against ground truth labels)
   - Metrics: Harmonic Purity, ARI, NMI

## Output

Results are saved to:
```
data/output/stackoverflow/
├── openai/
│   ├── generation_1.jsonl          # Generated topics
│   ├── generation_1.md             # Topic hierarchy
│   ├── refinement.md               # Refined topics
│   ├── refinement.jsonl            # Updated with refined topics
│   ├── refinement_mapping.json     # Topic merge mapping
│   ├── assignment.jsonl            # Assigned topics
│   └── assignment_corrected.jsonl  # Corrected assignments
├── together/
│   └── (same structure as openai/)
└── comparison_results.json         # Final metrics comparison
```

## Expected Runtime

**Note:** This script will make **many API calls** (~20,000+ total across both configurations):
- Generation: ~10,000 calls
- Refinement: ~50-100 calls (depending on topic pairs)
- Assignment: ~10,000 calls
- Correction: varies based on errors

**Estimated cost and time:**
- OpenAI: ~$100-200, 2-4 hours
- Together AI: ~$5-10, 1-2 hours

## Interpreting Results

The comparison table shows three metrics:

- **Harmonic Purity**: How well predicted topics align with ground truth (0-1, higher is better)
- **ARI** (Adjusted Rand Index): Cluster agreement (-1 to 1, higher is better)
- **NMI** (Normalized Mutual Information): Information shared between clusterings (0-1, higher is better)

## Troubleshooting

If the script fails mid-way:
- Check the output files to see which step completed
- Re-run the script - it will overwrite previous outputs
- Alternatively, manually run individual steps from the example in the main README

