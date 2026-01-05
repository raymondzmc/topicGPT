#!/usr/bin/env python3
"""
Convert StackOverflow TSV file to JSONL format for TopicGPT.

Usage:
    python scripts/convert_tsv_to_jsonl.py

Input: data/input/stackoverflow.tsv (TSV with 'text' and 'label' columns)
Output: data/input/stackoverflow.jsonl (JSONL with 'id', 'text', and 'label' fields)
"""

import json
import csv
from pathlib import Path


# Label mapping from numeric to string
LABEL_MAPPING = {
    1: "wordpress",
    2: "oracle",
    3: "svn",
    4: "apache",
    5: "excel",
    6: "matlab",
    7: "visual-studio",
    8: "cocoa",
    9: "osx",
    10: "bash",
    11: "spring",
    12: "hibernate",
    13: "scala",
    14: "sharepoint",
    15: "ajax",
    16: "qt",
    17: "drupal",
    18: "linq",
    19: "haskell",
    20: "magento"
}


def convert_tsv_to_jsonl(
    tsv_path: str = "data/input/stackoverflow.tsv",
    jsonl_path: str = "data/input/stackoverflow.jsonl",
    id_prefix: str = "so"
):
    """
    Convert TSV file to JSONL format with label mapping.
    
    Args:
        tsv_path: Path to input TSV file
        jsonl_path: Path to output JSONL file
        id_prefix: Prefix for generated IDs (default: "so")
    """
    tsv_file = Path(tsv_path)
    jsonl_file = Path(jsonl_path)
    
    if not tsv_file.exists():
        raise FileNotFoundError(f"Input file not found: {tsv_path}")
    
    # Read TSV and write JSONL
    with open(tsv_file, 'r', encoding='utf-8') as f_in, \
         open(jsonl_file, 'w', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in, delimiter='\t')
        
        row_count = 0
        for idx, row in enumerate(reader, start=1):
            # Convert numeric label to string label
            numeric_label = int(row['label'])
            string_label = LABEL_MAPPING.get(numeric_label, f"unknown-{numeric_label}")
            
            # Create JSON object with id, text, and label
            json_obj = {
                "id": f"{id_prefix}-{idx}",
                "text": row['text'],
                "label": string_label
            }
            
            # Write as single line JSON
            f_out.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            row_count += 1
        
        print(f"✓ Converted {row_count} rows from {tsv_path} to {jsonl_path}")
        print(f"  ID format: {id_prefix}-1 to {id_prefix}-{row_count}")
        print(f"  Labels: Mapped numeric labels to strings using LABEL_MAPPING")


def split_dataset(
    jsonl_path: str = "data/input/stackoverflow.jsonl",
    gen_path: str = "data/input/stackoverflow_gen.jsonl",
    eval_path: str = "data/input/stackoverflow_eval.jsonl",
    random_seed: int = 42
):
    """
    Split JSONL dataset into generation and evaluation subsets with stratified sampling.
    
    Args:
        jsonl_path: Path to input JSONL file
        gen_path: Path to output generation subset
        eval_path: Path to output evaluation subset
        random_seed: Random seed for reproducibility
    """
    import random
    from collections import defaultdict
    
    jsonl_file = Path(jsonl_path)
    
    if not jsonl_file.exists():
        raise FileNotFoundError(f"Input file not found: {jsonl_path}")
    
    # Read all records and group by label
    label_groups = defaultdict(list)
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                label_groups[record['label']].append(record)
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Split each label group 50/50
    gen_records = []
    eval_records = []
    
    for label, records in label_groups.items():
        # Shuffle records for this label
        shuffled = records.copy()
        random.shuffle(shuffled)
        
        # Split 50/50
        mid_point = len(shuffled) // 2
        gen_records.extend(shuffled[:mid_point])
        eval_records.extend(shuffled[mid_point:])
    
    # Shuffle the final datasets to mix labels
    random.shuffle(gen_records)
    random.shuffle(eval_records)
    
    # Write generation subset
    with open(gen_path, 'w', encoding='utf-8') as f:
        for record in gen_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Write evaluation subset
    with open(eval_path, 'w', encoding='utf-8') as f:
        for record in eval_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✓ Split {len(gen_records) + len(eval_records)} records into stratified subsets:")
    print(f"  Generation set: {gen_path} ({len(gen_records)} records)")
    print(f"  Evaluation set: {eval_path} ({len(eval_records)} records)")
    print(f"\n  Label distribution:")
    
    # Count labels in each subset
    gen_label_counts = defaultdict(int)
    eval_label_counts = defaultdict(int)
    
    for record in gen_records:
        gen_label_counts[record['label']] += 1
    for record in eval_records:
        eval_label_counts[record['label']] += 1
    
    for label in sorted(label_groups.keys()):
        total = len(label_groups[label])
        gen_count = gen_label_counts[label]
        eval_count = eval_label_counts[label]
        print(f"    {label:15s}: {gen_count:4d} gen, {eval_count:4d} eval (total: {total:4d})")


if __name__ == "__main__":
    # First convert TSV to JSONL
    convert_tsv_to_jsonl()
    
    # Then split into stratified subsets
    print("\n" + "="*60)
    split_dataset()

