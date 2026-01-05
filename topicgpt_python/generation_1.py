import pandas as pd
from topicgpt_python.utils import *
from tqdm import tqdm
import regex
import traceback
from sentence_transformers import SentenceTransformer, util
import argparse
import os
from anytree import Node, RenderTree
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sbert = SentenceTransformer("all-MiniLM-L6-v2")


def prompt_formatting(
    generation_prompt,
    api_client,
    doc,
    seed_file,
    topics_list,
    context_len,
    verbose,
    max_top_len=500,  # Maximum length of topics
):
    """
    Format prompt to include document and seed topics.
    Handle cases where prompt is too long.
    """
    # Load sentence transformer and calculate topic embeddings
    topic_str = "\n".join(
        [topic.split(":")[0].strip() for topic in topics_list]
    )  # Get rid of description in the actual prompt

    # Calculate length of document, seed topics, and prompt
    doc_len = api_client.estimate_token_count(doc)
    prompt_len = api_client.estimate_token_count(generation_prompt)
    topic_len = api_client.estimate_token_count(topic_str)
    total_len = prompt_len + doc_len + topic_len

    # Handle cases where prompt is too long
    if total_len > context_len:
        if doc_len > (context_len - prompt_len - max_top_len):  # Truncate document
            if verbose:
                print(f"Document is too long ({doc_len} tokens). Truncating...")
            doc = api_client.truncating(doc, context_len - prompt_len - max_top_len)
            prompt = generation_prompt.format(Document=doc, Topics=topic_str)
        else:  # Truncate topic list
            if verbose:
                print(f"Too many topics ({topic_len} tokens). Pruning...")
            cos_sim = {}
            doc_emb = sbert.encode(doc, convert_to_tensor=True)
            for top in topics_list:
                top_emb = sbert.encode(top, convert_to_tensor=True)
                cos_sim[top] = util.cos_sim(top_emb, doc_emb)
            sim_topics = sorted(cos_sim, key=cos_sim.get, reverse=True)

            # Retaining only similar topics that fit within the context length
            max_top_len = context_len - prompt_len - doc_len
            seed_len, seed_str = 0, ""
            while seed_len < max_top_len and sim_topics:
                new_seed = sim_topics.pop(0)
                if (
                    seed_len + api_client.estimate_token_count(new_seed + "\n")
                    > max_top_len
                ):
                    break
                else:
                    seed_str += new_seed + "\n"
                    seed_len += api_client.estimate_token_count(seed_str)
            prompt = generation_prompt.format(Document=doc, Topics=seed_str)
    else:
        prompt = generation_prompt.format(Document=doc, Topics=topic_str)
    return prompt


def _process_single_doc(args):
    """Helper function for concurrent processing of a single document."""
    doc, generation_prompt, api_client, seed_file, topics_list, context_len, max_tokens, temperature, top_p, verbose = args
    
    prompt = prompt_formatting(
        generation_prompt,
        api_client,
        doc,
        seed_file,
        topics_list,
        context_len,
        verbose=False,  # Reduce noise in concurrent mode
    )
    
    try:
        response = api_client.iterative_prompt(
            prompt, max_tokens, temperature, top_p=top_p, verbose=False
        )
        return response, None
    except Exception as e:
        return None, str(e)


def generate_topics(
    topics_root,
    topics_list,
    context_len,
    docs,
    seed_file,
    api_client,
    generation_prompt,
    temperature,
    max_tokens,
    top_p,
    verbose,
    early_stop=100,  # Modify this parameter to control early stopping
    max_workers=24,  # Increased for Together AI throughput
):
    """
    Generate topics from documents using LLMs.
    Uses concurrent processing in batches for faster execution.
    """
    responses = []
    running_dups = 0
    topic_format = regex.compile(r"^\[(\d+)\] ([\w\s]+):(.+)")
    
    # Process documents in batches for concurrent execution
    batch_size = max_workers * 2  # Process 2x workers at a time
    doc_batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
    
    pbar = tqdm(total=len(docs), desc="Generating topics")
    
    for batch in doc_batches:
        # Prepare arguments for concurrent processing
        # Each doc in batch uses current topics_list (frozen for this batch)
        batch_args = [
            (doc, generation_prompt, api_client, seed_file, topics_list, 
             context_len, max_tokens, temperature, top_p, verbose)
            for doc in batch
        ]
        
        batch_responses = []
        
        # Process batch concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single_doc, args): i 
                      for i, args in enumerate(batch_args)}
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    response, error = future.result()
                    if error:
                        print(f"Error processing document: {error}")
                        batch_responses.append((idx, "Error"))
                    else:
                        batch_responses.append((idx, response))
                except Exception as e:
                    traceback.print_exc()
                    batch_responses.append((idx, "Error"))
        
        # Sort by original order and process responses
        batch_responses.sort(key=lambda x: x[0])
        
        for _, response in batch_responses:
            pbar.update(1)
            
            if response == "Error":
                responses.append("Error")
                continue
                
            responses.append(response)
            
            # Parse topics and update tree
            topics = [t.strip() for t in response.split("\n") if t.strip()]
            for t in topics:
                if not regex.match(topic_format, t):
                    continue
                groups = regex.match(topic_format, t)
                lvl, name, desc = int(groups[1]), groups[2].strip(), groups[3].strip()

                if lvl != 1:
                    continue
                dups = topics_root.find_duplicates(name, lvl)

                if dups:
                    dups[0].count += 1
                    running_dups += 1
                else:
                    topics_root._add_node(lvl, name, 1, desc, topics_root.root)
                    topics_list = topics_root.to_topic_list(desc=False, count=False)
                    running_dups = 0
            
            if verbose:
                print(f"Topics: {response}")
                print("--------------------")
        
        # Check early stopping after each batch
        if running_dups > early_stop:
            pbar.close()
            return responses, topics_list, topics_root
    
    pbar.close()
    return responses, topics_list, topics_root


def generate_topic_lvl1(
    api, model, data, prompt_file, seed_file, out_file, topic_file, verbose
):
    """
    Generate high-level topics

    Parameters:
    - api (str): API to use ('openai', 'vertex', 'vllm', 'azure', 'gemini')
    - model (str): Model to use
    - data (str): Data file
    - prompt_file (str): File to read prompts from
    - seed_file (str): Markdown file to read seed topics from
    - out_file (str): File to write results to
    - topic_file (str): File to write topics to
    - verbose (bool): Whether to print out results

    Returns:
    - topics_root (TopicTree): Root node of the topic tree
    """
    api_client = APIClient(api=api, model=model)
    max_tokens, temperature, top_p = 1000, 0.0, 1.0

    if verbose:
        print("-------------------")
        print("Initializing topic generation...")
        print(f"Model: {model}")
        print(f"Data file: {data}")
        print(f"Prompt file: {prompt_file}")
        print(f"Seed file: {seed_file}")
        print(f"Output file: {out_file}")
        print(f"Topic file: {topic_file}")
        print("-------------------")

    # Model configuration
    context = (
        128000
        if model not in ["gpt-3.5-turbo", "gpt-4"]
        else (4096 if model == "gpt-3.5-turbo" else 8000)
    )
    context_len = context - max_tokens

    # Load data
    df = pd.read_json(data, lines=True)
    docs = df["text"].tolist()
    generation_prompt = open(prompt_file, "r").read()
    topics_root = TopicTree().from_seed_file(seed_file)
    topics_list = topics_root.to_topic_list(desc=True, count=False)

    # Generate topics
    responses, topics_list, topics_root = generate_topics(
        topics_root,
        topics_list,
        context_len,
        docs,
        seed_file,
        api_client,
        generation_prompt,
        temperature,
        max_tokens,
        top_p,
        verbose,
    )

    # Save generated topics
    topics_root.to_file(topic_file)

    try:
        df = df.iloc[: len(responses)]
        df["responses"] = responses
        df.to_json(out_file, lines=True, orient="records")
    except Exception as e:
        traceback.print_exc()
        with open(f"data/output/generation_1_backup_{model}.txt", "w") as f:
            for line in responses:
                print(line, file=f)

    return topics_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api",
        type=str,
        help="API to use ('openai', 'vertex', 'vllm', 'azure', 'gemini')",
        default="openai",
    )
    parser.add_argument("--model", type=str, help="Model to use", default="gpt-4")

    parser.add_argument(
        "--data",
        type=str,
        default="data/input/sample.jsonl",
        help="Data to run generation on",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt/generation_1.txt",
        help="File to read prompts from",
    )
    parser.add_argument(
        "--seed_file",
        type=str,
        default="prompt/seed_1.md",
        help="Markdown file to read seed topics from",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/output/generation_1.jsonl",
        help="File to write results to",
    )
    parser.add_argument(
        "--topic_file",
        type=str,
        default="data/output/generation_1.md",
        help="File to write topics to",
    )

    parser.add_argument(
        "--verbose", type=bool, default=False, help="Whether to print out results"
    )
    args = parser.parse_args()
    generate_topic_lvl1(
        args.api,
        args.model,
        args.data,
        args.prompt_file,
        args.seed_file,
        args.out_file,
        args.topic_file,
        args.verbose,
    )
