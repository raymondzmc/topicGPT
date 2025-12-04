import pandas as pd
from topicgpt_python.utils import *
from tqdm import tqdm
import regex
import traceback
from sentence_transformers import SentenceTransformer, util
import argparse
import os
from anytree import Node, RenderTree

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
    num_topics=None,  # Target number of topics to generate (None = no limit)
):
    """
    Generate topics from documents using LLMs.
    
    Parameters:
    - topics_root: TopicTree object
    - topics_list: List of topic strings
    - context_len: Maximum context length for the model
    - docs: List of documents to process
    - seed_file: Path to seed file
    - api_client: API client object
    - generation_prompt: Prompt template for generation
    - temperature: Sampling temperature
    - max_tokens: Maximum tokens for response
    - top_p: Top-p sampling parameter
    - verbose: Whether to print verbose output
    - early_stop: Number of consecutive duplicates before stopping (ignored if num_topics is set)
    - num_topics: Target number of topics to generate. If set, generation stops when this 
                  number is reached. Common values: 25, 50, 75, 100. None means no limit.
    
    Returns:
    - responses: List of model responses
    - topics_list: Updated list of topic strings
    - topics_root: Updated TopicTree object
    """
    responses = []
    running_dups = 0
    topic_format = regex.compile(r"^\[(\d+)\] ([\w\s]+):(.+)")

    for i, doc in enumerate(tqdm(docs)):
        # Check if we've reached the target number of topics
        current_topic_count = len(list(topics_root.root.children))
        if num_topics is not None and current_topic_count >= num_topics:
            if verbose:
                print(f"Reached target of {num_topics} topics. Stopping generation.")
            break
            
        prompt = prompt_formatting(
            generation_prompt,
            api_client,
            doc,
            seed_file,
            topics_list,
            context_len,
            verbose,
        )

        try:
            response = api_client.iterative_prompt(
                prompt, max_tokens, temperature, top_p=top_p, verbose=verbose
            )

            # Parsing topics and organizing topic tree
            topics = [t.strip() for t in response.split("\n")]
            for t in topics:
                # Check again after parsing each topic (in case multiple topics per response)
                current_topic_count = len(list(topics_root.root.children))
                if num_topics is not None and current_topic_count >= num_topics:
                    if verbose:
                        print(f"Reached target of {num_topics} topics. Stopping generation.")
                    responses.append(response)
                    return responses, topics_list, topics_root
                    
                if not regex.match(topic_format, t):
                    print(f"Invalid topic format: {t}. Skipping...")
                    continue
                groups = regex.match(topic_format, t)
                lvl, name, desc = int(groups[1]), groups[2].strip(), groups[3].strip()

                if lvl != 1:
                    print(f"Lower level topics are not allowed: {t}. Skipping...")
                    continue
                dups = topics_root.find_duplicates(name, lvl)

                if (
                    dups
                ):  # Implement early stopping if no new topics are generated for a while
                    dups[0].count += 1
                    running_dups += 1
                    # Only apply early stopping if num_topics is not set
                    if num_topics is None and running_dups > early_stop:
                        if verbose:
                            print(f"Early stopping: {running_dups} consecutive duplicates.")
                        return responses, topics_list, topics_root
                else:
                    topics_root._add_node(lvl, name, 1, desc, topics_root.root)
                    topics_list = topics_root.to_topic_list(desc=False, count=False)
                    running_dups = 0
                    
                    if verbose:
                        current_count = len(list(topics_root.root.children))
                        target_str = f"/{num_topics}" if num_topics else ""
                        print(f"New topic added: {name} (Total: {current_count}{target_str})")

            if verbose:
                print(f"Topics: {response}")
                print("--------------------")
            responses.append(response)

        except Exception as e:
            traceback.print_exc()
            responses.append("Error")
            break

    return responses, topics_list, topics_root


def generate_topic_lvl1(
    api, model, data, prompt_file, seed_file, out_file, topic_file, verbose,
    num_topics=None, early_stop=100
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
    - num_topics (int): Target number of topics to generate. Common values: 25, 50, 75, 100.
                        If None, generation continues until early stopping or all docs processed.
    - early_stop (int): Number of consecutive duplicates before stopping (default: 100).
                        Ignored if num_topics is set.

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
        if num_topics:
            print(f"Target number of topics: {num_topics}")
        else:
            print(f"Early stopping after {early_stop} consecutive duplicates")
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
        early_stop=early_stop,
        num_topics=num_topics,
    )

    # Save generated topics
    topics_root.to_file(topic_file)
    
    # Print summary
    final_topic_count = len(list(topics_root.root.children))
    print(f"\n=== Topic Generation Complete ===")
    print(f"Total topics generated: {final_topic_count}")
    if num_topics:
        print(f"Target was: {num_topics}")
    print(f"Topics saved to: {topic_file}")
    print("=================================\n")

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
        "--num_topics",
        type=int,
        default=None,
        help="Target number of topics to generate (e.g., 25, 50, 75, 100). If not set, uses early stopping.",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=100,
        help="Number of consecutive duplicates before stopping (default: 100). Ignored if --num_topics is set.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to print out results"
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
        num_topics=args.num_topics,
        early_stop=args.early_stop,
    )
