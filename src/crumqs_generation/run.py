import asyncio 
import argparse
from termcolor import colored
import time

from src.crumqs_generation.utils_deduplication import *
from src.crumqs_generation.create_dataset import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_name",
    type=str,
)
parser.add_argument(
    "--database_dir",
    type=str,
    default="./_database",
    help="Path to documents in the original database",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./generated_data/ood",
    help="Path to save generation process outputs",
)
parser.add_argument(
    "--report_requests_path",
    type=str,
    default="./_database/test_requests.jsonl",
    help="Path to the report requests (a JSON)"
)
parser.add_argument(
    "--load_index_path",
    type=str,
    default=None,
    help="Path to RAG index if it already exists",
)
parser.add_argument(
    "--save_index_dir",
    default="_rag_index",
    type=str,
    help="Path to save RAG index",
)
parser.add_argument(
    "--load_topics_path",
    type=str,
    default=None,
    help="Path to topics if it already exists",
)
parser.add_argument(
    "--load_nodes_path",
    type=str,
    default=None,
    help="Path to external articles if it already exists",
)
parser.add_argument(
    "--save_articles_dir",
    default="_ood_articles",
    type=str,
    help="Where to save the external articles",
)
parser.add_argument(
    "--save_dataset_path",
    type=str,
    default="unanswerable.json",
    help="Path to save final dataset of unanswerable questions",
)


parser.add_argument(
    "--rag_model_name",
    # default="gpt-4o-mini",
    # default="llama3.3-70b-instruct",
    default="meta-llama/Llama-3.3-70B-Instruct",
    type=str,
    help="RAG model",
)

parser.add_argument(
    "--num_articles_to_crawl",
    default=5,
    type=int,
    help="Number of news articles to crawl (distributed across discovered topics)",
)
parser.add_argument(
    "--num_topics",
    default=5,
    type=int,
    help="Number of topics to find",
)
parser.add_argument(
    "--rag_top_k",
    default=10,
    type=int,
    help="Number of articles to use for RAG unanswerability verification",
)
parser.add_argument(
    "--dataset_test_size",
    default=600,
    type=int,
    help="Target number of unanswerable questions in final dataset",
)
parser.add_argument(
    "--max_q_prompts_per_request",
    default=200,
    type=int,
    help="Max number of question generation prompts per report request",
)
parser.add_argument(
    "--max_unanswerable_qs_per_request",
    default=200,
    type=int,
    help="Max number of unanswerable questions in final dataset for each report request",
)
parser.add_argument(
    "--max_node_dicts",
    default=10,
    type=int,
    help="Max number of groups of document chunks to use for question generation",
)
parser.add_argument(
    "--max_per_split",
    default=5,
    type=int,
    help="Max number of groups of document chunks per ratio of internal/external documents",
)
parser.add_argument(
    "--restrict_crawl_fields",
    default=False,
    action="store_true",
)

parser.add_argument(
    "--ids_to_skip", 
    nargs='+', 
    default=[], 
    help="List of rr_ids to skip generation for (e.g, if already completed)",
)
parser.add_argument(
    "--ids_to_run", 
    nargs='+', 
    default=[], 
    help="List of rr_ids to skip generation for (e.g, if already completed)",
)
parser.add_argument(
    "--start_idx",
    type=int,
    default=None,
)


def save_args_to_file(args, during_run=False):
    
    args_dict = vars(args)
    
    if during_run:
        filepath = os.path.join(args.save_dir, "args_final.json")
    else:
        save_dir = f"{args.save_dir}/{args.exp_name}"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, "args.json")
    with open(filepath, 'w') as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
    
    print(f"All arguments saved to {filepath}")
    return filepath


async def main(args):

    start_time = time.time() 

    logging_file = os.path.join(args.save_dir, "logging.txt")

    # Create Dataset Pipeline
    pipeline = DatasetCreationPipeline(
        args.database_dir,
        logging_file=logging_file,
        chunk_size=1024,
        chunk_overlap=128,
    )
    
    # Create RAG Verifier
    log(colored("\n>>> Begin Build RAG<<<", "green"), logging_file, mode="w")
    pipeline.build_rag(
        load_index_path=args.load_index_path,
        save_index_dir=args.save_index_dir,
        model_name=args.rag_model_name,
        top_k=args.rag_top_k,
        tmpl_str=None,
    )
    
    # Extract Topics from Database
    log(colored("\n>>> Begin Gen topics<<<", "green"), logging_file)
    if args.load_topics_path:
        topics = await pipeline.find_topics(
            report_requests_path=args.report_requests_path,
            topic_size=args.num_topics,
            file_format=".json",
            load_topics_path=args.load_topics_path,
        )
    else: 
        topics = await pipeline.find_topics(
            ids_to_run=args.ids_to_run,
            report_requests_path=args.report_requests_path,
            topic_size=args.num_topics,
            file_format=".json",
            save_dir=args.save_dir,
        )
    
    # Build Nodes (Chunks) from External Articles
    log(colored("\n>>> Begin Build Nodes<<<", "green"), logging_file)
    pipeline.nodes_dict = pipeline.build_nodes(
        topics=topics,
        num_articles_to_crawl=args.num_articles_to_crawl,
        save_articles_dir=args.save_articles_dir,
        load_internal_nodes_path=f"{args.database_dir}/full_articles.json",
        load_nodes_path=args.load_nodes_path if args.load_nodes_path else None,
        restrict_crawl_fields=args.restrict_crawl_fields,
    )

    # # Extract Triples/Claims/Complex Triples
    # log(colored("\n>>> Begin Claim / Triple Extraction<<<", "green"), logging_file)
    # await pipeline.add_claims_triples_to_nodes(save_dir=args.save_dir)

    # Create dataset
    log(colored("\n>>> Begin Unanswerable Dataset Generation<<<", "green"), logging_file)
    questions, answers, contexts = await pipeline.build_datasets(
        report_requests_path=args.report_requests_path,
        test_size=args.dataset_test_size,
        save_dir=args.save_dir,
        max_q_prompts_per_request=args.max_q_prompts_per_request,
        max_unanswerable_qs_per_request=args.max_unanswerable_qs_per_request,
        max_node_dicts=args.max_node_dicts, 
        max_per_split=args.max_per_split,
        ids_to_skip=args.ids_to_skip,
        ids_to_run=args.ids_to_run,
    )
    
    # Save Dataset
    log(colored("\n>>> Save Dataset<<<", "green"), logging_file)
    pipeline.save_entire_dataset(args.save_dataset_path, questions, answers, contexts)

    total_time = (time.time() - start_time) / 60.
    log(colored(f"Done! Finished generating dataset in {total_time} minutes.", "magenta"), logging_file)


if __name__ == "__main__":

    args = parser.parse_args()

    save_args_to_file(args)

    save_dir = f"{args.save_dir}/{args.exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    args.save_articles_dir = os.path.join(save_dir, args.save_articles_dir)
    os.makedirs(args.save_articles_dir, exist_ok=True)

    args.save_index_dir = os.path.join(save_dir, args.save_index_dir)
    os.makedirs(args.save_index_dir, exist_ok=True)
    
    args.save_dataset_path = os.path.join(save_dir, args.save_dataset_path)

    args.save_dir = save_dir 
    save_args_to_file(args, during_run=True)
    asyncio.run(main(args))

    

