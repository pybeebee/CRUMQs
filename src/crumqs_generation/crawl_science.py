import os, sys
import contextlib
from sympy import N
from termcolor import colored
import fitz 
import pandas as pd

from src.crumqs_generation.utils_deduplication import log

@contextlib.contextmanager
def suppress_output(enabled=True):
    if not enabled:
        yield   # Do nothing
    else:
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

with suppress_output():
    from paperscraper.get_dumps import biorxiv, medrxiv, chemrxiv
    from paperscraper.pubmed import get_and_dump_pubmed_papers
    from paperscraper.arxiv import get_and_dump_arxiv_papers
    from paperscraper.xrxiv.xrxiv_query import XRXivQuery
    from paperscraper.scholar import get_and_dump_scholar_papers
    from paperscraper.pdf import save_pdf_from_dump

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def index_scientific_articles(
        start_date="2024-01-01",
        save_dir="./_database"
    ):
    if not os.path.exists(f"{save_dir}/medrxiv.jsonl"):
        medrxiv(start_date=start_date, save_path=f"{save_dir}/medrxiv.jsonl")
        print(colored(f"Done indexing MedrXiv articles since {start_date}.\nSaved to {save_dir}!", "magenta"))

    if not os.path.exists(f"{save_dir}/biorxiv.jsonl"):
        biorxiv(start_date=start_date, save_path=f"{save_dir}/biorxiv.jsonl")
        print(colored(f"Done indexing BiorXiv articles since {start_date}.\nSaved to {save_dir}!", "magenta"))

    if not os.path.exists(f".{save_dir}/chemrxiv.jsonl"):
        chemrxiv(start_date=start_date, save_path=f".{save_dir}/chemrxiv.jsonl")
        print(colored(f"Done indexing ChemrXiv articles since {start_date}.\nSaved to {save_dir}!", "magenta"))


def crawl_scientific_articles(
        topic: list, 
        save_dir: str,  # ".../_ood_articles"
        articles_per_source: int = 10, 
        max_crawled_articles: int = 2500,
        start_date: str="2024-01-01",
        restrict_crawl_fields=False
    ):

    logging_file = save_dir.replace("_ood_articles", "") + "logging.txt"

    articles = []
    fields = None 
    if restrict_crawl_fields:
        fields = ['title', 'abstract']
    
    index_scientific_articles(start_date=start_date)

    # Get PubMed Articles
    os.makedirs(f"{save_dir}/metadata/pubmed", exist_ok=True)
    pubmed_index_path = f'{save_dir}/metadata/pubmed/index.jsonl'
    try:
        get_and_dump_pubmed_papers(
            topic, 
            output_filepath=pubmed_index_path,
            max_results=articles_per_source,
            start_date="2024/01/01"
        )
        log(colored(f"Finished crawling PubMed articles for topic: {topic[0]}!", "magenta"), logging_file)
    except Exception as e: 
        log(colored(f"Error when getting PubMed articles: {e}", "red"), logging_file)

    # Get arXiv Articles
    os.makedirs(f"{save_dir}/metadata/arxiv", exist_ok=True)
    arxiv_index_path = f'{save_dir}/metadata/arxiv/index.jsonl'
    try:
        get_and_dump_arxiv_papers(
            topic, 
            output_filepath=arxiv_index_path,
            max_results=articles_per_source,
            start_date="2024-01-01"
        )
        log(colored(f"Finished crawling arXiv articles for topic: {topic[0]}!", "magenta"), logging_file)
    except Exception as e: 
        log(colored(f"Error when getting arXiv articles: {e}", "red"), logging_file)

    # Get Bio/Chem/Med Articles
    os.makedirs(f"{save_dir}/metadata/biorxiv", exist_ok=True)
    biorxiv_index_path = f'{save_dir}/metadata/biorxiv/index.jsonl'
    querier = XRXivQuery('./_database/biorxiv.jsonl')
    try:
        querier.search_keywords(
            topic, 
            output_filepath=biorxiv_index_path,
            fields=fields,
        )
        log(colored(f"Finished crawling BiorXiv articles for topic: {topic[0]}!", "magenta"), logging_file)
    except Exception as e: 
        log(colored(f"Error when getting BiorXiv articles: {e}", "red"), logging_file)

    os.makedirs(f"{save_dir}/metadata/chemrxiv", exist_ok=True)
    chemrxiv_index_path = f'{save_dir}/metadata/chemrxiv/index.jsonl'
    querier = XRXivQuery('./_database/chemrxiv.jsonl')
    try:
        querier.search_keywords(
            topic, 
            output_filepath=chemrxiv_index_path,
            fields=fields,
        )
        log(colored(f"Finished crawling ChemrXiv articles for topic: {topic[0]}!", "magenta"), logging_file)
    except Exception as e: 
        log(colored(f"Error when getting ChemrXiv articles: {e}", "red"), logging_file)

    os.makedirs(f"{save_dir}/metadata/medrxiv", exist_ok=True)
    medrxiv_index_path = f'{save_dir}/metadata/medrxiv/index.jsonl'
    querier = XRXivQuery('./_database/medrxiv.jsonl')
    try:
        querier.search_keywords(
            topic, 
            output_filepath=medrxiv_index_path,
            fields=fields,
        )
        log(colored(f"Finished crawling MedrXiv articles for topic: {topic[0]}!", "magenta"), logging_file)
    except Exception as e: 
        log(colored(f"Error when getting MedrXiv articles: {e}", "red"), logging_file)

    # Get Scholar Articles
    skip_scholar = True
    scholar_index_path = f'{save_dir}/metadata/scholar/index.jsonl'
    # skip_scholar = False
    # os.makedirs(f"{save_dir}/metadata/scholar", exist_ok=True)
    # scholar_index_path = f'{save_dir}/metadata/scholar/index.jsonl'
    # try:
    #     get_and_dump_scholar_papers(
    #         topic[0],
    #         output_filepath=scholar_index_path,
    #     )
    #     log(colored(f"Finished crawling Google Scholar articles for topic: {topic}!", "magenta"), logging_file)
    # except Exception as e: 
    #     log(colored(f"Failed to crawl Google Scholar articles for topic: {topic}", "red"), logging_file)
    #     log(colored("Error message: {e}", "red"), logging_file)
    #     skip_scholar = True

    # Get Article PDFs
    pdf_path = f"{save_dir}/pdfs/{topic[0]}"
    os.makedirs(pdf_path, exist_ok=True)
    for dump_path in [
        pubmed_index_path,
        arxiv_index_path,
        biorxiv_index_path,
        chemrxiv_index_path,
        medrxiv_index_path,
        scholar_index_path,
    ]:
        if skip_scholar and dump_path==scholar_index_path:
            pass
        else: 
            save_pdf_from_dump(
                dump_path=dump_path,
                pdf_path=pdf_path, 
                key_to_save='doi',
                total_papers=articles_per_source,
            )
    log(colored(f"Finished saving all article PDFs for topic: {topic[0]}!", "magenta"), logging_file)

    # Get Article Texts & Port to Json
    count = 0
    for filename in os.listdir(pdf_path):
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(pdf_path, filename))
            articles.append({
                "text": text,
                'source': 'science', 
                'doc_id': filename.replace(".pdf", ""),
            })
            count += 1
    try:
        biorxiv_papers = pd.read_json(biorxiv_index_path, lines=True)['abstract'].to_list()
        biorxiv_dois = pd.read_json(biorxiv_index_path, lines=True)['doi'].to_list()
        articles += [
            {
                "text":  x, 
                'source': 'science', 
                'doc_id': doi,
            } 
            for x, doi in zip(biorxiv_papers, biorxiv_dois)
        ]
        count += len(biorxiv_papers)
    except Exception as e: 
        log(colored("Didn't add BiorXiv abstracts as no relevant papers were found.", "yellow"), logging_file)
    log(colored(f"Finished converting PDFs to JSONs for topic: {topic[0]}!", "magenta"), logging_file)

    # Return List of Json's of Articles
    return articles


