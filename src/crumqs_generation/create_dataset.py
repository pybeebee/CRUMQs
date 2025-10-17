import json
import random
import os
import pickle 
from termcolor import colored
import asyncio 
import pandas as pd
from ast import literal_eval
import json_repair

from collections import defaultdict
from itertools import combinations, product

from tqdm import tqdm

from langchain.text_splitter import TokenTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import JSONLoader

from src.crumqs_generation.crawler import crawl_gnews_articles
from src.crumqs_generation.crawl_science import crawl_scientific_articles, suppress_output

with suppress_output():
    from ragas.testset.docstore import Node
    from ragas.testset.docstore import Document
    from ragas.llms import LangchainLLMWrapper

from src.crumqs_generation.rag import RAGVerifier
from src.crumqs_generation.prompts import PROMPT_REGISTRY, DOC_PROMPTS, CLAIM_PROMPTS, CRITERION_NAMES
from src.crumqs_generation.prompts.extraction import *
from src.crumqs_generation.prompts.generation import *
from src.crumqs_generation.prompts.verification import *
from src.crumqs_generation.prompts.pipeline import *

from src.crumqs_generation.utils import *
from src.crumqs_generation.utils_deduplication import *
from src.crumqs_generation.utils_inference import *

from src.crumqs_generation.dataset_format import *


class DatasetCreationPipeline:
    
    def __init__(
        self,
        database_dir,
        logging_file,
        chunk_size=1024,
        chunk_overlap=24,
    ):
    
        self.logging_file = logging_file
        self.database_dir = database_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.nodes_dict = None
        self.internal_nodes_dict = None
        self.rag = None
        self.rr_ids = None

    ### FXN: Instantiate RAG Verifier
    def build_rag(self, load_index_path, save_index_dir, model_name, top_k, tmpl_str=None):

        try:
            self.rag = RAGVerifier(
                folder=self.database_dir,           # Database source
                load_index_path=save_index_dir,     # RAG index
                save_index_dir=save_index_dir,      # Index save path
                model_name=model_name,              # RAG model
                top_k=top_k,
            )
        except:
            os.makedirs(save_index_dir, exist_ok=True)
            self.rag = RAGVerifier(
                folder=self.database_dir,           # Database source
                load_index_path=load_index_path,    # RAG index
                save_index_dir=save_index_dir,      # Index save path
                model_name=model_name,              # RAG model
                top_k=top_k,
            )

        if tmpl_str:
            self.rag.update_template(tmpl_str)

    ### FXN: Find Topics
    async def find_topics(self, report_requests_path, topic_size=1, file_format=".txt", load_topics_path=None, save_dir=None, ids_to_run=None):

        topics = set()
        self.rr_id_to_topics = {}
        self.topics_to_rr_id = {}

        # Read Topics Set if Already Created
        if not load_topics_path and os.path.exists(f"{save_dir}/topics.pkl"):
            load_topics_path = f"{save_dir}/topics.pkl"
        if load_topics_path: 
            save_dir = load_topics_path.replace("/topics.pkl", "")
            with open(load_topics_path, 'rb') as f:
                topics = pickle.load(f)
            log(colored(f"Loaded {len(topics)} topics from file!", "yellow"), self.logging_file)
            log(topics, self.logging_file)

            # Load rr_id->topics Mapping
            rr_topics_path = f"{save_dir}/rr_id_to_topics.json"
            try:
                with open(rr_topics_path, 'r') as f:
                    self.rr_id_to_topics = json.load(f)
            except Exception as e:
                log(colored(f"No rr_id->topics map file found: {e}","red"), self.logging_file)
                raise ValueError

            # Load topics->rr_id Mapping
            topics_rr_path = f"{save_dir}/topics_to_rr_id.json"
            try:
                with open(topics_rr_path, 'r') as f:
                    self.topics_to_rr_id = json.load(f)
            except Exception as e:
                log(colored(f"No topics->rr_id map file found: {e}","red"), self.logging_file)
                raise ValueError
            
            return topics

        # Read in the Report Requests
        request_df = pd.read_json(report_requests_path, lines=True)
        ids_to_run_int = [int(i) for i in ids_to_run]
        request_df = request_df[request_df["request_id"].isin(ids_to_run_int)]
        themes = dict(zip(request_df["request_id"], request_df["title"]))  
        rrs = dict(zip(request_df["request_id"], request_df["problem_statement"]))      
        uss = dict(zip(request_df["request_id"], request_df["background"]))      
        report_requests = request_df.problem_statement.to_list()      
        user_stories = request_df.background.to_list()    
        request_ids = request_df.request_id.to_list()
        
        # Extract Topics from Report Requests + User Stories
        prompts = [
            topic_extraction_prompt.format(user_story=user_story, report_request=rr).prompt_str
            for user_story, rr in zip(user_stories, report_requests)
        ]
        outputs = asyncio.run(
            get_response_batch(
                system_prompt=sys_prompt, 
                user_prompts=prompts, 
                provider=ModelProvider.LITELLM_LOCAL, 
                temperature=0., 
                use_cache=False,
            )
        )
        response = [
            json.loads(output)['topics'] 
            for output in outputs
        ]
        extracted_topics = [
            item.strip() 
            for topic_list in response 
            if isinstance(topic_list, str)
            for item in topic_list.split(",")
        ]
        topics = topics.union(set(extracted_topics))

        # with open(f"{save_dir}/request_guided_topics.pkl", 'wb') as f:
        #     pickle.dump(topics, f)
        # with open(f'{save_dir}/request_guided_topics.txt', 'w') as f:
        #     f.write(str(topics))

        # Save Topics with Request ID
        for request_id, output in zip(request_ids, outputs):
            rr_id = str(request_id)
            if rr_id not in ids_to_run:
                continue
            topic_list = [
                item.strip() 
                for topics in [json.loads(output)['topics']]
                if isinstance(topics, str)
                for item in topics.split(",")
            ]
            self.rr_id_to_topics[rr_id] = topic_list

        # Save Topics with Correspnding Requests/User Stories
        topic_triples = [
            (str(rr_id), us, rr, topic)
            for rr_id, rr, us, output in zip(request_ids, report_requests, user_stories, outputs)
            for topic in [
                item.strip()
                for topics in [json.loads(output)['topics']]
                if isinstance(topics, str)
                for item in topics.split(",")
            ]
        ]
        with open(f'{save_dir}/topic_rr_us_map.pkl', 'wb') as f:
            pickle.dump(topic_triples, f)
        self.rr_ids = set(rr_id for rr_id, _, _, _ in topic_triples)

        # Extract Topics from RR Themes + Gold Documents

        # Load Gold Documents
        file_paths = [
            os.path.join(self.database_dir, f) 
            for f in os.listdir(self.database_dir)
            if os.path.isfile(os.path.join(self.database_dir, f)) and f.endswith(file_format)
        ]
        random.shuffle(file_paths)

        # Prepare Prompts for Batch Topic Extraction
        prompts = []
        rr_id_per_prompt = []
        for file_path in tqdm(file_paths):

            filename = os.path.basename(file_path)
            if "full_articles" not in filename:
                with open(file_path, "r", encoding="utf-8") as file:

                    # If JSON
                    if filename.endswith(".json"):
                        try:
                            content = json.load(file).get("text", "")
                        except:
                            continue
                    
                    # If .txt
                    else:
                        content = file.read()

                # Get Theme
                rr_id = filename.split("__")[0]
                if rr_id not in ids_to_run:
                    continue
                theme = themes[int(rr_id)]
                rr = rrs[int(rr_id)]
                us = uss[int(rr_id)]

                # Format Prompt
                prompts.append(
                    doc_grounded_topic_extraction_prompt.format(theme=theme, text=content).prompt_str
                )
                rr_id_per_prompt.append(rr_id)
        
        # Extract Topics from Each Document
        outputs = asyncio.run(
            get_response_batch_chunked(
                system_prompt=sys_prompt, 
                user_prompts=prompts, 
                provider=ModelProvider.LITELLM_LOCAL, 
                temperature=0., 
                use_cache=False,
            )
        )
        response = [
            json_repair.repair_json(output, return_objects=True)['topics']
            for output in outputs
        ]
        extracted_topics = [
            item.strip() 
            for topic_list in response 
            for item in topic_list.split(",")
        ]

       # Save Topics with Request ID
        for rr_id, output in zip(rr_id_per_prompt, outputs):
            topic_list = [item.strip() for item in json_repair.repair_json(output, return_objects=True)['topics'].split(",")]
            if rr_id not in self.rr_id_to_topics:
                self.rr_id_to_topics[rr_id] = []
            self.rr_id_to_topics[rr_id].extend(topic_list)

        # Add Topics to List
        topics = topics.union(set(extracted_topics))
        orig_size = len(topics)
        topics, removed_topics = self.deduplicate_topics(topics)

        # Remove Duplicate Topics from rr_id->topics map
        for rr_id in self.rr_id_to_topics:
            self.rr_id_to_topics[rr_id] = [
                topic 
                for topic in self.rr_id_to_topics[rr_id] 
                if topic not in removed_topics
            ]
        
        # Create topic->rr_id Map
        for rr_id, topic_list in self.rr_id_to_topics.items():
            for topic in topic_list:
                self.topics_to_rr_id[topic] = rr_id

        # Limit Topics to Maximum if Necessary
        if len(topics) > topic_size:
            rr_ids = list(self.rr_id_to_topics.keys())
            topics_per_rr = topic_size // len(rr_ids)
            selected_topics = []
            for i, rr_id in enumerate(rr_ids):
                topics_for_rr = random.sample(
                    self.rr_id_to_topics[rr_id], 
                    min(topics_per_rr, len(self.rr_id_to_topics[rr_id]))
                )
                selected_topics += topics_for_rr
            topics = set(selected_topics)
            log(colored(f"Found {len(topics)} topics ({topics_per_rr} per rr_id), filtered down from original {orig_size} topics!:", "yellow"), self.logging_file)

        # Print Warning if Not Enough Topics
        if len(topics) < topic_size:
            log(colored(f"Found fewer than requested # of topics ({len(topics)} total)...", "yellow"), self.logging_file)

        # Save Topics to File
        with open(f"{save_dir}/topics.pkl", 'wb') as f:
            pickle.dump(topics, f)
        with open(f'{save_dir}/topics.txt', 'w') as f:
            f.write(str(topics))
        with open(f"{save_dir}/rr_id_to_topics.json", 'w') as f:
            json.dump(self.rr_id_to_topics, f, indent=4)
        with open(f"{save_dir}/topics_to_rr_id.json", 'w') as f:
            json.dump(self.topics_to_rr_id, f, indent=4)

        log(topics, self.logging_file)
        return topics
    
    ### FXN: Deduplicate Topics
    def deduplicate_topics(self, topics):

        topics_list = list(topics)
        request_id = "dedup_topics"

        deduplicator = FindPairs(
            embedding_model='BAAI/bge-large-en-v1.5',
            similarity_threshold=0.95,
        ) 
        deduplicator.find_similar_pairs(
            texts1=topics_list, 
            request_id=request_id,
        )
        
        # Verify with LLM
        asyncio.run(deduplicator.check_pairs_with_llm(request_id))

        # Get Filtered & Verified Overlapping Pairs
        pairs = deduplicator.get_pairs(request_id, verified_only=True)

        # Group Related Topics w/ Union-Find
        pairs_for_grouping = [(pair["q1"], pair["q2"]) for pair in pairs]
        group_mapping, _ = group_items(pairs_for_grouping)
        inverse_groups = invert_group_mapping(group_mapping)

        # Choose Representative Topic Per Group (Longest)
        deduplicated_topics = set()
        for group in inverse_groups.values():
            rep = max(group, key=len) 
            deduplicated_topics.add(rep)

        # Add Singleton Topics
        singletons = topics - set(group_mapping.keys())
        deduplicated_topics.update(singletons)

        # Track Removed Topics
        removed_topics = topics - deduplicated_topics
        assert(len(removed_topics) + len(deduplicated_topics) == len(topics))

        return deduplicated_topics, removed_topics

    ### FXN: Get Document Chunks
    def build_nodes(self, topics, num_articles_to_crawl, save_articles_dir, load_internal_nodes_path="./_database/translated/full_articles.json", load_nodes_path=None, restrict_crawl_fields=False):
        
        # Create Chunker
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        # Set Up Internal Corpus Nodes
        loader = JSONLoader(
            file_path=load_internal_nodes_path,
            jq_schema=".[]",
            content_key="text",
            text_content=False,
            metadata_func=custom_metadata,
        )
        internal_documents = loader.load()
        internal_nodes_dict = self.load_text_chunk(internal_documents, splitter)
        self.internal_nodes_dict = internal_nodes_dict

        # Use Existing External Articles (If There)
        try:
            os.makedirs(save_articles_dir, exist_ok=True)

            if not load_nodes_path:
                load_nodes_path = f"{save_articles_dir}/full_articles.json"

            # Load Documents
            loader = JSONLoader(
                file_path=load_nodes_path,
                jq_schema=".[]",
                content_key="text",
                text_content=False,
                metadata_func=custom_metadata,
            )
            documents = loader.load()

            # Convert Articles to Chunks/Nodes for Retrieval
            nodes_dict = self.load_text_chunk(documents, splitter)
        
        # Get External Articles
        except:

            articles_total, successful_rate = [], []
            articles_per_source = max(num_articles_to_crawl // (2*len(topics)), 1)

            # Get Articles for Each Topic
            for topic in topics:

                if "and" in topic[:4]:
                    continue

                rr_id = self.topics_to_rr_id[topic]
                log(colored(f"Retrieving articles for topic {rr_id}: {topic}", "cyan"), self.logging_file)

                try:
                    articles_news = crawl_gnews_articles(
                        keywords=[topic],
                        save_dir=save_articles_dir,
                        articles_per_feed=articles_per_source*3,
                    )
                    articles_science = crawl_scientific_articles(
                        topic=[topic],
                        save_dir=save_articles_dir,
                        articles_per_source=articles_per_source,
                        restrict_crawl_fields=restrict_crawl_fields,
                    )
                    for x in articles_news: 
                        x['rr_id'] = rr_id
                        x['topic'] = topic
                    for x in articles_science: 
                        x['rr_id'] = rr_id
                        x['topic'] = topic
                    articles_total += articles_news
                    articles_total += articles_science
                    topic_articles = articles_news + articles_science
                    length = len(topic_articles)
                    successful_rate.append(1)

                    # Save Articles by Topic
                    if save_articles_dir:
                        with open(
                            f"{save_articles_dir}/{topic}__{length}_articles.json", "w"
                        ) as f:
                            json.dump(topic_articles, f)

                except Exception as e:
                    log(colored(f"Error during node creation for topic {topic}: {e}", "red"), self.logging_file)
                    successful_rate.append(0)

            log(colored(f"External article retrieval success Rate is: {sum(successful_rate) / len(successful_rate)}", "yellow"), self.logging_file)

            # Save Articles
            with open(f"{save_articles_dir}/full_articles.json", "w") as f:
                json.dump(articles_total, f)
            
            # Convert Articles to Documents
            documents = []
            for article in articles_total:
                documents.append(
                    Document(
                        page_content=article["text"], 
                        metadata={
                            "source": article['source'],
                            "doc_id": article['doc_id'],
                            "topic": article['topic'],
                            "rr_id": str(article['rr_id']),
                        }
                    )
                )
            
            # Convert Articles to Chunks/Nodes for Retrieval
            nodes_dict = self.load_text_chunk(documents, splitter)

        log(colored(f"Found {len(documents)} articles, broken into {len(nodes_dict)} chunks!", "yellow"), self.logging_file)
        return nodes_dict

    ### FXN: Extract claims and entity-relation lists 
    async def add_claims_triples_to_nodes(self, save_dir):

        filepath = f"{save_dir}/nodes_with_claims_triples.pkl"
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.nodes_dict = pickle.load(f)
            log(colored(f"Loaded node dict with pre-extracted claims and triples from file!", "yellow"), self.logging_file)

        else:

            nodes = list(self.nodes_dict.values())

            # Create prompts
            claim_prompts = [
                claim_extraction.format(document=node.page_content)
                for node in nodes
            ]
            triple_prompts = [
                triple_extraction.format(document=node.page_content)
                for node in nodes
            ]
            complex_prompts = [
                complex_triple_extraction.format(document=node.page_content)
                for node in nodes
            ]

            # Send batch prompts
            claim_outputs = asyncio.run(
                get_response_batch_chunked(
                    system_prompt=sys_prompt,
                    user_prompts=claim_prompts,
                    provider=ModelProvider.LITELLM_LOCAL, 
                    temperature=0., 
                    use_cache=False,
                    max_tokens=5000,
                )
            )
            triple_outputs = asyncio.run(
                get_response_batch_chunked(
                    system_prompt=sys_prompt,
                    user_prompts=triple_prompts,
                    provider=ModelProvider.LITELLM_LOCAL, 
                    temperature=0., 
                    use_cache=False,
                    max_tokens=5000,
                )
            )
            
            complex_outputs = asyncio.run(
                get_response_batch_chunked(
                    system_prompt=sys_prompt,
                    user_prompts=complex_prompts,
                    provider=ModelProvider.LITELLM_LOCAL, 
                    temperature=0., 
                    use_cache=False,
                    max_tokens=5000,
                )
            )

            def try_eval(response):
                try:
                    return literal_eval(response.split("=")[0].strip())
                except Exception:
                    log(colored(f"Failed to parse claims, entities, and complex entities from response: {response}", "red"), self.logging_file)
                    return []

            log(colored(f"Parsing extracted claims and entities...", "yellow"), self.logging_file)

            # Add Directly to Node Metadata
            for node, claim_resp, triple_resp, complex_resp in zip(nodes, claim_outputs, triple_outputs, complex_outputs):
                node.metadata["claims"] = try_eval(claim_resp)
                node.metadata["triples"] = try_eval(triple_resp)
                node.metadata["complex_triples"] = try_eval(complex_resp)

            # Claim Verification
            log(colored(f"Verifying claims are entailed by original doc chunks...", "yellow"), self.logging_file)
            claim_verif_prompts = [
                claim_verif_factcg.format(document=node.page_content, claim=claim)
                for node in nodes
                for claim in node.metadata.get("claims", [])
            ]
            judgments = asyncio.run(
                get_response_batch_chunked(
                    system_prompt=sys_prompt,
                    user_prompts=claim_verif_prompts,
                    provider=ModelProvider.LITELLM_LOCAL, 
                    temperature=0., 
                    use_cache=False,
                    max_tokens=2,
                )
            )
            judgment_iter = iter(judgments)
            for node in nodes:
                claims = node.metadata["claims"]
                verified = []
                for claim in claims:
                    if next(judgment_iter).strip().upper()=="YES":
                        verified.append(claim)
                node.metadata["claims"] = verified

            # Triple Verification
            log(colored(f"Verifying simple triples are entailed by original doc chunks...", "yellow"), self.logging_file)
            triple_verif_prompts = [
                triple_verif_factcg.format(document=node.page_content, triple=triple)
                for node in nodes
                for triple in node.metadata.get("triples", [])
            ]
            judgments = asyncio.run(
                get_response_batch_chunked(
                    system_prompt=sys_prompt,
                    user_prompts=triple_verif_prompts,
                    provider=ModelProvider.LITELLM_LOCAL, 
                    temperature=0., 
                    use_cache=False,
                    max_tokens=2,
                )
            )
            judgment_iter = iter(judgments)
            for node in nodes:
                triples = node.metadata["triples"]
                verified = []
                for triple in triples:
                    if next(judgment_iter).strip().upper()=="YES":
                        verified.append(triple)
                node.metadata["triples"] = verified

            # Complex Triple Verification
            log(colored(f"Verifying complex triples are entailed by original doc chunks...", "yellow"), self.logging_file)
            complex_triple_verif_prompts = [
                triple_verif_factcg.format(document=node.page_content, triple=triple)
                for node in nodes
                for triple in node.metadata.get("complex_triples", [])
            ]
            judgments = asyncio.run(
                get_response_batch_chunked(
                    system_prompt=sys_prompt,
                    user_prompts=complex_triple_verif_prompts,
                    provider=ModelProvider.LITELLM_LOCAL, 
                    temperature=0., 
                    use_cache=False,
                    max_tokens=2,
                )
            )
            judgment_iter = iter(judgments)
            for node in nodes:
                triples = node.metadata["complex_triples"]
                verified = []
                for triple in triples:
                    if next(judgment_iter).strip().upper()=="YES":
                        verified.append(triple)
                node.metadata["complex_triples"] = verified

            log(colored(f"Finished parsing and verifying extracted claims and entities! Saved into nodes' metadata.", "yellow"), self.logging_file)

            with open(filepath, 'wb') as file:
                pickle.dump(self.nodes_dict, file)

    ### HELPER: Load Chunks / Nodes from Documents
    def load_text_chunk(self, documents, splitter):
    
        # Load Documents
        docs = [Document.from_langchain_document(doc) for doc in documents]

        # Chunk Documents & Convert to Nodes (Units for Retrieval)
        nodes = [ Node.from_langchain_document(d) for d in splitter.transform_documents(docs) ]

        # Store Nodes as Dict
        nodes_dict = dict()
        for idx, node in enumerate(nodes):
            nodes_dict[idx] = node

        return nodes_dict
    
    ### Helper: Confirm Question is Unanwerable with Original Documents
    async def verify_unanswerable_query(self, question, answer, ground_truth):

        # Compare Answer vs. Ground Truth 
        prompt = unanswerable_check_prompt.format(
            question=question,
            answer=answer, 
            ground_truth=ground_truth,
        ).prompt_str
        output = asyncio.run(
            get_response(
                system_prompt=sys_prompt, 
                user_prompt=prompt, 
                provider=ModelProvider.LITELLM_LOCAL, 
                temperature=0., 
                use_cache=False,
            )
        )
        # response = json.loads(output)
        response = safe_extract_output(output, placeholder_response={"reason": "RAG answerability judgment error", "verdict": "judgment error"})

        return response['reason'], response["verdict"]

    ### HELPER: Save Unanswerable Question Dataset
    def save_entire_dataset(self, save_dataset_path, questions, answers, contexts):

        assert len(questions)==len(answers)==len(contexts)
        
        dataset = {
            "question": questions,
            "ground_truth": answers,
            "contexts": contexts,
        }

        with open(save_dataset_path, "w") as f:
            json.dump(dataset, f, indent=4)

        log(colored(f"Saved dataset to {save_dataset_path}!", "yellow"), self.logging_file)

    ### HELPER: Save Questions for Current rr_id to NuggetBank Form
    def save_nuggetbank(self, save_dir, rr_id, report_requests_path, mode, no_print=False):

        # Ensure rr_id is Valid
        if rr_id not in self.rr_id_to_question_metadata:
            raise ValueError(f"rr_id '{rr_id}' not found in question metadata")
        
        creator = Creator(
            is_human=False,
            llm_model="meta-llama/Llama-3.3-70B-Instruct",
            format="v3",
        )
        
        # Get Report Request Title
        request_df = pd.read_json(report_requests_path, lines=True)
        idx = request_df["request_id"]==int(rr_id)
        title = request_df.loc[idx, "title"].values[0]
        rr = request_df.loc[idx, "problem_statement"].values[0]
        us = request_df.loc[idx, "background"].values[0]

        # Create NuggetBanks
        nugget_bank = NuggetBank(
            query_id=rr_id,
            title_query=title,
            full_query=rr,
            full_background=us,
            creator=[creator],
            metadata={
                "description": f"All multihop nuggets for rr_id {rr_id}, mode {mode}"
            }
        )
        answerable_nugget_bank = NuggetBank(
            query_id=rr_id,
            title_query=title,
            full_query=rr,
            full_background=us,
            creator=[creator],
            metadata={
                "description": f"*Answerable* multihop nuggets for rr_id {rr_id}, mode {mode}",
                "mode": mode,
            }
        )
        unanswerable_nugget_bank = NuggetBank(
            query_id=rr_id,
            title_query=title,
            full_query=rr,
            full_background=us,
            creator=[creator],
            metadata={
                "description": f"*Unanswerable* multihop nuggets for rr_id {rr_id}, mode {mode}",
                "mode": mode,
            }
        )
        murky_nugget_bank = NuggetBank(
            query_id=rr_id,
            title_query=title,
            full_query=rr,
            full_background=us,
            creator=[creator],
            metadata={
                "description": f"Murky multihop nuggets for rr_id {rr_id}, mode {mode}",
                "mode": mode,
            }
        )
        
        # Iterate Over Topics & Questions
        nuggets = []
        answerable_nuggets = []
        unanswerable_nuggets = []
        murky_nuggets = []
        questions = self.rr_id_to_question_metadata[rr_id]
        for question, question_metadata in questions.items():

            # Create Nugget
            nugget = NuggetQuestion(
                query_id=rr_id,
                question=question,
                answers={
                    "answer": Answer(
                        answer=question_metadata['external_answer'],
                        metadata={
                            'cot_annotation': question_metadata['cot_annotation'],
                            'scores': question_metadata['scores'],
                        }
                    ),
                    "rag_answer": Answer(
                        answer=question_metadata['rag_answer'],
                        metadata={
                            'reason': question_metadata['reason'],
                        }
                    ),
                },
                aggregator_type=AggregatorType.OR,
                metadata={
                    'answerability': question_metadata['answerability'],
                    'generation_template': question_metadata['generation_template'],
                    'generation_kwargs': question_metadata['generation_kwargs'],
                    'full_context': question_metadata['context'],
                    'full_internal_context': question_metadata['internal_context'],
                    'sources': question_metadata['sources'],
                    'doc_ids': question_metadata['doc_ids'],
                }
            )

            if question_metadata['answerability']=='answerable':
                answerable_nuggets.append(nugget)
            elif question_metadata['answerability']=='unanswerable':
                unanswerable_nuggets.append(nugget)
            else: 
                murky_nuggets.append(nugget)
            
            # Add Nugget to List
            nuggets.append(nugget)
        
        # Add Nuggets to Banks
        nugget_bank.add_nuggets(nuggets)
        answerable_nugget_bank.add_nuggets(answerable_nuggets)
        unanswerable_nugget_bank.add_nuggets(unanswerable_nuggets)
        murky_nugget_bank.add_nuggets(murky_nuggets)
        
        # Save NuggetBanks
        save_path = os.path.join(save_dir, f"{rr_id}__multihop_nugget_bank__mode_{mode}.json")
        answerable_save_path = os.path.join(save_dir, f"{rr_id}__multihop_nugget_bank__mode_{mode}__answerable.json")
        unanswerable_save_path = os.path.join(save_dir, f"{rr_id}__multihop_nugget_bank__mode_{mode}_unanswerable.json")
        murky_save_path = os.path.join(save_dir, f"{rr_id}__multihop_nugget_bank__mode_{mode}__murky.json")
        write_nugget_bank_json(nugget_bank, save_path)
        write_nugget_bank_json(answerable_nugget_bank, answerable_save_path)
        write_nugget_bank_json(unanswerable_nugget_bank, unanswerable_save_path)
        write_nugget_bank_json(murky_nugget_bank, murky_save_path)
        
        if not no_print:
            log(colored(f"Saved NuggetBank for rr_id {rr_id} generated under mode {mode} to: {save_dir}\nTotal answerable nuggets: {len(answerable_nuggets)}\nTotal unanswerable nuggets: {len(unanswerable_nuggets)}\nTotal murky nuggets: {len(murky_nuggets)}\nTOTAL NUGGETS: {len(nuggets)}", "yellow"), self.logging_file)

    ### HELPER: Select Combo's of Internal & External Docs for Generation
    def select_documents(self, current_nodes, current_internal_nodes, max_node_dicts, max_per_split, seed=32) -> list:
        
        random.seed(seed)

        selected_node_dicts = []
        used_current = set()
        used_internal = set()

        current_ids = list(current_nodes.keys())
        internal_ids = list(current_internal_nodes.keys())

        # Generate All Valid (num_current, num_internal) Combo's Where Total Is 2–6 and Both ≥1
        valid_splits = [(i, j) for i in range(1, 6) for j in range(1, 6) if 2 <= i + j <= 6]

        # Helper: Sample w/o Replacement But Refill If Needed
        def sample_nodes(node_list, k, exclude=set()):
            available = list(set(node_list) - exclude)
            if len(available) >= k:
                return random.sample(available, k)
            else:
                # Refill and sample
                return random.sample(node_list, k)

        # Stage 1: Ensure All Nodes Used
        uncovered_current = set(current_ids)
        uncovered_internal = set(internal_ids)

        while uncovered_current or uncovered_internal:
            for num_current, num_internal in valid_splits:
                cur_sample = sample_nodes(current_ids, num_current, exclude=used_current)
                int_sample = sample_nodes(internal_ids, num_internal, exclude=used_internal)

                node_dict = {}
                for cid in cur_sample:
                    node_dict[f"external_{cid}"] = current_nodes[cid]
                    used_current.add(cid)
                    uncovered_current.discard(cid)
                for iid in int_sample:
                    node_dict[f"internal_{iid}"] = current_internal_nodes[iid]
                    used_internal.add(iid)
                    uncovered_internal.discard(iid)

                selected_node_dicts.append(node_dict)

                if len(selected_node_dicts) >= max_node_dicts:
                    break

                if not uncovered_current and not uncovered_internal:
                    break

        # Stage 2: Fill To max_per_split Per Split
        split_counter = defaultdict(int)
        while any(split_counter[split] < max_per_split for split in valid_splits) and len(selected_node_dicts) < max_node_dicts:
            for num_current, num_internal in valid_splits:

                if len(selected_node_dicts) >= max_node_dicts:
                    break

                if split_counter[(num_current, num_internal)] >= max_per_split:
                    continue

                cur_sample = random.sample(current_ids, min(num_current, len(current_ids)))
                int_sample = random.sample(internal_ids, min(num_internal, len(internal_ids)))

                # Only create node_dict if it has enough nodes
                if len(cur_sample) < num_current or len(int_sample) < num_internal:
                    continue

                node_dict = {}
                for cid in cur_sample:
                    node_dict[cid] = current_nodes[cid]
                for iid in int_sample:
                    node_dict[iid] = current_internal_nodes[iid]

                selected_node_dicts.append(node_dict)
                split_counter[(num_current, num_internal)] += 1

        return selected_node_dicts

    ### HELPER: Create Generation Context (Documents)
    def prepare_context(self, selected_nodes: dict, internal_docs_only=False) -> str:

        nodes = list(selected_nodes.values())

        contexts = []
        for idx, node in enumerate(nodes):
            if not internal_docs_only or (internal_docs_only and node.metadata['source'] not in ['gnews', 'science']):
                i = idx+1
                text = node.page_content.replace("\n", " ")
                node_contents = f"Article {i}: {text}"
                contexts.append(node_contents)

        return "\n\n".join(contexts)

    ### HELPER: Create Generation Context (Claims)
    def prepare_claims(self, selected_nodes: dict, internal_docs_only=False) -> str:

        nodes = list(selected_nodes.values())

        contexts = []
        for idx, node in enumerate(nodes):
            if not internal_docs_only or (internal_docs_only and node.metadata['source'] not in ['gnews', 'science']):
                i = idx+1
                claims_list = getattr(node, 'metadata')['claims']
                node_contents = f"Claims from Article {i}: {claims_list}"
                contexts.append(node_contents)

        return "\n\n".join(contexts)
         
    ### HELPER: Create Generation Context (Entity-Relation Triples)
    def prepare_triples(self, selected_nodes: dict, internal_docs_only=False) -> str:

        nodes = list(selected_nodes.values())

        contexts = []
        for idx, node in enumerate(nodes):
            if not internal_docs_only or (internal_docs_only and node.metadata['source'] not in ['gnews', 'science']):
                i = idx+1
                claims_list = getattr(node, 'metadata')['triples']
                node_contents = f"Claims from Article {i}: {claims_list}"
                contexts.append(node_contents)

        return "\n\n".join(contexts)

    ### HELPER: Create Generation Context (Natural Language Entity-Relation Triples)
    def prepare_complex_triples(self, selected_nodes: dict, internal_docs_only=False) -> str:
    
        nodes = list(selected_nodes.values())

        contexts = []
        for idx, node in enumerate(nodes):
            if not internal_docs_only or (internal_docs_only and node.metadata['source'] not in ['gnews', 'science']):
                i = idx+1
                claims_list = getattr(node, 'metadata')['complex_triples']
                node_contents = f"Claims from Article {i}: {claims_list}"
                contexts.append(node_contents)

        return "\n\n".join(contexts)

    ### FXN: Create Unanswerable Multihop Dataset
    async def build_datasets(self, report_requests_path, test_size, save_dir, max_q_prompts_per_request=20, max_unanswerable_qs_per_request=None, max_node_dicts=10, max_per_split=5, ids_to_skip=[], ids_to_run=[]):

        test_size = 4*test_size

        nugget_save_dir = f"{save_dir}/_data_multihop"
        os.makedirs(nugget_save_dir, exist_ok=True)

        # Initialize Metadata
        self.rr_id_to_questions = {}  # rr_id -> [questions]
        self.rr_id_to_question_metadata = {}  # rr_id -> question -> metadata

        if ids_to_run:
            self.rr_ids = set(ids_to_run)
        else:
            if not self.rr_ids:
                with open(f"{save_dir}/topic_rr_us_map.pkl", "rb") as f:
                    topic_triples = pickle.load(f)
                self.rr_ids = set(rr_id for rr_id, _, _, _ in topic_triples)

        # Initialize Tracking
        all_questions = []
        all_answers = []
        all_output_contexts = []
        total_count = 0

        # Generate PER REQUEST Externally **Answerable** Seed Questions
        for current_rr_id in self.rr_ids:
            
            # If on rr_id that was completed
            if current_rr_id in ids_to_skip:
                continue

            log(colored(f"\nProcessing rr_id {current_rr_id}...", "cyan"), self.logging_file)

            current_rr_id_count = 0

            # Get Nodes for Current Topic
            current_nodes = {
                node_id: node for node_id, node in self.nodes_dict.items()
                if getattr(node, 'metadata')['rr_id']==current_rr_id
            }
            log(colored(f"Loaded {len(current_nodes)} **external** nodes (document chunks) for rr_id {current_rr_id}", "yellow"), self.logging_file)

            current_internal_nodes = {
                node_id: node for node_id, node in self.internal_nodes_dict.items()
                if getattr(node, 'metadata')['rr_id']==current_rr_id
            }
            log(colored(f"Loaded {len(current_internal_nodes)} **internal** nodes (document chunks) for rr_id {current_rr_id}", "yellow"), self.logging_file)

            # Select Documents for Multihop Generation
            grouped_nodes = self.select_documents(current_nodes, current_internal_nodes, max_node_dicts, max_per_split)

            # Generate Questions
            kwargs = {
                "current_rr_id": current_rr_id, 
                "report_requests_path": report_requests_path,
                "nugget_save_dir": nugget_save_dir, 
                "save_dir": save_dir, 
                "test_size": test_size, 
                "max_unanswerable_qs_per_request": max_unanswerable_qs_per_request, 
                "max_q_prompts_per_request": max_q_prompts_per_request, 
            }
            all_questions, all_answers, all_output_contexts, total_count, current_rr_id_count = await self.generate_questions(
                grouped_nodes=grouped_nodes, 
                total_count=total_count,
                current_rr_id_count=current_rr_id_count,     
                all_questions=all_questions,
                all_answers=all_answers,
                all_output_contexts=all_output_contexts,
                mode="documents",
                **kwargs
            )
            # all_questions, all_answers, all_output_contexts, total_count, current_rr_id_count = await self.generate_questions(
            #     grouped_nodes=grouped_nodes, 
            #     total_count=total_count,
            #     current_rr_id_count=current_rr_id_count,     
            #     all_questions=all_questions,
            #     all_answers=all_answers,
            #     all_output_contexts=all_output_contexts,
            #     mode="claims",
            #     **kwargs
            # )
            # all_questions, all_answers, all_output_contexts, total_count, current_rr_id_count = await self.generate_questions(
            #     grouped_nodes=grouped_nodes, 
            #     total_count=total_count,
            #     current_rr_id_count=current_rr_id_count,     
            #     all_questions=all_questions,
            #     all_answers=all_answers,
            #     all_output_contexts=all_output_contexts,
            #     mode="triples",
            #     **kwargs
            # )
            # all_questions, all_answers, all_output_contexts, total_count, current_rr_id_count = await self.generate_questions(
            #     grouped_nodes=grouped_nodes, 
            #     total_count=total_count,
            #     current_rr_id_count=current_rr_id_count,     
            #     all_questions=all_questions,
            #     all_answers=all_answers,
            #     all_output_contexts=all_output_contexts,
            #     mode="complex_triples",
            #     **kwargs
            # )

            # If Target # Unanswerable Questions Achieved, Break
            if total_count >= test_size:
                break

            log(colored(f"Completed processing rr_id {current_rr_id}!\nTotal questions generated: {total_count} / {test_size}\nMetadata and NuggetBank saved to: {nugget_save_dir}", "cyan"), self.logging_file)

        log(colored(f"\nCreated {len(all_questions)} total questions!\nMetadata saved to: {save_dir}\nNuggetBanks saved to: {nugget_save_dir}", "cyan"), self.logging_file)
        return all_questions, all_answers, all_output_contexts
    
    ### HELPER: Create Multihop Questions for Given Node (Chunk) Combinations
    async def generate_questions(self, 
                                 grouped_nodes: list, 
                                 current_rr_id, 
                                 report_requests_path,
                                 nugget_save_dir, 
                                 save_dir, 
                                 test_size, 
                                 max_unanswerable_qs_per_request, 
                                 max_q_prompts_per_request, 
                                 total_count, 
                                 current_rr_id_count,
                                 all_questions,
                                 all_answers,
                                 all_output_contexts,
                                 mode="documents"):

        # Check Run Conditions
        if total_count >= test_size:
            return all_questions, all_answers, all_output_contexts, total_count, current_rr_id_count
        if max_unanswerable_qs_per_request and current_rr_id_count >= max_unanswerable_qs_per_request:
            return all_questions, all_answers, all_output_contexts, total_count, current_rr_id_count
        
        if os.path.exists(f"{nugget_save_dir}/{current_rr_id}__multihop_backup_no_quality_scoring__mode_{mode}.pkl"):

            with open(f"{nugget_save_dir}/{current_rr_id}__multihop_backup_no_quality_scoring__mode_{mode}.pkl", "rb") as f:
                data = pickle.load(f)
            extracted_qa_pairs = data["qa_pairs"]
            expanded_contexts = data["contexts"]
            expanded_internal_contexts = data["internal_contexts"]
            expanded_metadata = data["metadata"]
            with open(f"{save_dir}/multihop_rr_id_to_questions.json", 'r') as f:
                self.rr_id_to_questions = json.load(f)
            with open(f"{save_dir}/multihop_rr_id_to_question_metadata.json", "rb") as f:
                self.rr_id_to_question_metadata = json.load(f)
            
        else: 

            # Prepare Question Generation Prompts
            log(colored(f"Preparing question generation prompts...", "yellow"), self.logging_file)

            prompts = []
            contexts = []
            internal_contexts = []
            prompt_metadata = []
            for node_dict in grouped_nodes: 

                # Prepare Input Context
                if mode=="documents":
                    input_context = self.prepare_context(node_dict)
                    internal_context = self.prepare_context(node_dict, internal_docs_only=True)
                elif mode=="claims":
                    input_context = self.prepare_claims(node_dict)
                    internal_context = self.prepare_claims(node_dict, internal_docs_only=True)
                elif mode=="triples": 
                    input_context = self.prepare_triples(node_dict)
                    internal_context = self.prepare_triples(node_dict, internal_docs_only=True)
                elif mode=="complex_triples":
                    input_context = self.prepare_complex_triples(node_dict)
                    internal_context = self.prepare_complex_triples(node_dict, internal_docs_only=True)
                else: 
                    raise ValueError("Invalid input format specified for question generation!")
                
                # Prepare Question Generation Prompts
                if mode=="documents":
                    template_name = random.choice(DOC_PROMPTS)
                    if template_name=="d6": 
                        all_kwargs = [
                            {'question_type': qt, 'question_format': qf, 'answer_length': al}
                            for qt, qf, al in product(question_types, question_format, answer_lengths)
                        ]
                        prompts.extend([
                            PROMPT_REGISTRY[template_name].format(context=input_context, **kwargs)
                            for kwargs in all_kwargs
                        ])
                        contexts.extend([
                            input_context for _ in all_kwargs
                        ])
                        internal_contexts.extend([
                            internal_context for _ in all_kwargs
                        ])
                        prompt_metadata.extend([
                            {
                                'rr_id': current_rr_id,
                                "generation_template": f"{template_name}",
                                "generation_kwargs": kwarg,
                                'sources': [getattr(node, 'metadata', {}).get('source', '') for node in node_dict],
                                'doc_ids': [getattr(node, 'metadata', {}).get('doc_id', '') for node in node_dict],
                            }
                            for kwarg in all_kwargs
                        ])
                    else: 
                        try:
                            prompts.append(PROMPT_REGISTRY[template_name].format(context=input_context))
                        except: 
                            import ipdb; ipdb.set_trace()
                        contexts.append(input_context)
                        internal_contexts.append(internal_context)
                        prompt_metadata.append(
                            {
                                'rr_id': current_rr_id,
                                "generation_template": f"{template_name}",
                                "generation_kwargs": {},
                                'sources': [getattr(node, 'metadata', {}).get('source', '') for node in node_dict],
                                'doc_ids': [getattr(node, 'metadata', {}).get('doc_id', '') for node in node_dict],
                            }
                        )
                else: 
                    template_name = random.choice(CLAIM_PROMPTS)
                    if template_name=="c1": 
                        all_kwargs = [
                            {'question_type': qt, 'question_format': qf, 'answer_length': al}
                            for qt, qf, al in product(question_types, question_format, answer_lengths)
                        ]
                        prompts.extend([
                            PROMPT_REGISTRY[template_name].format(claims=input_context, **kwargs)
                            for kwargs in all_kwargs
                        ])
                        contexts.extend([
                            input_context for _ in all_kwargs
                        ])
                        internal_contexts.extend([
                            internal_context for _ in all_kwargs
                        ])
                        prompt_metadata.extend([
                            {
                                'rr_id': current_rr_id,
                                "generation_template": f"{template_name}",
                                "generation_kwargs": kwarg,
                                'sources': [getattr(node, 'metadata', {}).get('source', '') for node in node_dict],
                                'doc_ids': [getattr(node, 'metadata', {}).get('doc_id', '') for node in node_dict],
                            }
                            for kwarg in all_kwargs
                        ])
                    elif template_name=="c4": 
                        kwargs = {
                            'question_type': random.choice(question_types),
                        }
                        prompts.append(PROMPT_REGISTRY[template_name].format(claims=input_context, **kwargs))
                        contexts.append(input_context)
                        internal_contexts.append(internal_context)
                        prompt_metadata.append(
                            {
                                'rr_id': current_rr_id,
                                "generation_template": f"{template_name}",
                                "generation_kwargs": {},
                                'sources': [getattr(node, 'metadata', {}).get('source', '') for node in node_dict],
                                'doc_ids': [getattr(node, 'metadata', {}).get('doc_id', '') for node in node_dict],
                            }
                        )
                    else:
                        prompts.append(PROMPT_REGISTRY[template_name].format(claims=input_context))
                        contexts.append(input_context)
                        internal_contexts.append(internal_context)
                        prompt_metadata.append(
                            {
                                'rr_id': current_rr_id,
                                "generation_template": f"{template_name}",
                                "generation_kwargs": {},
                                'sources': [getattr(node, 'metadata', {}).get('source', '') for node in node_dict],
                                'doc_ids': [getattr(node, 'metadata', {}).get('doc_id', '') for node in node_dict],
                            }
                        )
            orig_prompt_ct = len(prompts)

            # Truncate as Needed
            if max_q_prompts_per_request and len(prompts) > max_q_prompts_per_request: 
                random.seed(42)
                indices = random.sample(range(len(prompts)), max_q_prompts_per_request)
                prompts = [prompts[i] for i in indices]
                contexts = [contexts[i] for i in indices]
                prompt_metadata = [prompt_metadata[i] for i in indices]
            log(colored(f"Generating multihop questions (num prompts: {len(prompts)} / {orig_prompt_ct})...", "yellow"), self.logging_file)
                
            # Generate & Process Q/A's
            log(colored(f"Generating questions...", "yellow"), self.logging_file)
            outputs = asyncio.run(
                get_response_batch_chunked(
                    system_prompt=sys_prompt,
                    user_prompts=prompts,
                    provider=ModelProvider.LITELLM_LOCAL, 
                    temperature=0., 
                    use_cache=False,
                    max_tokens=5000,
                )
            )
            qa_pairs = parse_qa_outputs(outputs) 
            assert(len(qa_pairs)==len(prompts))

            log(colored(f"Parsing & saving questions & metadata...", "yellow"), self.logging_file)
            extracted_qa_pairs = []
            expanded_contexts = []
            expanded_internal_contexts = []
            expanded_metadata = []
            for context, internal_context, qa_list, metadata, output in zip(contexts, internal_contexts, qa_pairs, prompt_metadata, outputs):
                for qa_pair in qa_list: # tuple
                    extracted_qa_pairs.append(qa_pair)
                    expanded_contexts.append(context)
                    expanded_internal_contexts.append(internal_context)
                    expanded_metadata.append(metadata)
            with open(f"{nugget_save_dir}/{current_rr_id}__multihop_raw_prompts__mode_{mode}.txt", "w") as f:
                f.write(str(prompts))
            with open(f"{nugget_save_dir}/{current_rr_id}__multihop_raw_qa_pairs__mode_{mode}.txt", "w") as f:
                f.write(str(extracted_qa_pairs))

            # Save PER REQUEST Mapping Information
            for metadata in expanded_metadata:
                if metadata['rr_id'] not in self.rr_id_to_questions:
                    self.rr_id_to_questions[metadata['rr_id']] = []
                    self.rr_id_to_question_metadata[metadata['rr_id']] = {}
            for (q, _), metadata in zip(extracted_qa_pairs, expanded_metadata):
                self.rr_id_to_questions[metadata['rr_id']].append(q)

            # Save Metadata for Current rr_id
            with open(f"{nugget_save_dir}/{current_rr_id}__multihop_questions_by_topic.json", 'w') as f:
                json.dump(self.rr_id_to_questions[current_rr_id], f, indent=4)

            # Save Raw Results to File
            with open(f"{nugget_save_dir}/{current_rr_id}__multihop_backup_no_quality_scoring__mode_{mode}.pkl", "wb") as f:
                pickle.dump({
                    "qa_pairs": extracted_qa_pairs,
                    "contexts": expanded_contexts,
                    "internal_contexts": expanded_internal_contexts,
                    "metadata": expanded_metadata,
                }, f)
            with open(f"{save_dir}/multihop_rr_id_to_questions.json", 'w') as f:
                json.dump(self.rr_id_to_questions, f, indent=4)
            
            with open(f"{save_dir}/multihop_rr_id_to_question_metadata.json", 'w') as f:
                json.dump(self.rr_id_to_question_metadata, f, indent=4)

        
        if os.path.exists(f"{nugget_save_dir}/{current_rr_id}__multihop_backup_no_cot_scoring__mode_{mode}.pkl"):
    
            with open(f"{nugget_save_dir}/{current_rr_id}__multihop_backup_no_cot_scoring__mode_{mode}.pkl", "rb") as f:
                data = pickle.load(f)
            extracted_qa_pairs = data["qa_pairs"]
            expanded_contexts = data["contexts"]
            expanded_internal_contexts = data["internal_contexts"]
            expanded_metadata = data["metadata"]
            scores = data['scores']
            per_qa_scores = data['scores']
            cs = scores['context_sufficiency']
            cn = scores['context_necessity']
            csi = scores['context_sufficiency_internal']
            cni = scores['context_necessity_internal']
            ac = scores['answer_correctness']
            au = scores['answer_uniqueness']
            with open(f"{save_dir}/multihop_rr_id_to_questions.json", 'r') as f:
                self.rr_id_to_questions = json.load(f)
            with open(f"{save_dir}/multihop_rr_id_to_question_metadata.json", "rb") as f:
                self.rr_id_to_question_metadata = json.load(f)

        else: 
            
            scores = {}
            
            # Score Data Quality (Gold & Internal Contexts)
            if os.path.exists(f"{nugget_save_dir}/{current_rr_id}__raw_scores__mode_{mode}.pkl"):
                with open(f"{nugget_save_dir}/{current_rr_id}__raw_scores__mode_{mode}.pkl", "rb") as f:
                    scores = pickle.load(f)

            log(colored(f"Scoring question & answer quality...", "yellow"), self.logging_file)
            for criterion_name in CRITERION_NAMES:
                if criterion_name.lower().replace(" ", "_") in scores:
                    log(colored(f"Skipping already-done criterion: {criterion_name}...", "cyan"), self.logging_file)
                    continue

                log(colored(f"On criterion: {criterion_name}...", "cyan"), self.logging_file)

                # Create Prompts
                judgment_prompts = []
                for context, (q, a) in zip(expanded_contexts, extracted_qa_pairs):
                    judgment_prompts.append(
                        get_template[criterion_name].format(
                            context=context, 
                            question=q,
                            answer=a, 
                            criterion=criteria[criterion_name],
                            rubric=rubrics[criterion_name],
                        )
                    )
                # Run Inference
                judgment_outputs = asyncio.run(
                    get_response_batch_chunked(
                        system_prompt=sys_prompt,
                        user_prompts=judgment_prompts,
                        provider=ModelProvider.LITELLM_LOCAL, 
                        temperature=0., 
                        use_cache=False,
                        max_tokens=5000,
                    )
                )
                # Parse Outputs
                explanations_and_scores = [
                    parse_qa_quality(output)
                    for output in judgment_outputs
                ]
                scores[criterion_name.lower().replace(" ", "_")] = explanations_and_scores
                with open(f"{nugget_save_dir}/{current_rr_id}__raw_scores__mode_{mode}.pkl", 'wb') as file:
                    pickle.dump(scores, file)
            
                # Score Internal Context Necessity / Sufficiency
                if "Context" in criterion_name:

                    if criterion_name.lower().replace(" ", "_")+"_internal" in scores:
                        log(colored(f"Skipping already-done criterion: {criterion_name}_internal...", "cyan"), self.logging_file)
                        continue

                    judgment_prompts = []
                    for internal_context, (q, a) in zip(expanded_internal_contexts, extracted_qa_pairs):
                        judgment_prompts.append(
                            get_template[criterion_name].format(
                                context=internal_context, 
                                question=q,
                                answer=a, 
                                criterion=criteria[criterion_name],
                                rubric=rubrics[criterion_name],
                            )
                        )
                    # Run Inference
                    judgment_outputs = asyncio.run(
                        get_response_batch_chunked(
                            system_prompt=sys_prompt,
                            user_prompts=judgment_prompts,
                            provider=ModelProvider.LITELLM_LOCAL, 
                            temperature=0., 
                            use_cache=False,
                            max_tokens=5000,
                        )
                    )
                    # Parse Outputs
                    explanations_and_scores = [
                        parse_qa_quality(output)
                        for output in judgment_outputs
                    ]
                    scores[criterion_name.lower().replace(" ", "_")+"_internal"] = explanations_and_scores
                    with open(f"{nugget_save_dir}/{current_rr_id}__raw_scores__mode_{mode}.pkl", 'wb') as file:
                        pickle.dump(scores, file)
                    
            cs = scores['context_sufficiency']
            cn = scores['context_necessity']
            csi = scores['context_sufficiency_internal']
            cni = scores['context_necessity_internal']
            ac = scores['answer_correctness']
            au = scores['answer_uniqueness']
            scores = {
                "context_sufficiency": cs,
                "context_necessity": cn,
                "answer_correctness": ac,
                "answer_uniqueness": au,
                "context_sufficiency_internal": csi,
                "context_necessity_internal": cni,
            } 
            per_qa_scores = [
                {
                    "context_sufficiency": a,
                    "context_necessity": b,
                    "answer_correctness": c,
                    "answer_uniqueness": d,
                    "context_sufficiency_internal": e,
                    "context_necessity_internal": f,
                }
                for a, b, c, d, e, f in zip(cs, cn, ac, au, csi, cni)
            ]    
            # Save Raw Results to File
            with open(f"{nugget_save_dir}/{current_rr_id}__multihop_backup_no_cot_scoring__mode_{mode}.pkl", "wb") as f:
                pickle.dump({
                    "qa_pairs": extracted_qa_pairs,
                    "contexts": expanded_contexts,
                    "internal_contexts": expanded_internal_contexts,
                    "metadata": expanded_metadata,
                    "scores": scores,
                    "per_qa_scores": per_qa_scores,
                }, f)

        if os.path.exists(f"{nugget_save_dir}/{current_rr_id}__multihop_backup_shuffled__mode_{mode}.pkl"):
        
            with open(f"{nugget_save_dir}/{current_rr_id}__multihop_backup_shuffled__mode_{mode}.pkl", "rb") as f:
                data = pickle.load(f)
            extracted_qa_pairs = data["qa_pairs"]
            expanded_contexts = data["contexts"]
            expanded_internal_contexts = data["internal_contexts"]
            expanded_metadata = data["metadata"]
            scores = data['scores']
            per_qa_scores = data['per_qa_scores']
            cot_results = data['cot_annotations']
            with open(f"{save_dir}/multihop_rr_id_to_questions.json", 'r') as f:
                self.rr_id_to_questions = json.load(f)
            with open(f"{save_dir}/multihop_rr_id_to_question_metadata.json", "rb") as f:
                self.rr_id_to_question_metadata = json.load(f)
        
        else:
            # Get CoT Rationales
            log(colored(f"Annotating questions with CoT rationales...", "yellow"), self.logging_file)
            cot_prompts = [
                cot_rationale.format(context=context, question=q, answer=a)
                for context, (q, a) in zip(expanded_contexts, extracted_qa_pairs)
            ]
            cot_outputs = asyncio.run(
                get_response_batch_chunked(
                    system_prompt=sys_prompt,
                    user_prompts=cot_prompts,
                    provider=ModelProvider.LITELLM_LOCAL, 
                    temperature=0., 
                    use_cache=False,
                    max_tokens=5000,
                )
            )
            cot_results = [
                parse_cot_annotation(output) for output in cot_outputs
            ]
            with open(f"{nugget_save_dir}/{current_rr_id}__raw_cot_annotations__mode_{mode}.pkl", 'wb') as file:
                pickle.dump(cot_results, file)
            # assert(len(cot_results)==len(cot_prompts))
            cot_erroneous_citations = [
                check_cot_citation_validity(rationale=result['rationale'], context_articles=context)
                for result, context in zip(cot_results, expanded_contexts)
            ]
            for result, erroenous_citations in zip(cot_results, cot_erroneous_citations):
                result["erroneous_citations"] = erroenous_citations
            with open(f"{nugget_save_dir}/{current_rr_id}__raw_cot_annotations__mode_{mode}.pkl", 'wb') as file:
                pickle.dump(cot_results, file)
            # assert(len(cot_results)==len(cs))
            # assert(len(cot_results)==len(csi))
            # assert(len(cot_results)==len(cn))
            # assert(len(cot_results)==len(cni))
            # assert(len(cot_results)==len(ac))
            # assert(len(cot_results)==len(au))
            # assert(len(cot_results)==len(extracted_qa_pairs))
            # assert(len(cot_results)==len(expanded_contexts))
            # assert(len(cot_results)==len(expanded_internal_contexts))
            # assert(len(cot_results)==len(expanded_metadata))

            # SHUFFLE BEFORE PROCESSING TO ENSURE DIVERSITY
            combined = list(zip(extracted_qa_pairs, expanded_contexts, expanded_internal_contexts, expanded_metadata, cs, cn, csi, cni, ac, au, cot_results))
            random.shuffle(combined)
            extracted_qa_pairs, expanded_contexts, expanded_internal_contexts, expanded_metadata, cs, cn, csi, cni, ac, au, cot_results = zip(*combined)

            extracted_qa_pairs, expanded_contexts, expanded_internal_contexts, expanded_metadata, cs, cn, csi, cni, ac, au, cot_results = list(extracted_qa_pairs), list(expanded_contexts), list(expanded_internal_contexts), list(expanded_metadata), list(cs), list(cn), list(csi), list(cni), list(ac), list(au), list(cot_results)
            scores = {
                "context_sufficiency": cs,
                "context_necessity": cn,
                "answer_correctness": ac,
                "answer_uniqueness": au,
                "context_sufficiency_internal": csi,
                "context_necessity_internal": cni,
            } 
            per_qa_scores = [
                {
                    "context_sufficiency": a,
                    "context_necessity": b,
                    "answer_correctness": c,
                    "answer_uniqueness": d,
                    "context_sufficiency_internal": e,
                    "context_necessity_internal": f,
                }
                for a, b, c, d, e, f in zip(cs, cn, ac, au, csi, cni)
            ]           
            # Save Raw Shuffled Results to File
            with open(f"{nugget_save_dir}/{current_rr_id}__multihop_backup_shuffled__mode_{mode}.pkl", "wb") as f:
                pickle.dump({
                    "qa_pairs": extracted_qa_pairs,
                    "contexts": expanded_contexts,
                    "internal_contexts": expanded_internal_contexts,
                    "metadata": expanded_metadata,
                    "cot_annotations": cot_results,
                    "scores": scores,
                    "per_qa_scores": per_qa_scores,
                }, f)

        # Verify RAG Unanswerability
        log(colored(f"Verifying questions are NOT answerable with given corpus for rr_id {current_rr_id}...", "yellow"), self.logging_file)
        index = 0
        for (q, a), context, internal_context, metadata, cot_annotation, scores in zip(extracted_qa_pairs, expanded_contexts, expanded_internal_contexts, expanded_metadata, cot_results, per_qa_scores):

            index += 1
            print(colored(f"On index {index}", "cyan"))

            log(colored("\n#### Question-Answer Pair", "magenta"), self.logging_file)
            log(colored(f"\nQuestion: {q}", "magenta"), self.logging_file)
            log(f"\nExternal Answer: {a}", self.logging_file)

            question_metadata = {
                'context': context,
                'internal_context': internal_context,
                'generation_template': metadata['generation_template'],
                'generation_kwargs': metadata['generation_kwargs'],
                'sources': metadata.get('sources', ''),
                'doc_ids': metadata.get('doc_ids', ''),
                'external_answer': a,
                'cot_annotation': cot_annotation,
                'scores': scores,
                'rag_answer': "",
                'answerability': None,
                'reason': None,
            }

            # Confirm Question NOT Anwerable with Database
            rag_response = self.rag.rag_query_engine.query(q)
            question_metadata['rag_answer'] = str(rag_response)
            log(f"\nRAG Answer: {rag_response}", self.logging_file)
            
            reason, rag_verdict = await self.verify_unanswerable_query(
                question=q,
                answer=rag_response,
                ground_truth=a,
            )
            question_metadata['reason'] = reason
            log(f"\nReason: {reason}", self.logging_file)
            log(f"\nRAG Verdict: {rag_verdict}", self.logging_file)

            if rag_verdict=="judgment error":     
                log(colored("\nAnswerability: Murky (externally answerable but RAG judgment error)...", "magenta"), self.logging_file)
                question_metadata['answerability'] = 'murky (externally answerable but rag answerability judgment error)'
            
            elif rag_verdict=="1":
                question_metadata['answerability'] = 'answerable'
                log(colored("\nAnswerability: Answerable", "magenta"), self.logging_file)

            elif rag_verdict=="0":
                question_metadata['answerability'] = 'murky (externally answerable but rag answer plausible too -- question not underspecified but multiple plausible answers)'
                log(colored("\nAnswerability: Murky (externally answerable but rag answer plausible too -- question not underspecified but multiple plausible answers)...", "magenta"), self.logging_file)

            else:
                question_metadata['answerability'] = 'unanswerable'
                log(colored("\nAnswerability: UNanswerable!!!", "magenta"), self.logging_file)
                all_questions.append(q)
                all_answers.append(a)
                all_output_contexts.append(context)
                total_count += 1
                current_rr_id_count += 1

                with open(f"{save_dir}/multihop_total_count.txt", "w") as f:
                    f.write(str(total_count))

                # Save ALL Raw Questions/Answers/Contexts So Far
                with open(f"{save_dir}/multihop_raw_questions.txt", "w") as f:
                    f.write(str(all_questions))
                with open(f"{save_dir}/multihop_raw_answers.txt", "w") as f:
                    f.write(str(all_answers))
                with open(f"{save_dir}/multihop_raw_output_contexts.txt", "w") as f:
                    f.write(str(all_output_contexts))

                if total_count % 5 == 0:
                    log(colored(f"Logging: Generated {total_count} total questions so far!", "yellow"), self.logging_file)
        
            # Store Question Metadata
            self.rr_id_to_question_metadata[current_rr_id][q] = question_metadata

            # Save Metadata for Current rr_id
            with open(f"{nugget_save_dir}/{current_rr_id}__multihop_questions_metadata.json", 'w') as f:
                json.dump(self.rr_id_to_question_metadata[current_rr_id], f, indent=4)

            # Save Metadata for ALL rr_id's
            with open(f"{save_dir}/multihop_rr_id_to_questions.json", 'w') as f:
                json.dump(self.rr_id_to_questions, f, indent=4)
            
            with open(f"{save_dir}/multihop_rr_id_to_question_metadata.json", 'w') as f:
                json.dump(self.rr_id_to_question_metadata, f, indent=4)

            # Save Current NuggetBank
            self.save_nuggetbank(
                save_dir=nugget_save_dir,
                rr_id=current_rr_id,
                report_requests_path=report_requests_path,
                mode=mode,
            )

            # If Target # Unanswerable Questions Achieved, Break
            if total_count >= test_size:
                print("here")
                break

            if max_unanswerable_qs_per_request and current_rr_id_count >= max_unanswerable_qs_per_request:
                print("here2")
                break
        
        return all_questions, all_answers, all_output_contexts, total_count, current_rr_id_count
