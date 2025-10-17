import json
import os
import argparse
import random
from termcolor import colored
import pickle
from tqdm import tqdm

import nest_asyncio
nest_asyncio.apply()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, PromptTemplate, Settings, get_response_synthesizer, load_index_from_storage
from llama_index.core.schema import QueryBundle

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding

from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.packs.raptor import RaptorRetriever

from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine, MultiStepQueryEngine

from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.indices.query.query_transform.base import  StepDecomposeQueryTransform

from llama_index.core.postprocessor import LLMRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank

from prompts_generation import PROMPT_REGISTRY


class CustomRetriever(BaseRetriever):
    """
    Custom retriever that performs both semantic search and hybrid search.
    """

    def __init__(self, vector_retriever, raptor_retriever, mode="OR"):

        self._vector_retriever = vector_retriever
        self._raptor_retriever = raptor_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle):
        """
        Retrieve nodes given query.
        """

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        raptor_nodes = self._raptor_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        raptor_ids = {n.node.node_id for n in raptor_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in raptor_nodes})

        if self._mode=="AND":
            retrieve_ids = vector_ids.intersection(raptor_ids)
        else:
            retrieve_ids = vector_ids.union(raptor_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]

        return retrieve_nodes

class RAGEngine:
    def __init__(
        self,
        folder,
        load_index_path,
        save_index_path,
        model_name,
        template_name,
        # top_k=10,
        embed_model="openai",
        retriever_type="vector",
        reranker_type=None,
        rewriting=None,
        retriever_load_path=None,
        retriever_save_path=None,
    ):
        
        
        # Instantiate Document Database for RAG
        self.folder = folder
        self.documents = SimpleDirectoryReader(folder).load_data()

        # Set Load / Save Paths
        self.load_index_path = load_index_path
        self.save_index_path = save_index_path

        # Embedding Model
        print(colored("Creating embedding model...", "yellow"))
        self.embed_model = self.load_embed_model(embed_model)

        # Retriever Setup
        self.retriever_type = retriever_type
        self.retriever_load_path = retriever_load_path
        self.retriever_save_path = retriever_save_path
        # self.top_k = top_k

        # Generator Model
        print(colored("Building generator model...", "yellow"))
        self.model_name = model_name
        self.llm = OpenAI(
            temperature=0.0000000000001, 
            model=model_name, # MUST BE GPT-4o
            max_tokens=500,
            max_retries=0,
            reasoning_effort="low",
            verbosity="low",
        )

        print(colored("Building reranker...", "yellow"))
        self.rewriting = rewriting
        self.reranker = self.build_reranker(reranker_type)

        Settings.llm=self.llm
        Settings.chunk_size=1024
        Settings.chunk_overlap=24

        print(colored("Building index...", "yellow"))
        self.storage_context, self.vector_index = self.build_index()

        print(colored("Building RAG engine...", "yellow"))
        self.rag_query_engine, self.rag_query_engine_rewrite, self.retriever = self.build_engine()

        self.update_template(template_name)

    ### HELPER: Load Embedding Model
    def load_embed_model(self, embed_model):

        if embed_model=="bge":
            print(colored("Using BGE embedding...", "cyan"))
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-large-en-v1.5",
            )
            Settings.embed_model = embed_model

        elif embed_model=="cohere":
            print(colored("Using Cohere embedding...", "cyan"))
            embed_model = CohereEmbedding(
                api_key=os.environ["COHERE_API_KEY"],
                model_name="embed-english-v3.0",
                input_type="search_query",
            )
            Settings.embed_model = embed_model

        elif embed_model=="openai":
            print(colored("Using OpenAI embedding...", "cyan"))
            embed_model = OpenAIEmbedding(
                api_key=os.environ["OPENAI_API_KEY"],
                model="text-embedding-3-small"
            )
            Settings.embed_model = embed_model

        else: 
            raise ValueError("Invalid embedding model specified.")
        
        return embed_model

    # HELPER: Build Reranker
    def build_reranker(self, reranker_type=None):

        # If reranker type specified
        if reranker_type:

            if reranker_type=="gpt-4o":
                return LLMRerank(
                    choice_batch_size=5,
                    top_n=5,
                )

            if reranker_type=="cohere":
                api_key = os.environ["COHERE_API_KEY"]
                return CohereRerank(
                    api_key=api_key,
                    top_n=5,
                )

        else: 
            return None

    # HELPER: Build Document Index
    def build_index(self):
       
        if not self.load_index_path:

            parser = SentenceSplitter()
            nodes = parser.get_nodes_from_documents(self.documents)

            # Create Storage Context Using Default Stores
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                vector_store=SimpleVectorStore(),
                index_store=SimpleIndexStore(),
            )

            # Create Docstore & Add Nodes
            storage_context.docstore.add_documents(nodes)

            # Create Index
            vector_index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
            vector_index.storage_context.persist(
                persist_dir=self.save_index_path,
            )

        else:
            storage_context = StorageContext.from_defaults(
                persist_dir=self.load_index_path
            )
            vector_index = load_index_from_storage(storage_context)

        return storage_context, vector_index

    ### HELPER: Build Retriever
    def build_retriever(self):

        similarity_top_k = 5
        # if self.retriever_type:
        #     similarity_top_k = 10

        # If Using BM25 Retriever
        if self.retriever_type=="bm25":
            retriever = BM25Retriever.from_defaults(
                docstore=self.storage_context.docstore,
                similarity_top_k=similarity_top_k,
            )

        # If Using Raptor / Ensemble Retriever
        elif self.retriever_type=="raptor" or self.retriever_type=="ensemble":

            if self.retriever_load_path:
                retriever = RaptorRetriever.from_persist_dir(
                    persist_dir=self.retriever_load_path,
                )

            else:
                retriever = RaptorRetriever(
                    self.documents,
                    embed_model=self.embed_model,
                    tree_depth=2,
                    llm=self.llm,           # used for generating summaries
                    similarity_top_k=2,     # top k for each layer, or overall top-k for collapsed
                    mode="tree_traversal",  # sets default mode
                )
                retriever.persist(self.retriever_save_path)

        # If Using Default (Vector Index) Retriever
        elif self.retriever_type=="vector":

            retriever = VectorIndexRetriever(
                index=self.vector_index, similarity_top_k=similarity_top_k
            )

        # If Using Ensembling
        if self.retriever_type=="ensemble":
            vector_retriever = VectorIndexRetriever(
                index=self.vector_index, similarity_top_k=similarity_top_k
            )
            retriever = CustomRetriever(vector_retriever, retriever)

        return retriever

    # HELPER: Build Retriever, Query Engine, Query Rewriter
    def build_engine(self):

        print(colored("Building retriever...", "cyan"))
        retriever = self.build_retriever()
        
        print(colored("Building response synthesizer...", "cyan"))
        response_synthesizer = get_response_synthesizer(
            # service_context=self.service_context,
            response_mode="tree_summarize",
        )

        print(colored("Building query engine...", "cyan"))
        if self.reranker:
            rag_query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[self.reranker],
            )
        else:
            rag_query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
            )

        rag_query_engine_rewrite = None
        if self.rewriting and self.rewriting=="hyde":
            print(colored("Building rewriter...", "cyan"))
            hyde = HyDEQueryTransform(include_original=True)
            rag_query_engine_rewrite = TransformQueryEngine(
                rag_query_engine, 
                hyde,
            )

        if self.rewriting and self.rewriting=="multistep":
            print(colored("Building rewriter...", "cyan"))
            step_decompose_transform = StepDecomposeQueryTransform(
                llm=self.llm, 
                verbose=True,
            )
            index_summary = "used for answer questions"
            rag_query_engine_rewrite = MultiStepQueryEngine(
                query_engine=rag_query_engine,
                query_transform=step_decompose_transform,
                index_summary=index_summary,
            )

        return rag_query_engine, rag_query_engine_rewrite, retriever

    ### HELPER: Define Prompt Template for RAG Engine
    def update_template(self, template_name="DEFAULT"):
        new_summary_tmpl = PromptTemplate(PROMPT_REGISTRY[template_name])
        self.rag_query_engine.update_prompts(
            {"response_synthesizer:summary_template": new_summary_tmpl}
        )

    ### HELPER: Generate Answers on Given Dataset
    def get_answer_predictions(self, data_dict, num_samples, checkpoint_file, num_done=None):
    
        loaded_answers = False
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "rb") as f:
                answers = pickle.load(f)
            loaded_answers = True
            if num_done:
                answers = answers[:num_done]
        else:
            answers = []

        questions = data_dict['question']
        random.shuffle(questions)
        for i, q in enumerate(tqdm(questions[:num_samples])):

            if loaded_answers and i < len(answers): 
                continue

            try:
                # Get Prediction
                answer = self.rag_query_engine_rewrite.query(q)

                # Try Again if Empty
                if answer.strip()=="Empty Response":
                    answer = self.rag_query_engine.query(q)

            except Exception as e:

                if "quota" in str(e).lower():
                    raise ValueError("Quota error")

                # Try Again if Failure
                try:
                    answer = self.rag_query_engine.query(q)
                except:
                    answer = "No answer."

            answers.append(str(answer).strip())

            with open(checkpoint_file, "wb") as f:
                pickle.dump(answers, f)

        return questions[:num_samples], answers

    ### HELPER: Get Retrieval Results & LLM Predictions on Given Dataset
    def get_retrieval_results(self, data_dict, num_samples, checkpoint_file, load_retrieval_results_file=None):
    
        loaded_retrieval_results = False
        if load_retrieval_results_file:
            with open(load_retrieval_results_file, "rb") as f:
                retrieved_articles_per_q = pickle.load(f)
            loaded_retrieval_results = True
        elif os.path.exists(checkpoint_file):
            with open(checkpoint_file, "rb") as f:
                retrieved_articles_per_q = pickle.load(f)
            loaded_retrieval_results = True
        else:
            retrieved_articles_per_q = []

        questions = data_dict['question']
        random.shuffle(questions)
        for i, q in enumerate(tqdm(questions[:num_samples])):

            if loaded_retrieval_results and i < len(retrieved_articles_per_q): 
                continue

            # Retrieve Nodes for Current Question
            try:
                # retrieve_nodes = self.retriever._retrieve(QueryBundle(q))
                retrieve_nodes = self.retriever.retrieve(q)
            except:
                import ipdb; ipdb.set_trace()
            
            nodes = [node.text for node in retrieve_nodes]
            retrieved_articles_per_q.append(nodes)

            with open(checkpoint_file, "wb") as f:
                pickle.dump(retrieved_articles_per_q, f)
                
        return questions[:num_samples], retrieved_articles_per_q
                
### HELPER: Save Args to File
def save_args_to_file(args, save_dir):
    
    args_dict = vars(args)
    filepath = os.path.join(save_dir, "args.json")

    with open(filepath, 'w') as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
    
    return filepath

### HELPER: Get Identifying String for Experiment
def get_exp_identifier(args):

    return f"{args.generator_llm}__emb_{args.embedding}__ret_{args.retrieval}__rerank_{args.reranker or 'None'}__rewr_{args.rewriting or 'None'}__prompt_{args.prompt_template}"

### HELPER: Get Proprietary LLM Predictions
def get_proprietary_predictions(questions, retrieved_articles_per_q, generator_llm, prompt_template, num_samples, checkpoint_file, num_done):

    if "gpt" in generator_llm: 
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    elif "gemini" in generator_llm:
        from google.generativeai import GenerationConfig
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(generator_llm)

    else: 
        raise ValueError("Invalid proprietary LLM name provided.")

    assert(len(questions)==len(retrieved_articles_per_q))
    assert(len(questions)==num_samples)

    loaded_answers = False
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            answers = pickle.load(f)
        loaded_answers = True
        if num_done:
            answers = answers[:num_done]
    else:
        answers = []

    prompts = [
        PROMPT_REGISTRY[prompt_template].format(
            context_str=context,
            query_str=q,
        )
        for q, context in zip(questions, retrieved_articles_per_q) 
    ]

    for i, prompt in enumerate(tqdm(prompts)):

        if loaded_answers and i < len(answers): 
                continue

        if "gpt" in generator_llm: 
            messages=[
                {"role": "user", "content": prompt}
            ]
            try:
                try:
                    response = client.chat.completions.create(
                        model=generator_llm,
                        messages=messages,
                        max_completion_tokens=500,
                        temperature=0,
                        # reasoning_effort="minimal",
                        # verbosity="medium",
                        # verbosity="low",
                    ).choices
                except:
                    response = client.chat.completions.create(
                        model=generator_llm,
                        messages=messages,
                        max_completion_tokens=500,
                        # reasoning_effort="minimal",
                        # verbosity="medium",
                        # verbosity="low",
                    ).choices
                answer = [x.message.content for x in response][0]
            except Exception as e: 
                print(e)
                if "quota" in str(e).lower():
                    raise ValueError("Quota error")
                answer = "Response failure"

        elif "gemini" in generator_llm:
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=GenerationConfig( 
                        # max_output_tokens=1000, 
                        candidate_count=1,
                        top_k=1,
                        temperature=0,
                    ),
                )
                answer = response.text.strip()
            except Exception as e: 
                print(e)
                answer = "Response failure"

        answers.append(answer)
        with open(checkpoint_file, "wb") as f:
            pickle.dump(answers, f)
    
    return answers 


def main(args):

    ### Create Output Path
    dataset_name = os.path.basename(args.dataset_path).replace(".json", "")
    exp_identifier = get_exp_identifier(args)
    
    save_predictions_path = os.path.join(args.results_dir, args.evaluation_mode, dataset_name, exp_identifier)
    os.makedirs(save_predictions_path, exist_ok=True)

    ### Save Config
    save_args_to_file(args, save_predictions_path)
    print(colored(f"All arguments saved to {save_predictions_path}!", "yellow"))

    ### Create Index Dir
    args.save_index_dir = os.path.join(args.database_dir, args.save_index_dir)
    os.makedirs(args.save_index_dir, exist_ok=True)

    ### Create RAG Agent
    rag_engine = RAGEngine(
        folder=args.database_dir,
        load_index_path=args.load_index_path,
        save_index_path=args.save_index_dir,
        model_name="gpt-4o",
        template_name=args.prompt_template,
        embed_model=args.embedding,
        retriever_type=args.retrieval,
        reranker_type=args.reranker,
        rewriting= args.rewriting,
        # retriever_load_path=,
        # retriever_save_path=,
    )

    ### Load Data
    # Assume format is .json with question, answer keys
    print(colored(f"Loading data...", "yellow"))
    with open(args.dataset_path, "r") as f:
        data = json.load(f)

    ### Run Results
    print(colored(f"Getting predictions...", "yellow"))
    if args.evaluation_mode=="gateway":
        retrieval_checkpoint_file = os.path.join(save_predictions_path, "retrieved_articles_per_q.pkl")
        answer_checkpoint_file = os.path.join(save_predictions_path, "answers.pkl")
        print(colored(f"Getting retrieval results...", "cyan"))
        questions, retrieved_articles_per_q = rag_engine.get_retrieval_results(
            data_dict=data,
            num_samples=args.num_samples,
            checkpoint_file=retrieval_checkpoint_file,
            load_retrieval_results_file=args.load_retrieval_results_file,
        )
        print(colored(f"Getting LLM generation results...", "cyan"))
        answers = get_proprietary_predictions(
            questions=questions,
            retrieved_articles_per_q=retrieved_articles_per_q, 
            generator_llm=args.generator_llm,
            prompt_template=args.prompt_template,
            num_samples=args.num_samples,
            checkpoint_file=answer_checkpoint_file,
            num_done=args.num_done,
        )
   
    elif args.evaluation_mode=="rag":
        checkpoint_file = os.path.join(save_predictions_path, "answers.pkl")
        questions, answers = rag_engine.get_answer_predictions(
            data_dict=data,
            num_samples=args.num_samples,
            checkpoint_file=checkpoint_file,
            num_done=args.num_done,
        )
    
    else:
        raise ValueError("Invalid experimental setting specified.")

    ### Save Predictions
    print(colored(f"Saving results...", "yellow"))
    results_path = os.path.join(save_predictions_path, "predictions.json")
    results = {
        "questions": questions,
        "answers": answers,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(colored(f"Saved predictions for dataset `{dataset_name}` to {results_path}!", "green"))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
    )
    parser.add_argument(
        "--database_dir",
        type=str,
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
        "--results_dir",
        type=str,
    )

    parser.add_argument(
        "--evaluation_mode",
        type=str,
        default="gateway",
        choices=["rag", "gateway"],
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3048,
    )
    parser.add_argument(
        "--num_done",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--load_retrieval_results_file",
        type=str,
        default=None,
    )

    # Default Args Set to Proprietary Exp Settings
    parser.add_argument(
        "--generator_llm",
        type=str,
        default="gpt-4o",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="openai",
        choices=["bge", "cohere", "openai"],
    )
    parser.add_argument(
        "--retrieval",
        type=str,
        default="ensemble",
        choices=["bm25", "raptor", "ensemble", "vector"]
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default=None,
        choices=["gpt-4o", "cohere"]
    )
    parser.add_argument(
        "--rewriting",
        type=str,
        default=None,
        choices=["hyde", "multistep"]
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="DEFAULT",
        choices=["DEFAULT", "PROMPT1", "PROMPT2"]
    )

    args = parser.parse_args()
    main(args)
