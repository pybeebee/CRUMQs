import json
import os 

import nest_asyncio
nest_asyncio.apply()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, StorageContext, PromptTemplate, Settings
from llama_index.core import get_response_synthesizer, load_index_from_storage

from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, MultiStepQueryEngine, TransformQueryEngine

from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.packs.raptor import RaptorRetriever

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform

from llama_index.core.postprocessor import LLMRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank


class CustomRetriever(BaseRetriever):
    """
    Custom retriever to perform both semantic search and hybrid search.
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

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(raptor_ids)
        else:
            retrieve_ids = vector_ids.union(raptor_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]

        return retrieve_nodes

class RAGVerifier:
    def __init__(
        self,
        folder,
        load_index_path,
        save_index_dir,
        model_name,
        top_k=10,
        retriever_type=None,
        reranker_type=None,
        rewriting=None,
        retriever_load_path=None,
        retriever_save_path=None,
        embed_model="bge",
        # embed_model="openai",
    ):
        
        # Instantiate Document Database for RAG
        self.folder = folder
        self.documents = SimpleDirectoryReader(folder).load_data()

        # Set Load / Save Paths
        self.load_index_path = load_index_path
        self.save_index_dir = save_index_dir

        self.retriever_type = retriever_type
        self.retriever_load_path = retriever_load_path
        self.retriever_save_path = retriever_save_path

        self.rewriting = rewriting
        self.top_k = top_k

        # LLM for Embedding, Summarization, Query Decomposition
        self.model_name = model_name
        self.llm = OpenAILike(
            model=model_name,
            api_base=os.getenv("LITELLM_IP"),
            api_key=os.getenv("LITELLM_API_KEY"),
            max_tokens=250,
            temperature=0.,
            context_window=128000,
            is_chat_model=True,
            is_function_calling_model=False,
            # additional_kwargs={
            #     "top_p": top_p,
            #     "stop": stop,
            # }
        )
        self.embed_model = self.load_embed_model(embed_model)
        
        Settings.llm=self.llm
        Settings.chunk_size=1024
        Settings.chunk_overlap=24
        
        # Instantiate Parts of RAG Pipeline Using Helpers
        self.reranker = self.build_reranker(reranker_type)
        self.storage_context, self.vector_index = self.build_index()
        self.rag_query_engine, self.rag_query_engine_rewrite, self.retriever = self.build_engine()

    ### HELPER: Load Embedding Model
    def load_embed_model(self, embed_model):

        if embed_model == "bge":
            print("BGE embedding is used")
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-large-en-v1.5",
            )
            Settings.embed_model = embed_model
            return embed_model

        if embed_model == "cohere":
            print("Cohere embedding is used")
            embed_model = CohereEmbedding(
                api_key=os.environ["COHERE_API_KEY"],
                model_name="embed-english-v3.0",
                input_type="search_query",
            )
            Settings.embed_model = embed_model
            return embed_model

        print("OpenAI embedding is used")
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small"
        )
        Settings.embed_model = embed_model
        return embed_model

    # HELPER: Build Reranker
    def build_reranker(self, reranker_type=None):

        reranker = None
        if not reranker_type:
            return None

        if reranker_type == "llm":
            reranker = LLMRerank(
                choice_batch_size=5,
                top_n=5,
            )

        if reranker_type == "cohere":
            api_key = os.environ["COHERE_API_KEY"]
            reranker = CohereRerank(api_key=api_key, top_n=5)

        return reranker

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
                persist_dir=self.save_index_dir,
            )

        else:
            storage_context = StorageContext.from_defaults(
                persist_dir=self.load_index_path,
            )
            vector_index = load_index_from_storage(storage_context)

        return storage_context, vector_index

    ### HELPER: Build Retriever
    def build_retriever(self):

        similarity_top_k = self.top_k
        # if self.retriever_type:
        #     similarity_top_k = 10

        # If Using BM25 Retriever
        if self.retriever_type == "bm25":
            retriever = BM25Retriever.from_defaults(
                docstore=self.storage_context.docstore,
                similarity_top_k=similarity_top_k,
            )

        # If Using Raptor / Ensemble Retriever
        elif self.retriever_type == "raptor" or self.retriever_type == "ensemble":

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

        # If Using Default Retriever
        else:
            retriever = VectorIndexRetriever(
                index=self.vector_index, 
                similarity_top_k=similarity_top_k,
            )

        # If Using Ensembling
        if self.retriever_type == "ensemble":
            vector_retriever = VectorIndexRetriever(
                index=self.vector_index, 
                similarity_top_k=similarity_top_k,
            )
            retriever = CustomRetriever(vector_retriever, retriever)

        return retriever

    # HELPER: Build Retriever, Query Engine, Query Rewriter
    def build_engine(self):

        retriever = self.build_retriever()
        response_synthesizer = get_response_synthesizer(
            # service_context=self.service_context,
            response_mode="tree_summarize",
        )

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
        if self.rewriting and self.rewriting == "hyde":
            hyde = HyDEQueryTransform(include_original=True)
            rag_query_engine_rewrite = TransformQueryEngine(
                rag_query_engine, 
                hyde,
            )

        if self.rewriting and self.rewriting == "multi-step":
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

    def update_template(self, template_usage=None):

        if template_usage:
            new_summary_tmpl = PromptTemplate(template_usage)
            self.rag_query_engine.update_prompts(
                {"response_synthesizer:summary_template": new_summary_tmpl}
            )
