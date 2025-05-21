"""
Joint Retrieval System

This module provides a comprehensive retrieval pipeline including query classification,
optimization, document searching, reranking, and compression.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import json
import re
import torch
import time
from pathlib import Path

from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    Document,
    Settings,
    PromptHelper,
    load_index_from_storage,
    StorageContext
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
    ChatMessage
)
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, VectorContextRetriever
from llama_index.core.schema import QueryBundle, QueryType, NodeWithScore
from llama_index.core.indices.tree.base import TreeRetrieverMode
from llama_index.core.node_parser import SimpleNodeParser, SentenceWindowNodeParser

from openai import OpenAI


@dataclass
class RetrievalConfig:
    """
    Configuration class for retrieval settings.

    Attributes:
        with_retrieval_classification: Whether to use retrieval classification
        with_query_optimization: Whether to optimize the query
        with_rerank: Whether to rerank retrieved documents
        with_compress: Whether to compress retrieved documents
        query_method: Method used for query optimization
        rerank_model: Model used for reranking
        search_k: Number of documents to retrieve
        top_k: Number of documents to return after reranking
        index_method: Method used for indexing
        dataset: Dataset name
        force_rebuild: Whether to force rebuild the index
        data_volume: Volume of data to use
        Vector_Store: Vector store to use
        classification_model: Model used for classification
        search_method: Method used for searching
        compression_method: Method used for compression
        repack_method: Method used for repacking
        compression_ratio: Ratio for compression
    """
    with_retrieval_classification: bool
    with_query_optimization: bool
    with_rerank: bool
    with_compress: bool
    query_method: str
    rerank_model: str
    search_k: int
    top_k: int
    index_method: str
    dataset: str
    force_rebuild: bool
    data_volume: int
    Vector_Store: str
    classification_model: str
    search_method: str
    compression_method: str
    repack_method: str
    compression_ratio: float = 0.4


class NimClient:
    """
    Client for interacting with the Nim API.
    
    Attributes:
        client: OpenAI client instance
    """

    def __init__(self, saturn_token: str, base_url: str = "http://localhost:8000/v1"):
        """
        Initialize the NimClient.
        
        Args:
            saturn_token: Authentication token
            base_url: API base URL
        """
        self.client = OpenAI(
            base_url=base_url,
            default_headers={'Authorization': f'token {saturn_token}'},
            api_key='not-used'
        )


class ModelWrapper:
    """
    Wrapper class for model inference.
    
    Attributes:
        client: Client for API calls
        model_name: Name of the model to use
    """

    def __init__(self, client: Any, model_name: str = "meta/llama-3_1-8b-instruct"):
        """
        Initialize the ModelWrapper.
        
        Args:
            client: Client for API calls
            model_name: Name of the model to use
        """
        self.client = client
        self.model_name = model_name

    def __call__(self, text: str, max_tokens: int = 300, return_full_text: bool = False, **kwargs) -> List[Dict[str, str]]:
        """
        Call the model with the given text.
        
        Args:
            text: Input text
            max_tokens: Maximum number of tokens to generate
            return_full_text: Whether to return the full text
            **kwargs: Additional arguments
            
        Returns:
            List of dictionaries containing generated text
        """
        response = self.client.completions.create(
            model=self.model_name,
            prompt=text,
            max_tokens=max_tokens,
            temperature=kwargs.get('temperature', 0.4),
            top_p=kwargs.get('top_p', 0.9),
            stream=False
        )
        return [{'generated_text': response.choices[0].text}]


class LlamaCustomLLM(CustomLLM):
    """
    Custom LLM implementation for Llama model.
    
    Attributes:
        context_window: Size of the context window
        num_output: Maximum number of tokens to generate
        model_name: Name of the model
        client: Client for API calls
    """

    context_window: int = 12800
    num_output: int = 1024
    model_name: str = "meta/llama-3_1-8b-instruct"
    client: Any = None

    def __init__(self, client):
        """
        Initialize the LlamaCustomLLM.
        
        Args:
            client: Client for API calls
        """
        super().__init__()
        self.client = client.client

    @property
    def metadata(self) -> LLMMetadata:
        """
        Get LLM metadata.
        
        Returns:
            LLMMetadata object
        """
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Complete the given prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments
            
        Returns:
            CompletionResponse object
        """
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=kwargs.get('max_tokens', 500),
            temperature=kwargs.get('temperature', 0.4),
            top_p=kwargs.get('top_p', 0.9),
            n=1,
            stream=False
        )

        return CompletionResponse(text=response.choices[0].text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        Stream complete the given prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments
            
        Returns:
            CompletionResponseGen generator
        """
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=kwargs.get('max_tokens', 300),
            temperature=kwargs.get('temperature', 0.4),
            top_p=kwargs.get('top_p', 0.9),
            n=1,
            stream=True
        )

        text = ""
        for chunk in response:
            if chunk.choices[0].text:
                text += chunk.choices[0].text
                yield CompletionResponse(text=text, delta=chunk.choices[0].text)


def llm_task_cls(llm, tokenizer, query: str) -> bool:
    """
    Determine if retrieval is necessary using an LLM.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        query: Query text
        
    Returns:
        True if retrieval is necessary, False otherwise
    """
    messages = [
        {"role": "system", "content": """You are a judgment assistant. Please judge whether the given question requires additional knowledge base retrieval.
    Rules:
    1. retrieval is required if the problem involves knowledge of specific facts, data, definitions, etc. that are beyond the scope of language model training
    2. if the problem is common sense or pure reasoning, no retrieval is required
    3. If you are not sure, retrieval is recommended

    Please return 'True' or 'False'. """},
        {"role": "user", "content": f"Query：{query}"}
    ]

    prompt = llm.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        analysis = llm(prompt, max_tokens=50, return_full_text=False)[0]['generated_text']
        print(f'Need retrieval? {"True" if analysis == "True" else "False"}')
        return True if analysis == 'True' else False
    except:
        # Default to retrieval if analysis fails
        return True


def llm_compress(llm, tokenizer, query: str, context: str) -> str:
    """
    Compress retrieved documents using an LLM.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        query: Query text
        context: Context to compress
        
    Returns:
        Compressed context
    """
    messages = [
        {"role": "system", "content": f"""Generate a concise and accurate summary of the following information using exact words from the context wherever possible. 
    Ensure that the summary contains all the key information, including but not limited to, time, place, name, people, relationships, and events. 
    The summary should not exceed the number of previous words.
    If the original text is sufficiently brief that important information can no longer be omitted, please retain the original text.
    Give the answer directly, do not add "here's the summary:" ."""},
        {"role": "user", "content": f"Information：{context}"}
    ]
    prompt = llm.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        analysis = llm(prompt, max_tokens=300, return_full_text=False)[0]['generated_text']
        return analysis.strip()
    except:
        return context


def llm_rerank(llm, tokenizer, mode: str, query: str, docs: List[str], top_k: Optional[int] = None) -> Tuple[
    List[str], List[float]]:
    """
    Entry point for reranking documents using various strategies.

    Args:
        llm: Language model
        tokenizer: Tokenizer
        mode: Reranking mode (e.g. "llama3")
        query: Query text
        docs: List of documents
        top_k: Number of documents to return

    Returns:
        Tuple of reranked documents and scores
    """
    if top_k is None or top_k > len(docs):
        top_k = len(docs)

    reranked_docs = []
    similarity_scores = []

    if mode == "llama3":
        # Prepare prompt with documents
        prompt_result = "\n".join([f"Document {i}: {doc}" for i, doc in enumerate(docs)])

        messages = [
            {"role": "system", "content": """You are an expert in document sorting. Please evaluate the relevance of each document to the query and return a sorted indexed list.
Rules:
1. Consider semantic relevance
2. Consider completeness of information
3. Consider the directness of the answer

Please return the indexed list in JSON format and no more words, with the most relevant documents indexed first.

Example input format:
Document 0: doc_content
Document 1: doc_content
Document 2: doc_content
Document 3: doc_content

Example output format:
{
    "ranked_indices": [0, 2, 1, 3]
}
Remember: Your output MUST contain all indices from 0 to N-1 exactly once, where N is the total number of input documents.
DO NOT skip any indices in your output. 
If multiple Documents are useless or have the same score, simply sort them in their original sequence. 
"""},
            {"role": "user", "content": f"""
    Query: {query} \n

    Document List：
    {prompt_result}"""}
        ]

        prompt = llm.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = llm(prompt, max_tokens=200, return_full_text=False)[0]['generated_text']

        result = re.search(r'\{.*?\}', result, re.S).group() if re.search(r'\{.*?\}', result, re.S) else ""
        print(result)
        try:
            ranked_indices = json.loads(result)["ranked_indices"]
            reranked_docs = [docs[i] for i in ranked_indices]
        except:
            reranked_docs = docs

    # other modes can be expanded here

    return reranked_docs[:top_k], similarity_scores


class JointRetrieval:
    """
    Main class for joint retrieval implementation.
    
    This class implements a complete retrieval pipeline, including:
    - Query classification (determining if retrieval is necessary)
    - Query optimization
    - Document retrieval
    - Document reranking
    - Document compression
    
    Attributes:
        device: Device to use for computation
        retrieval_config: Retrieval configuration
        saturn_token: Authentication token
        llamatokenizer: Tokenizer for the Llama model
        embed_model: Embedding model
        index_store: Vector store index
        collection_name: Collection name
        milvus_id: Milvus ID
        id_to_text: Dictionary mapping IDs to text
    """
    
    def __init__(
            self,
            saturn_token: str,
            llamatokenizer: Any,
            embed_model: Any,
            collection_name: str = None,
            milvus_id: int = 1,
            retrieval_config: Optional[Dict] = None,
            device: str = "auto",
            embed_dim: int = 0,
            index_store: Any = None,
            llm_mode: str = "hf",
            llm=None,
    ):
        """
        Initialize the JointRetrieval.
        
        Args:
            saturn_token: Authentication token
            llamatokenizer: Tokenizer for the Llama model
            embed_model: Embedding model
            collection_name: Collection name
            milvus_id: Milvus ID
            retrieval_config: Retrieval configuration
            device: Device to use for computation
            embed_dim: Embedding dimension
            index_store: Vector store index
        """
        super().__init__()
        self.device = device
        self.retrieval_config = retrieval_config or {}
        self.saturn_token = saturn_token
        self.milvus_id = milvus_id
        self.collection_name = collection_name
        self.llm_mode = llm_mode
        self.llamatokenizer = llamatokenizer
        self.embed_model = embed_model
        self.indexStore = index_store
        self.id_to_text = {}

        # Initialize all required models
        self.llm = llm
        if self.llm is None:
            self._initialize_models()
        self.normal_model = self.llm
        self.retrieval_classification_model = self.llm
        self.rerank_model = self.llm
        self.compressor_model = self.llm
        self.retrieval_classification_tokenizer = self.llamatokenizer
        self.rerank_tokenizer = self.llamatokenizer
        self.compressor_tokenizer = self.llamatokenizer

    def _initialize_models(self) -> None:
        from llm_interface import create_llm
        from utils import DEFAULT_HF_TOKEN

        nim_client = NimClient(self.saturn_token)
        model_name = self.retrieval_config.get("model_name", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.llm = create_llm(
            mode=self.llm_mode,
            model_name=model_name,
            hf_token=DEFAULT_HF_TOKEN,
            saturn_token=self.saturn_token
        )

        from llama_index.core import Settings
        from llama_index.core.llms import CustomLLM, CompletionResponse
        class WrappedLLM(CustomLLM):
            def __init__(self, llm_interface):
                super().__init__()
                self.llm_interface = llm_interface

            def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
                response = self.llm_interface(prompt, **kwargs)[0]['generated_text']
                return CompletionResponse(text=response)

            @llm_completion_callback()
            def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
                response = self.llm_interface(prompt, **kwargs)[0]['generated_text']
                text = ""
                text += response
                yield CompletionResponse(text=text, delta=response)

            @property
            def metadata(self):
                return LLMMetadata(
                    context_window=4096,
                    num_output=1024,
                    model_name="unified-llm"
                )

        llm_index = WrappedLLM(self.llm)
        Settings.llm = llm_index
        Settings.embed_model = self.embed_model
        Settings.tokenizer = self.llamatokenizer

    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> float:
        """
        Build or load an index for the given documents.
        
        Args:
            documents: List of documents
            force_rebuild: Whether to force rebuilding the index
            
        Returns:
            float: Time taken to build the index
        """
        start_time = time.time()
        index_type = self.retrieval_config['index_method']
        persist_path = Path(self.retrieval_config.get('persist_dir', 'outputs/indices')) / self.collection_name / index_type
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Process documents into nodes based on dataset type
        if self.retrieval_config.get('dataset') == 'arxiv':
            # Use sentence window parsing for arxiv
            node_parser_sentence_window = SentenceWindowNodeParser.from_defaults(
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
            nodes_sentence_window = node_parser_sentence_window.get_nodes_from_documents(documents)
            
            # Import the filter function directly
            from data_loaders import filter_large_nodes
            nodes = filter_large_nodes(nodes_sentence_window)
        else:
            # Use simple parsing for other datasets
            parser = SimpleNodeParser(chunk_size=100000)
            nodes = parser.get_nodes_from_documents(documents)
            
        # Customize node IDs
        for node in nodes:
            doc_id = node.ref_doc_id
            if doc_id.endswith('_node'):
                doc_id = doc_id[:-5]
            node.node_id = doc_id
            self.id_to_text[node.node_id] = node.text
            
        # Print node statistics
        print(f'Number of nodes: {len(nodes)}')
        if len(nodes) >= 2:
            print(f'First node length: {len(self.llamatokenizer.encode(nodes[0].text))}')
            print(f'Second node length: {len(self.llamatokenizer.encode(nodes[1].text))}')
        print(f'Average node length: {sum(len(self.llamatokenizer.encode(node.text)) for node in nodes) / len(nodes)}')
        
        # Load or create index
        if not force_rebuild and persist_path.exists():
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
            index = load_index_from_storage(storage_context)
            print(f"Successfully loaded existing {index_type} index: {self.collection_name}")
        else:
            if index_type == 'vector' or index_type.lower() == 'vectorstoreindex':
                index = VectorStoreIndex(nodes, use_async=True, show_progress=True)
            else:
                # Import necessary index types
                from llama_index.core.indices import (
                    PropertyGraphIndex, TreeIndex, KeywordTableIndex, 
                    DocumentSummaryIndex, KnowledgeGraphIndex
                )
                
                if index_type == 'KnowledgeGraphIndex':
                    index = KnowledgeGraphIndex(nodes, show_progress=True, use_async=True, include_embeddings=True)
                elif index_type == 'TreeIndex':
                    index = TreeIndex(nodes, show_progress=True, use_async=True, num_children=10)
                elif index_type == 'KeywordTableIndex':
                    index = KeywordTableIndex(nodes, show_progress=True, use_async=True, similarity_top_k=10)
                elif index_type == 'DocumentSummaryIndex':
                    index = DocumentSummaryIndex(nodes, show_progress=True, use_async=True)
                else:
                    index = VectorStoreIndex(nodes, use_async=True, show_progress=True)
                    
            # Persist the index
            index.storage_context.persist(persist_dir=str(persist_path))
            print(f'Successfully created {index_type} index')
            
        self.indexStore = index
        end_time = time.time()
        return end_time - start_time

    def should_retrieval(self, query: str) -> bool:
        """
        Determine if retrieval should be performed.
        
        Args:
            query: Query text
            
        Returns:
            bool: True if retrieval should be performed, False otherwise
        """
        if self.retrieval_config.get("with_retrieval_classification"):
            return llm_task_cls(
                self.retrieval_classification_model,
                self.retrieval_classification_tokenizer,
                query
            )
        return True

    def query_optimization(self, query: str) -> Any:
        """
        Optimize the query if needed.
        
        Args:
            query: Query text
            
        Returns:
            Any: Optimized query
        """
        if not self.retrieval_config.get("with_query_optimization"):
            return query

        query_method = self.retrieval_config.get("query_method")
        if query_method == "hyde":
            hyde = HyDEQueryTransform(include_original=True)
            pseudo_doc = hyde.run(query)
            return pseudo_doc
        return query

    def _search(self, query: str, retrieval_config: Optional[Dict] = None) -> List[str]:
        """
        Perform search operation.
        
        Args:
            query: Query text
            retrieval_config: Optional retrieval configuration that overrides the instance config
            
        Returns:
            List[str]: List of retrieved documents
        """
        config = retrieval_config or self.retrieval_config
        top_k = config.get("search_k", 5)
        
        try:
            nodes = self.indexStore.as_retriever(similarity_top_k=top_k).retrieve(query)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

        results = []
        for rank, node in enumerate(nodes, 1):
            text = node.text
            # Truncate very long texts
            words = text.split()
            truncated_text = " ".join(words[:500]) if len(words) > 500 else text
            results.append(truncated_text)

            # Print retrieval results for debugging
            print("=" * 100)
            print(f"{rank}   {node.node_id}\t{node.score}\n{text[:300]}...")

        return results

    def _rerank(self, query: str, docs: List[str]) -> List[str]:
        """
        Rerank documents if enabled.
        
        Args:
            query: Query text
            docs: List of documents
            
        Returns:
            List[str]: Reranked list of documents
        """
        if not self.retrieval_config.get("with_rerank"):
            return docs

        reranked_docs, _ = llm_rerank(
            self.rerank_model,
            self.rerank_tokenizer,
            mode=self.retrieval_config.get("rerank_model"),
            query=query,
            docs=docs,
            top_k=self.retrieval_config.get("top_k")
        )
        return reranked_docs

    def _compress(self, query: str, docs: List[str]) -> Tuple[str, float]:
        """
        Compress documents if enabled.
        
        Args:
            query: Query text
            docs: List of documents
            
        Returns:
            Tuple[str, float]: Compressed documents and compression ratio
        """
        doc = "".join(docs).strip()
        if not self.retrieval_config.get("with_compress"):
            return docs, 0

        compressed_docs = llm_compress(
            self.compressor_model,
            self.compressor_tokenizer,
            query,
            doc
        )

        # Calculate compression ratio
        token_pre = len(self.compressor_tokenizer.encode(doc))
        token_after = len(self.compressor_tokenizer.encode(compressed_docs))
        token_diff = token_after / token_pre

        print(f"Compression tokens: {token_pre} -> {token_after}")
        print(f"Compression ratio: {token_diff}")

        return (compressed_docs, token_diff) if token_diff < 1 else (docs, 1)

    def retrieval_all(self, query: str) -> Tuple[str, float]:
        """
        Execute the complete retrieval pipeline.
        
        Args:
            query: Query text
            
        Returns:
            Tuple[str, float]: Retrieved documents and compression ratio
        """
        print(f"Processing query: {query}")

        if not self.should_retrieval(query):
            return "", 0

        optimized_query = self.query_optimization(query)
        search_docs = self._search(optimized_query)
        reranked_docs = self._rerank(query, search_docs)
        compressed_docs, token_diff = self._compress(query, reranked_docs)

        return compressed_docs, token_diff


# Create a function to generate text with a retrieved context
def generate_with_retrieved_context(model, tokenizer, retrieved_context: str, question: str) -> str:
    """
    Generate a response to a question based on the retrieved context.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        retrieved_context: Retrieved context
        question: Question to answer
        
    Returns:
        str: Generated response
    """
    template = [
        {"role": "system", "content": 'You are an information retrieval specialist. You are able to find answers to the questions from the contextual passage snippets provided.'},
        {"role": "user", "content": f'Based on the following pieces of information, write a short answer for the following question in a few words. If there is no or enough context, just answer based on your own knowledge. \nContext:{retrieved_context}\nQuestion:{question}'}
    ]
    
    prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)

    sequences = model(
        prompt,
        max_tokens=500,
        do_sample=True,
        top_k=10,
        temperature=0.4,
        top_p=0.9,
        return_full_text=False,
        num_return_sequences=1,
    )
    
    return str(sequences[0]['generated_text'])


def create_retrieval_config(
    search_k: int = 5,
    index_method: str = "vector",
    dataset: str = "hotpotqa",
    force_rebuild: bool = False,
    data_volume: int = 100,
    top_k: int = 5,
    vector_store: str = "milvus",
    with_retrieval_classification: bool = True,
    with_query_optimization: bool = True,
    with_rerank: bool = True,
    with_compress: bool = True,
    classification_model: str = "llama3",
    query_method: str = "hyde",
    search_method: str = "original",
    rerank_model: str = "llama3",
    compression_method: str = "llama3",
    repack_method: str = "sides",
    compression_ratio: float = 0.4,
) -> Dict[str, Any]:
    """
    Create a retrieval configuration dictionary.
    
    Args:
        Various configuration parameters
        
    Returns:
        Dict[str, Any]: Retrieval configuration dictionary
    """
    return {
        "search_k": search_k,
        "index_method": index_method,
        "dataset": dataset,
        "force_rebuild": force_rebuild,
        "data_volume": data_volume,
        "top_k": top_k,
        "Vector_Store": vector_store,
        "with_retrieval_classification": with_retrieval_classification,
        "with_query_optimization": with_query_optimization,
        "with_rerank": with_rerank,
        "with_compress": with_compress,
        "classification_model": classification_model,
        "query_method": query_method,
        "search_method": search_method,
        "rerank_model": rerank_model,
        "compression_method": compression_method,
        "repack_method": repack_method,
        "compression_ratio": compression_ratio,
    }