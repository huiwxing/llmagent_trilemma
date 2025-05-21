"""
Unified evaluation framework for retrieval and generation tasks.

This module provides classes and functions for evaluating retrieval and generation 
performance with various metrics including energy efficiency.
"""

import sys
import os
import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import numpy as np
import nest_asyncio

from transformers import AutoTokenizer
from uptrain import EvalLLM, Evals, Settings, ResponseMatching
from llama_index.core.evaluation import RetrieverEvaluator, EmbeddingQAFinetuneDataset
from llama_index.core.schema import QueryBundle, QueryType, NodeWithScore
from llama_index.core import (
    VectorStoreIndex,
    Document,
    load_index_from_storage,
    StorageContext
)

from energy_modeling import UnifiedEnergyMonitor, LLMEnergyModel
from retrieval import JointRetrieval, generate_with_retrieved_context, create_retrieval_config


# Apply nest_asyncio for Jupyter notebook compatibility
nest_asyncio.apply()


class BaseEvaluator:
    """
    Base class for all evaluators.
    
    Attributes:
        retrieval_config (Dict): Configuration for retrieval
        tokenizer: Tokenizer for the model
        SATURN_TOKEN (str): Auth token for API access
        collection_name (str): Name of the collection
        embed_model: Embedding model
        embed_dim (int): Dimension of embeddings
        persist_dir (str): Directory for persisting indices
        OPENAI_API_KEY (str): OpenAI API key for evaluation
    """
    
    def __init__(
            self,
            retrieval_config: Dict[str, Any],
            tokenizer,
            SATURN_TOKEN: str,
            collection_name: str = None,
            embed_model = None,
            embed_dim: int = 0,
            persist_dir: str = "outputs/indices",
            OPENAI_API_KEY: str = None
    ):
        """
        Initialize the base evaluator.
        
        Args:
            retrieval_config: Configuration for retrieval
            tokenizer: Tokenizer for the model
            SATURN_TOKEN: Auth token for API access
            collection_name: Name of the collection
            embed_model: Embedding model
            embed_dim: Dimension of embeddings
            persist_dir: Directory for persisting indices
            OPENAI_API_KEY: OpenAI API key for evaluation
        """
        self.retrieval_config = retrieval_config
        self.tokenizer = tokenizer
        self.SATURN_TOKEN = SATURN_TOKEN
        self.collection_name = collection_name or retrieval_config.get('dataset')
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        
        self.persist_dir = Path(persist_dir) / self.collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.OPENAI_API_KEY = OPENAI_API_KEY
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> float:
        """
        Build the index from documents.
        
        Args:
            documents: List of documents
            force_rebuild: Whether to force rebuilding the index
            
        Returns:
            float: Time taken to build the index
        """
        raise NotImplementedError("Subclasses must implement build_index")
    
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate")


class RetrievalEvaluator(BaseEvaluator):
    """
    Evaluator for retrieval tasks.
    
    This class evaluates the retrieval component of a system, measuring metrics
    like relevance, accuracy, and energy efficiency.
    """
    
    def __init__(self, *args,  llm=None, llm_mode="hf", **kwargs):
        """Initialize the retrieval evaluator."""
        super().__init__(*args, **kwargs)
        
        # Initialize retrieval system
        self.joint_retriever = JointRetrieval(
            retrieval_config=self.retrieval_config,
            saturn_token=self.SATURN_TOKEN,
            llamatokenizer=self.tokenizer,
            embed_model=self.embed_model,
            embed_dim=self.embed_dim,
            collection_name=self.collection_name,
            llm=llm,
            llm_mode=llm_mode
        )
        
        # Initialize index store and ID to text mapping
        self.index = None
        self.id_to_text = self.joint_retriever.id_to_text

    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> float:
        """
        Build the index from documents.

        Args:
            documents: List of documents
            force_rebuild: Whether to force rebuilding the index

        Returns:
            float: Time taken to build the index
        """

        # Build the index using the joint retriever
        index_time = self.joint_retriever.build_index(documents, force_rebuild)
        self.index = self.joint_retriever.indexStore

        self.id_to_text = self.joint_retriever.id_to_text

        for doc in documents:
            doc_id = doc.doc_id if doc.doc_id else f"doc_{id(doc)}"
            if doc_id.endswith('_node'):
                doc_id = doc_id[:-5]
            if doc_id not in self.id_to_text:
                self.id_to_text[doc_id] = doc.text

        return index_time
    
    def generate_all(self, retrieval_eval: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, float, float]:
        """
        Generate evaluation results for all queries.
        
        Args:
            retrieval_eval: List of evaluation items
            
        Returns:
            Tuple containing results, time taken, and metrics
        """
        start_time = time.time()
        top_k = self.retrieval_config.get("top_k", 5)
        
        # Get the retriever based on the index method
        index_type = self.retrieval_config.get('index_method')
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Setup retriever based on index type
        if index_type == 'vector' or index_type.lower() == 'vectorstoreindex':
            retl = self.index.as_retriever(similarity_top_k=top_k)
        elif index_type == 'KnowledgeGraphIndex':
            retl = self.index.as_retriever(include_text=True, retriever_mode="embedding", similarity_top_k=top_k)
        elif index_type == 'TreeIndex':
            from llama_index.core.indices.tree.base import TreeRetrieverMode
            retl = self.index.as_retriever(retriever_mode=TreeRetrieverMode.SELECT_LEAF_EMBEDDING)
        elif index_type == 'KeywordTableIndex':
            retl = self.index.as_retriever(retriever_mode="simple")
        elif index_type == 'DocumentSummaryIndex':
            retl = self.index.as_retriever()
        else:
            retl = self.index.as_retriever(similarity_top_k=top_k)
        
        # Create custom evaluator for retrieval
        class CustomRetrieverEvaluator(RetrieverEvaluator):
            @classmethod
            def from_metric_names(cls, metric_names: List[str], retriever, **kwargs):
                evaluator = super().from_metric_names(
                    metric_names=metric_names,
                    retriever=retriever,
                    **kwargs
                )
                return cls(
                    metrics=evaluator.metrics,
                    retriever=evaluator.retriever,
                    node_postprocessors=evaluator.node_postprocessors
                )
            
            def evaluate(self, query: str, expected_ids: List[str]) -> Dict[str, Any]:
                # Get retrieved IDs and texts
                retrieved_ids, retrieved_texts = asyncio.run(
                    self._aget_retrieved_ids_and_texts(query)
                )

                # Compute all metrics
                metric_outputs = {}
                for metric in self.metrics:
                    result = metric.compute(
                        retrieved_ids=retrieved_ids,
                        expected_ids=expected_ids
                    )
                    metric_outputs[metric.metric_name] = result.score
                
                return {
                    "metrics": metric_outputs,
                    "retrieved_ids": retrieved_ids,
                    "texts": retrieved_texts
                }
        
        # Initialize evaluator with metrics
        evaluator = CustomRetrieverEvaluator.from_metric_names(
            metric_names=["recall", "mrr"],
            retriever=retl
        )
        
        all_answers = []
        metrics_1, metrics_2 = 0, 0  # MRR and Recall metrics
        
        for i, ret in enumerate(retrieval_eval):
            print(f'Evaluating question: {i+1}/{len(retrieval_eval)}')
            
            # Evaluate retrieval
            eval_results = evaluator.evaluate(
                query=ret['query'],
                expected_ids=ret['expected_ids'],
            )

            # Update metrics
            metrics_1 += eval_results['metrics'].get('mrr', 0)
            metrics_2 += eval_results['metrics'].get('recall', 0)
            
            # Get valid IDs for context construction
            valid_ids = [ids for ids in eval_results['retrieved_ids'] if ids in self.id_to_text]
            valid_ids = valid_ids[:top_k]  # Limit to top_k
            
            # Construct retrieved context
            retrieved_context = ''.join(self.id_to_text[ids] for ids in valid_ids)
            
            # Add to results
            result = {
                'question': ret['query'],
                'context': retrieved_context,
                "ground_truth": ret['answer'],
            }
            all_answers.append(result)
        
        end_time = time.time()
        
        return all_answers, end_time - start_time, metrics_1, metrics_2
    
    def evaluate(self, 
                 documents: List[Document], 
                 retrieval_eval: List[Dict[str, Any]],
                 index_time: float = None) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            documents: List of documents
            retrieval_eval: List of evaluation items
            index_time: Time taken to build the index (if already measured)
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Measure indexing energy if not provided
        if index_time is None:
            monitor = UnifiedEnergyMonitor(cpu_sampling_interval=0.5, include_idle=True)
            index_time, indexing_measurements = monitor.measure_energy(
                lambda: self.build_index(documents, self.retrieval_config.get('force_rebuild', False))
            )
            print(f"Indexing energy: {indexing_measurements['total_energy']:.2f} J")
        
        # Measure retrieval energy
        monitor = UnifiedEnergyMonitor(cpu_sampling_interval=0.5, include_idle=True)
        results = monitor.measure_energy(
            lambda: self.generate_all(retrieval_eval)
        )
        
        # Unpack results
        (all_answer, retr_time, metrics_1, metrics_2), retrieval_measurements = results

        # Initialize metrics dictionary
        metrics = {
            'CONTEXT_RELEVANCE': 0, 
            'CONTEXT_ACCURACY': (metrics_1 + metrics_2) / (len(all_answer) * 2),
            'questions_num': len(all_answer), 
            'avg_index_time': 0, 
            'avg_retr_time': 0,
            'total_input_token_num': sum(len(self.tokenizer.encode(doc.text)) for doc in documents),
            'total_retr_token_num': 0,
            'index_energy': indexing_measurements['total_energy'] if 'indexing_measurements' in locals() else 0,
            'retrieval_energy': retrieval_measurements['total_energy']
        }
        
        # Evaluate context relevance using OpenAI
        settings = Settings(model='gpt-4o-mini', openai_api_key=self.OPENAI_API_KEY)
        eval_llm = EvalLLM(settings)
        
        eval_results = eval_llm.evaluate(
            data=all_answer,
            checks=[Evals.CONTEXT_RELEVANCE],
        )
        
        # Calculate metrics
        metrics['CONTEXT_RELEVANCE'] = sum(res['score_context_relevance'] for res in eval_results) / len(eval_results)
        metrics['total_retr_token_num'] = sum(len(self.tokenizer.encode(ans['context'])) for ans in all_answer)
        metrics['avg_index_time'] = index_time / metrics['total_input_token_num']
        metrics['avg_retr_time'] = retr_time / metrics['total_retr_token_num']
        
        # Add energy metrics
        metrics['avg_index_energy_per_token'] = metrics['index_energy'] / metrics['total_input_token_num']
        metrics['avg_retrieval_energy_per_token'] = metrics['retrieval_energy'] / metrics['total_retr_token_num']
        
        print("Retrieval Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        return metrics


class GenerationEvaluator(BaseEvaluator):
    """
    Evaluator for generation tasks.
    
    This class evaluates the generation component of a system, measuring metrics
    like response quality, consistency, and energy efficiency.
    """
    
    def __init__(self, *args,  llm=None, llm_mode="hf", **kwargs):
        """Initialize the generation evaluator."""
        super().__init__(*args, **kwargs)
        
        # Initialize retrieval system
        self.joint_retriever = JointRetrieval(
            retrieval_config=self.retrieval_config,
            saturn_token=self.SATURN_TOKEN,
            llamatokenizer=self.tokenizer,
            embed_model=self.embed_model,
            embed_dim=self.embed_dim,
            collection_name=self.collection_name,
            llm=llm,
            llm_mode=llm_mode
        )
        
        # QA pair storage
        self.qa_pairs = {}
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> float:
        """
        Build the index from documents.
        
        Args:
            documents: List of documents
            force_rebuild: Whether to force rebuilding the index
            
        Returns:
            float: Time taken to build the index
        """
        return self.joint_retriever.build_index(documents, force_rebuild)
    
    def set_qa_pairs(self, qa_pairs: Dict[str, Any]):
        """
        Set the QA pairs for evaluation.
        
        Args:
            qa_pairs: Dictionary mapping question IDs to QA pairs
        """
        self.qa_pairs = qa_pairs
    
    def generate_with_retrieved_context(self, retrieved_context: str, question: str) -> str:
        """
        Generate a response based on the retrieved context.
        
        Args:
            retrieved_context: Retrieved context
            question: Question to answer
            
        Returns:
            str: Generated response
        """
        return generate_with_retrieved_context(
            self.joint_retriever.normal_model,
            self.tokenizer,
            retrieved_context,
            question
        )
    
    def generate_all(self, question_ids: List[str]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Generate responses for all questions.
        
        Args:
            question_ids: List of question IDs
            
        Returns:
            Tuple containing answers and time taken
        """
        start_time = time.time()
        all_answers = []
        compress_ratio = 0
        
        for i, qid in enumerate(question_ids):
            print(f'Generating answer for question {i+1}/{len(question_ids)}')
            
            # Get QA pair
            qa_pair = self.qa_pairs[qid]
            
            # Retrieve context
            retrieved_context, token_diff = self.joint_retriever.retrieval_all(query=qa_pair['question'])
            compress_ratio += token_diff
            
            # Handle list context (if no compressor was used)
            if isinstance(retrieved_context, list):
                retrieved_context = "".join(retrieved_context).strip()
            
            # Check if context is empty
            if retrieved_context is None or retrieved_context.strip() == "":
                print("Warning: Empty context retrieved")
                retrieved_context = ""
            
            # Generate response
            response = self.generate_with_retrieved_context(retrieved_context, qa_pair['question'])
            
            # Store result
            result = {
                'question': qa_pair['question'],
                'response': response,
                "ground_truth": qa_pair['answer'],
                'context': retrieved_context,
            }
            all_answers.append(result)
        
        end_time = time.time()
        print(f'Average compression ratio: {compress_ratio / len(question_ids)}')
        
        return all_answers, end_time - start_time
    
    def evaluate(self, documents: List[Document], question_ids: List[str]) -> Dict[str, float]:
        """
        Evaluate generation performance.
        
        Args:
            documents: List of documents
            question_ids: List of question IDs
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Build index with energy monitoring
        monitor = UnifiedEnergyMonitor(cpu_sampling_interval=0.5, include_idle=True)
        _, index_measurements = monitor.measure_energy(
            lambda: self.build_index(documents, self.retrieval_config.get('force_rebuild', False))
        )
        
        # Generate responses with energy monitoring
        monitor = UnifiedEnergyMonitor(cpu_sampling_interval=0.5, include_idle=True)
        results = monitor.measure_energy(
            lambda: self.generate_all(question_ids)
        )
        
        # Unpack results
        (all_answer, response_time), generation_measurements = results
        
        # Initialize metrics dictionary
        metrics = {
            'RESPONSE_CONSISTENCY': 0,
            'RESPONSE_RELEVANCE': 0,
            'ResponseMatching': 0,
            'questions_num': len(all_answer),
            'avg_response_time': 0,
            'total_response_token_num': 0,
            'index_energy': index_measurements['total_energy'],
            'generation_energy': generation_measurements['total_energy']
        }
        
        # Evaluate with OpenAI
        settings = Settings(model='gpt-4o-mini', openai_api_key=self.OPENAI_API_KEY)
        eval_llm = EvalLLM(settings)
        
        eval_results = eval_llm.evaluate(
            data=all_answer,
            checks=[
                Evals.RESPONSE_CONSISTENCY, 
                Evals.RESPONSE_RELEVANCE, 
                ResponseMatching(method='rouge')
            ],
        )
        
        # Calculate metrics
        metrics['RESPONSE_CONSISTENCY'] = sum(res['score_response_consistency'] for res in eval_results) / len(eval_results)
        metrics['RESPONSE_RELEVANCE'] = sum(res['score_response_relevance'] for res in eval_results) / len(eval_results)
        metrics['ResponseMatching'] = sum(res['score_response_match_rouge'] for res in eval_results) / 100 / len(eval_results)
        metrics['total_response_token_num'] = sum(len(self.tokenizer.encode(ans['response'])) for ans in all_answer)
        metrics['avg_response_time'] = response_time / metrics['total_response_token_num']
        
        # Add energy metrics
        total_input_token_num = sum(len(self.tokenizer.encode(doc.text)) for doc in documents)
        metrics['avg_index_energy_per_token'] = metrics['index_energy'] / total_input_token_num
        metrics['avg_generation_energy_per_token'] = metrics['generation_energy'] / metrics['total_response_token_num']
        
        print("Generation Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        return metrics


class GEORMetricsEvaluator(GenerationEvaluator):
    """
    Evaluator for the Generative Energy Optimization Ratio (GEOR) metrics.
    
    This class extends the generation evaluator to compute energy efficiency metrics
    for retrieval-augmented generation systems.
    """
    
    def __init__(self, energy_model_params: np.ndarray = None, *args, **kwargs):
        """
        Initialize the GEOR metrics evaluator.
        
        Args:
            energy_model_params: Parameters for the energy model
            *args, **kwargs: Additional arguments for the base class
        """
        super().__init__(*args, **kwargs)
        self.params = energy_model_params
        
        # Template for generation
        self.template = [
            {"role": "system", "content": 'You are an information retrieval specialist. You are able to find answers to the questions from the contextual passage snippets provided.'},
            {"role": "user", "content": 'Based on the following pieces of information, write a short answer for the following question in a few words. If there is no or enough context, just answer based on your own knowledge. \nContext:%s\nQuestion:%s'}
        ]
    
    def evaluate(self, documents: List[Document], question_ids: List[str]) -> Dict[str, float]:
        """
        Evaluate GEOR metrics.
        
        Args:
            documents: List of documents
            question_ids: List of question IDs
            
        Returns:
            Dict[str, float]: GEOR evaluation metrics
        """
        # Initialize metrics dictionary
        metrics = {
            'ResponseMatching': 0,
            'correct_num': 0,
            'real_energy': 0,
            'optimal_energy': 0,
            'GEOR': 0
        }
        
        # Make sure index is built
        self.build_index(documents, self.retrieval_config.get('force_rebuild', False))
        
        # Setup evaluation
        settings = Settings(model='gpt-4o-mini', openai_api_key=self.OPENAI_API_KEY)
        eval_llm = EvalLLM(settings)
        
        for i, qid in enumerate(question_ids):
            print(f'Evaluating GEOR for question {i+1}/{len(question_ids)}')
            
            # Get QA pair
            qa_pair = self.qa_pairs[qid]
            
            # Create optimal config (no optimizations)
            optimal_config = self.retrieval_config.copy()
            optimal_config["with_retrieval_classification"] = False
            optimal_config["with_query_optimization"] = False
            optimal_config["with_rerank"] = False
            optimal_config["with_compress"] = False
            
            # Measure optimal retrieval energy
            monitor1 = UnifiedEnergyMonitor(cpu_sampling_interval=0.02, include_idle=True)
            result, measurements1 = monitor1.measure_energy(
                lambda: self.joint_retriever._search(
                    query=qa_pair['question'], 
                    retrieval_config=optimal_config
                )
            )
            
            # Measure actual retrieval energy
            monitor2 = UnifiedEnergyMonitor(cpu_sampling_interval=0.02, include_idle=True)
            (retrieved_context, token_diff), measurements2 = monitor2.measure_energy(
                lambda: self.joint_retriever.retrieval_all(query=qa_pair['question'])
            )
            
            # Handle list context
            if isinstance(retrieved_context, list):
                retrieved_context = "".join(retrieved_context).strip()
            
            # Check if context is empty
            if retrieved_context is None or retrieved_context.strip() == "":
                print("Warning: Empty context retrieved")
                retrieved_context = ""
            
            # Measure generation energy
            monitor3 = UnifiedEnergyMonitor(cpu_sampling_interval=0.02, include_idle=True)
            response, measurements3 = monitor3.measure_energy(
                lambda: self.generate_with_retrieved_context(
                    retrieved_context, 
                    qa_pair['question']
                )
            )
            
            # Create result for evaluation
            result = {
                'question': qa_pair['question'],
                'response': response,
                "ground_truth": str(qa_pair['answer']),
                'context': retrieved_context,
            }
            
            # Evaluate response quality
            eval_results = eval_llm.evaluate(
                data=[result],
                checks=[ResponseMatching(method='rouge')],
            )
            
            score = eval_results[0]['score_response_match_rouge']
            total_energy = measurements2['total_energy'] + measurements3['total_energy']
            opt_retrieval_energy = measurements1['total_energy']
            
            # Only count correct answers (score >= 80)
            if score >= 80:
                metrics['correct_num'] += 1
                metrics['ResponseMatching'] += score
                
                # Calculate token counts
                output_tokens_num = len(self.tokenizer.encode(result['ground_truth']))
                
                # Get ground truth context
                if 'expected_ids' in qa_pair:
                    ground_truth_doc = ''.join(
                        self.joint_retriever.id_to_text.get(ids, "") 
                        for ids in qa_pair['expected_ids']
                    )
                    
                    # Calculate input tokens
                    input_tokens_num = len(self.tokenizer.encode(result['question'])) + len(
                        self.tokenizer.encode(ground_truth_doc)
                    )
                    
                    # Add template tokens
                    for message in self.template:
                        tokens = self.tokenizer.encode(message["content"])
                        input_tokens_num += len(tokens)
                    
                    # Remove %s tokens from count
                    input_tokens_num -= 2
                    
                    # Calculate optimal energy
                    optimal_gene_energy = self._energy_function(
                        self.params, 
                        input_tokens=input_tokens_num,
                        output_tokens=output_tokens_num
                    )
                    
                    # Update metrics
                    metrics['optimal_energy'] += optimal_gene_energy + opt_retrieval_energy
                    metrics['real_energy'] += total_energy
                    # metrics['GEOR'] += (optimal_gene_energy + opt_retrieval_energy) / total_energy
        
        # Calculate averages for metrics
        if metrics['correct_num'] > 0:
            metrics['ResponseMatching'] = metrics['ResponseMatching'] / metrics['correct_num']
            metrics['optimal_energy'] = metrics['optimal_energy'] / metrics['correct_num']
            metrics['real_energy'] = metrics['real_energy'] / metrics['correct_num']
            metrics['GEOR'] = metrics['optimal_energy'] / metrics['real_energy']
        
        print("GEOR Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        return metrics
    
    def _energy_function(self, params: np.ndarray, input_tokens: int, output_tokens: int) -> float:
        """
        Energy consumption function: e_K(τin, τout) = αK,0*τin + αK,1*τout + αK,2*τin*τout
        
        Args:
            params: Model parameters
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            float: Predicted energy consumption
        """
        alpha_0, alpha_1, alpha_2 = params
        return (alpha_0 * input_tokens +
                alpha_1 * output_tokens +
                alpha_2 * input_tokens * output_tokens)


def create_evaluator(
    eval_type: str,
    retrieval_config: Dict[str, Any],
    tokenizer,
    SATURN_TOKEN: str,
    embed_model = None,
    embed_dim: int = 0,
    energy_model_params: np.ndarray = None,
    OPENAI_API_KEY: str = None,
    llm=None,
    llm_mode: str = "hf"
) -> BaseEvaluator:
    """
    Factory function to create the appropriate evaluator.
    
    Args:
        eval_type: Type of evaluator to create
        retrieval_config: Configuration for retrieval
        tokenizer: Tokenizer for the model
        SATURN_TOKEN: Auth token for API access
        embed_model: Embedding model
        embed_dim: Dimension of embeddings
        energy_model_params: Parameters for the energy model
        OPENAI_API_KEY: OpenAI API key for evaluation
        
    Returns:
        BaseEvaluator: The created evaluator
    """
    if eval_type == "retrieval":
        return RetrievalEvaluator(
            retrieval_config=retrieval_config,
            tokenizer=tokenizer,
            SATURN_TOKEN=SATURN_TOKEN,
            embed_model=embed_model,
            embed_dim=embed_dim,
            OPENAI_API_KEY=OPENAI_API_KEY,
            llm=llm,
            llm_mode=llm_mode
        )
    elif eval_type == "generation":
        return GenerationEvaluator(
            retrieval_config=retrieval_config,
            tokenizer=tokenizer,
            SATURN_TOKEN=SATURN_TOKEN,
            embed_model=embed_model,
            embed_dim=embed_dim,
            OPENAI_API_KEY=OPENAI_API_KEY,
            llm=llm,
            llm_mode=llm_mode
        )
    elif eval_type == "geor":
        return GEORMetricsEvaluator(
            energy_model_params=energy_model_params,
            retrieval_config=retrieval_config,
            tokenizer=tokenizer,
            SATURN_TOKEN=SATURN_TOKEN,
            embed_model=embed_model,
            embed_dim=embed_dim,
            OPENAI_API_KEY=OPENAI_API_KEY,
            llm=llm,
            llm_mode=llm_mode
        )
    else:
        raise ValueError(f"Unknown evaluator type: {eval_type}")