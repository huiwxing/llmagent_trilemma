"""
Main entry point for the retrieval system.

This script provides command-line functionality for running various components
of the retrieval and generation system.
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
import numpy as np

# Add script directory to path
script_dir = str(Path(__file__).parent)
sys.path.insert(0, script_dir)

# Import our modules
from data_loaders import load_data
from energy_modeling import UnifiedEnergyMonitor, LLMEnergyModel
from retrieval import JointRetrieval, create_retrieval_config, generate_with_retrieved_context
from evaluation import create_evaluator
from utils import (
    init_tokenizer_and_model, 
    init_embedding_model,
    get_data_path, 
    save_results, 
    timestamp_filename,
    Timer,
    DEFAULT_HF_TOKEN,
    DEFAULT_SATURN_TOKEN,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_ENERGY_MODEL_PARAMS,
    find_latest_energy_model_file,
    load_results,
    logger
)
from llm_interface import create_llm

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation System")
    
    # Task specification
    parser.add_argument(
        "--task", 
        choices=["fit_energy", "retrieval", "generation", "geor"],
        default="generation",
        help="Task to perform"
    )

    # LLM mode selection
    parser.add_argument(
        "--llm-mode",
        choices=["hf", "nim"],
        default="hf",
        help="Local LLM: HuggingFace or Nvidia NIM"
    )

    # Model configuration
    parser.add_argument(
        "--model", 
        default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
        help="Model name/path"
    )
    parser.add_argument(
        "--hf-token", 
        default=DEFAULT_HF_TOKEN, 
        help="Hugging Face token"
    )
    parser.add_argument(
        "--saturn-token", 
        default=DEFAULT_SATURN_TOKEN, 
        help="Saturn token"
    )
    parser.add_argument(
        "--openai-key", 
        default=DEFAULT_OPENAI_API_KEY, 
        help="OpenAI API key for evaluation"
    )
    parser.add_argument(
        "--embed-model", 
        default="BAAI/bge-base-en-v1.5", 
        help="Embedding model name/path"
    )
    
    # Quantization options
    parser.add_argument(
        "--use-4bit", 
        action="store_true", 
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--use-8bit", 
        action="store_true", 
        help="Use 8-bit quantization"
    )
    
    # Data configuration
    parser.add_argument(
        "--dataset", 
        default="hotpotqa", 
        choices=["hotpotqa", "musique", "locomo", "arxiv"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--data-path", 
        default=None, 
        help="Path to dataset (overrides --dataset)"
    )
    parser.add_argument(
        "--data-volume", 
        default=100, 
        type=int, 
        help="Number of examples to use"
    )
    parser.add_argument(
        "--base-dir", 
        default="", 
        help="Base directory for data paths"
    )
    
    # Retrieval configuration
    parser.add_argument(
        "--index-method", 
        default="vector", 
        choices=["vector", "KnowledgeGraphIndex", "KeywordTableIndex", "DocumentSummaryIndex"],
        help="Indexing method"
    )
    parser.add_argument(
        "--force-rebuild", 
        action="store_true", 
        help="Force rebuilding the index"
    )
    parser.add_argument(
        "--search-k", 
        default=5, 
        type=int, 
        help="Number of documents to retrieve initially"
    )
    parser.add_argument(
        "--top-k", 
        default=5, 
        type=int, 
        help="Number of documents to return after reranking"
    )
    parser.add_argument(
        "--disable-classification", 
        action="store_true", 
        help="Disable retrieval classification"
    )
    parser.add_argument(
        "--disable-query-optimization", 
        action="store_true", 
        help="Disable query optimization"
    )
    parser.add_argument(
        "--disable-rerank", 
        action="store_true", 
        help="Disable reranking"
    )
    parser.add_argument(
        "--disable-compress", 
        action="store_true", 
        help="Disable compression"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-file", 
        default=None,
        help="Output file for results"
    )
    parser.add_argument(
        "--persist-dir", 
        default="outputs/indices",
        help="Directory to persist indices"
    )
    
    # Parse arguments
    return parser.parse_args()


async def fit_energy_model(args):
    """
    Fit an energy consumption model.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Fitted energy model
    """
    logger.info("Fitting energy model...")
    from llm_interface import create_llm
    # Initialize tokenizer and model
    llm = create_llm(
        mode=args.llm_mode,
        model_name=args.model,
        hf_token=args.hf_token,
        saturn_token=args.saturn_token,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit
    )
    # Initialize energy model
    energy_model = LLMEnergyModel(
        model_name=args.model,
        tokenizer=llm,
        model=llm
    )
    
    # Collect measurements
    data_path = args.data_path or get_data_path(args.dataset, args.base_dir)
    measurements = await energy_model.collect_measurements(data_path)
    
    # Fit model
    success = energy_model.fit(measurements)
    
    if success:
        alpha_0, alpha_1, alpha_2 = energy_model.get_coefficients()
        logger.info(f"Fitted coefficients: alpha_0={alpha_0:.6f}, alpha_1={alpha_1:.6f}, alpha_2={alpha_2:.6f}")
        
        # Save coefficients
        output_file = args.output_file
        save_results(
            {"coefficients": [float(alpha_0), float(alpha_1), float(alpha_2)]},
            output_file
        )
    else:
        logger.error("Failed to fit energy model")
    
    return energy_model


def run_retrieval_evaluation(args):
    """
    Run retrieval evaluation.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Evaluation results
    """
    logger.info("Running retrieval evaluation...")
    tokenizer, _ = init_tokenizer_and_model(args.model, args.hf_token, load_model=False)
    llm = create_llm(
        mode=args.llm_mode,
        model_name=args.model,
        hf_token=args.hf_token,
        saturn_token=args.saturn_token,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit
    )
    
    # Initialize embedding model
    embed_model, embed_dim = init_embedding_model(args.embed_model)
    
    # Create retrieval configuration
    retrieval_config = create_retrieval_config(
        search_k=args.search_k,
        index_method=args.index_method,
        dataset=args.dataset,
        force_rebuild=args.force_rebuild,
        data_volume=args.data_volume,
        top_k=args.top_k,
        with_retrieval_classification=not args.disable_classification,
        with_query_optimization=not args.disable_query_optimization,
        with_rerank=not args.disable_rerank,
        with_compress=not args.disable_compress,
    )
    
    # Load data
    data_path = args.data_path or get_data_path(args.dataset, args.base_dir)
    documents, retrieval_eval = load_data(args.dataset, data_path, args.data_volume, mode="retrieval")
    
    # Create evaluator
    evaluator = create_evaluator(
        eval_type="retrieval",
        retrieval_config=retrieval_config,
        tokenizer=tokenizer,
        SATURN_TOKEN=args.saturn_token,
        embed_model=embed_model,
        embed_dim=embed_dim,
        OPENAI_API_KEY=args.openai_key,
        llm=llm,
        llm_mode=args.llm_mode
    )
    
    # Evaluate
    with Timer("Retrieval evaluation"):
        metrics = evaluator.evaluate(documents, retrieval_eval)
    
    # Save results
    if args.output_file:
        save_results(metrics, args.output_file)
    else:
        output_file = timestamp_filename(f"retrieval_eval_{args.dataset}.json")
        save_results(metrics, output_file)
    
    return metrics


def run_generation_evaluation(args):
    """
    Run generation evaluation.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Evaluation results
    """
    logger.info("Running generation evaluation...")
    tokenizer, _ = init_tokenizer_and_model(args.model, args.hf_token, load_model=False)
    llm = create_llm(
        mode=args.llm_mode,
        model_name=args.model,
        hf_token=args.hf_token,
        saturn_token=args.saturn_token,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit
    )
    
    # Initialize embedding model
    embed_model, embed_dim = init_embedding_model(args.embed_model)
    
    # Create retrieval configuration
    retrieval_config = create_retrieval_config(
        search_k=args.search_k,
        index_method=args.index_method,
        dataset=args.dataset,
        force_rebuild=args.force_rebuild,
        data_volume=args.data_volume,
        top_k=args.top_k,
        with_retrieval_classification=not args.disable_classification,
        with_query_optimization=not args.disable_query_optimization,
        with_rerank=not args.disable_rerank,
        with_compress=not args.disable_compress
    )
    
    # Load data
    data_path = args.data_path or get_data_path(args.dataset, args.base_dir)
    documents, questions, qa_pairs = load_data(args.dataset, data_path, args.data_volume, mode="qa")
    
    # Create evaluator
    evaluator = create_evaluator(
        eval_type="generation",
        retrieval_config=retrieval_config,
        tokenizer=tokenizer,
        SATURN_TOKEN=args.saturn_token,
        embed_model=embed_model,
        embed_dim=embed_dim,
        OPENAI_API_KEY=args.openai_key,
        llm=llm,
        llm_mode=args.llm_mode
    )
    
    # Set QA pairs
    evaluator.set_qa_pairs(qa_pairs)
    
    # Evaluate
    with Timer("Generation evaluation"):
        metrics = evaluator.evaluate(documents, questions)
    
    # Save results
    if args.output_file:
        save_results(metrics, args.output_file)
    else:
        output_file = timestamp_filename(f"generation_eval_{args.dataset}.json")
        save_results(metrics, output_file)
    
    return metrics


def run_geor_evaluation(args):
    """
    Run GEOR (Generation Energy Optimality Ratio) evaluation.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Evaluation results
    """
    logger.info("Running GEOR evaluation...")
    tokenizer, _ = init_tokenizer_and_model(args.model, args.hf_token, load_model=False)
    llm = create_llm(
        mode=args.llm_mode,
        model_name=args.model,
        hf_token=args.hf_token,
        saturn_token=args.saturn_token,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit
    )
    
    # Initialize embedding model
    embed_model, embed_dim = init_embedding_model(args.embed_model)
    
    # Create retrieval configuration
    retrieval_config = create_retrieval_config(
        search_k=args.search_k,
        index_method=args.index_method,
        dataset=args.dataset,
        force_rebuild=args.force_rebuild,
        data_volume=args.data_volume,
        top_k=args.top_k,
        with_retrieval_classification=not args.disable_classification,
        with_query_optimization=not args.disable_query_optimization,
        with_rerank=not args.disable_rerank,
        with_compress=not args.disable_compress
    )

    energy_params = DEFAULT_ENERGY_MODEL_PARAMS
    output_dir = os.path.dirname(args.output_file) if args.output_file else "outputs"
    energy_model_file = find_latest_energy_model_file(output_dir, args.model, args.dataset)

    if energy_model_file:
        try:
            logger.info(f"Found energy model file: {energy_model_file}")
            energy_model_data = load_results(energy_model_file)
            if "coefficients" in energy_model_data:
                energy_params = np.array(energy_model_data["coefficients"])
                logger.info(f"Using fitted coefficients: {energy_params}")
            else:
                logger.warning("Energy model file doesn't contain coefficients. Using default parameters.")
        except Exception as e:
            logger.error(f"Error loading energy model file: {e}. Using default parameters.")
    else:
        logger.warning("No energy model file found. Using default parameters.")

    # Load data
    data_path = args.data_path or get_data_path(args.dataset, args.base_dir)
    documents, questions, qa_pairs = load_data(args.dataset, data_path, args.data_volume, mode="geor")
    
    # Create evaluator
    evaluator = create_evaluator(
        eval_type="geor",
        retrieval_config=retrieval_config,
        tokenizer=tokenizer,
        SATURN_TOKEN=args.saturn_token,
        embed_model=embed_model,
        embed_dim=embed_dim,
        energy_model_params=DEFAULT_ENERGY_MODEL_PARAMS,
        OPENAI_API_KEY=args.openai_key,
        llm=llm,
        llm_mode=args.llm_mode
    )

    # Set QA pairs
    evaluator.set_qa_pairs(qa_pairs)
    
    # Evaluate
    with Timer("GEOR evaluation"):
        metrics = evaluator.evaluate(documents, questions)
    
    # Save results
    if args.output_file:
        save_results(metrics, args.output_file)
    else:
        output_file = timestamp_filename(f"geor_eval_{args.dataset}.json")
        save_results(metrics, output_file)
    
    return metrics


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info(f"Starting task: {args.task}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}, volume: {args.data_volume}")
    
    # Run the appropriate task
    if args.task == "fit_energy":
        # Run energy model fitting asynchronously
        asyncio.run(fit_energy_model(args))
    elif args.task == "retrieval":
        run_retrieval_evaluation(args)
    elif args.task == "generation":
        run_generation_evaluation(args)
    elif args.task == "geor":
        run_geor_evaluation(args)
    else:
        logger.error(f"Unknown task: {args.task}")
        sys.exit(1)
    
    logger.info(f"Task {args.task} completed successfully")


if __name__ == "__main__":
    main()