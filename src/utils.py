"""
Utility functions for the project.

This module contains various utility functions and constants used across the project.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retrieval_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Default tokens and keys
def load_config(config_file='env_config.txt'):
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                value = value.strip()

                if (value.startswith('"') and value.endswith('"')) or \
                        (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]

                config[key.strip()] = value
        logger.info(f"Configurations loaded from {config_file}.")
    except FileNotFoundError:
        logger.warning(f"Configuration file not found {config_file}.")

    return config

CONFIG = load_config()
DEFAULT_HF_TOKEN = CONFIG.get('HF_TOKEN', "hf_...")
DEFAULT_SATURN_TOKEN = CONFIG.get('SATURN_TOKEN', 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...')
DEFAULT_OPENAI_API_KEY = CONFIG.get('OPENAI_API_KEY', "sk-proj-CaEHqDIuEeQS1ZW6Ofk7SJNewm0RuI78Tmro...")


# Dataset paths
DATASET_PATHS = {
    "hotpotqa": "data/original_data/hotpotqa/hotpot_dev_distractor_v1.json",
    "musique": "data/original_data/MuSiQue/musique_ans_v1.0_dev.jsonl",
    "locomo": "data/original_data/locomo/locomo10.json",
    "arxiv": "data/original_data/arxiv/benchmark.json"
}


def find_latest_energy_model_file(base_dir="outputs", model_name=None, dataset=None):

    import glob
    import os

    search_pattern = os.path.join(base_dir, "fit_energy_")

    if dataset:
        search_pattern += f"{dataset}_"
    else:
        search_pattern += "*_"

    if model_name:
        model_name_short = model_name.split('/')[-1] if '/' in model_name else model_name
        search_pattern += f"{model_name_short}_*.json"
    else:
        search_pattern += "*.json"

    matching_files = glob.glob(search_pattern)

    if not matching_files:
        search_pattern = os.path.join(base_dir, "**", "fit_energy_*.json")
        matching_files = glob.glob(search_pattern, recursive=True)

    if not matching_files:
        return None

    return max(matching_files, key=os.path.getmtime)

# Default energy model coefficients / should be replaced by Fit Energy Model result
DEFAULT_ENERGY_MODEL_PARAMS = np.array([0.255901, 45.724722, 0.001338])


def init_tokenizer_and_model(model_name: str, hf_token: str = DEFAULT_HF_TOKEN, use_4bit: bool = False, use_8bit: bool = False, 
                            device: str = "auto", load_model: bool = True) -> Tuple[Any, Optional[Any]]:
    """
    Initialize tokenizer and model.
    
    Args:
        model_name: Name of the model to initialize
        hf_token: Hugging Face token
        use_4bit: Whether to use 4-bit quantization
        use_8bit: Whether to use 8-bit quantization
        device: Device to use
        load_model: Whether to load the model (if False, only tokenizer is loaded)
        
    Returns:
        Tuple containing the tokenizer and model (or None if load_model is False)
    """
    # Login to Hugging Face if token is provided
    if hf_token:
        import huggingface_hub
        huggingface_hub.login(hf_token)
        
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Return early if model not needed
    if not load_model:
        return tokenizer, None
    
    # Configure quantization
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device
        )
    elif use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device
        )
    
    # Apply padding configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model


def init_embedding_model(model_name: str = "BAAI/bge-base-en-v1.5") -> Tuple[Any, int]:
    """
    Initialize embedding model.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Tuple containing the model and embedding dimension
    """
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    embed_dim = len(embed_model.get_text_embedding('test'))
    
    return embed_model, embed_dim


def get_data_path(dataset_name: str, base_dir: str = "") -> str:
    """
    Get the path to a dataset.
    
    Args:
        dataset_name: Name of the dataset
        base_dir: Base directory to prepend to the path
        
    Returns:
        str: Path to the dataset
    """
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    path = DATASET_PATHS[dataset_name]
    
    if base_dir:
        return os.path.join(base_dir, path)
    
    return path


def save_results(results: Dict[str, Any], filename: str = "evaluation_results.json") -> None:
    """
    Save evaluation results to a file.
    
    Args:
        results: Results to save
        filename: Filename to save to
    """
    # Convert numpy values to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    # Convert results
    results_to_save = convert_numpy(results)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Save results
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    logger.info(f"Results saved to {filename}")


def load_results(filename: str = "evaluation_results.json") -> Dict[str, Any]:
    """
    Load evaluation results from a file.
    
    Args:
        filename: Filename to load from
        
    Returns:
        Dict[str, Any]: Loaded results
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Results file {filename} not found")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {filename}")
        return {}


def timestamp_filename(base_filename: str) -> str:
    """
    Add a timestamp to a filename.
    
    Args:
        base_filename: Base filename
        
    Returns:
        str: Filename with timestamp
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name, ext = os.path.splitext(base_filename)
    return f"{name}_{timestamp}{ext}"


class Timer:
    """
    Simple timer class for measuring execution time.
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize the timer.
        
        Args:
            name: Name for the timer
        """
        self.name = name
        self.start_time = None
        self.total_time = 0
    
    def __enter__(self):
        """Start the timer when entering a context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop the timer when exiting a context."""
        self.total_time = time.time() - self.start_time
        if self.name:
            logger.info(f"{self.name} completed in {self.total_time:.2f}s")