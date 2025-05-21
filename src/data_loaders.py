"""
Data loading utilities for various datasets used in retrieval and generation tasks.
This module provides functions to load and process data from multiple sources.
"""

import json
import uuid
import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser, SentenceWindowNodeParser
from datasets import load_dataset
from typing import List, Dict, Tuple, Any, Optional


def filter_large_nodes(nodes, max_length=8000):
    """
    Filters out nodes with text length greater than max_length.
    
    Args:
        nodes (list): List of node objects.
        max_length (int): Maximum allowed length for 'window' and 'text'.

    Returns:
        list: Filtered list of nodes.
    """
    filtered_nodes = []
    for node in nodes:
        text_length = len(node.text)
        window_length = len(node.metadata.get('window', ''))
        
        if text_length <= max_length and window_length <= max_length:
            filtered_nodes.append(node)
            
    return filtered_nodes


def load_hotpot_data(file_path: str, data_volume: int, mode: str = "qa"):
    """
    Load data from HotpotQA dataset.
    
    Args:
        file_path (str): Path to the HotpotQA data file.
        data_volume (int): Number of examples to load.
        mode (str): Mode for loading data - "qa" for question-answering or "retrieval" for retrieval evaluation.
        
    Returns:
        Tuple containing loaded documents and additional data depending on the mode.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    documents = []
    
    if mode == "qa":
        questions = []
        qa_pairs = {}
        
        for item in raw_data[:data_volume]:
            question_id = item['_id']
            questions.append(question_id)
            
            qa_pairs[question_id] = {
                'question': item['question'],
                'answer': item.get('answer'),
            }
            
            for title, sentences in item['context']:
                full_text = "".join(sentences)
                doc = Document(text=full_text)
                documents.append(doc)
                
        return documents, questions, qa_pairs
    
    elif mode == "geor":
        questions = []
        qa_pairs = {}
        
        for item in raw_data[:data_volume]:
            expected_ids = []
            question_id = item['_id']
            questions.append(question_id)
            
            for title, sentences in item['context']:
                sent_id = 0
                for sentence in sentences:
                    full_text = "".join(sentence)
                    doc_id = f"{item['_id']}_{title}_{sent_id}"
                    doc = Document(text=full_text, doc_id=doc_id)
                    sent_id += 1
                    documents.append(doc)
            
            for title, fact_idx in item["supporting_facts"]:
                doc_id = f"{item['_id']}_{title}_{fact_idx}"
                expected_ids.append(doc_id)
                
            qa_pairs[question_id] = {
                'question': item['question'],
                'answer': item.get('answer'),
                "expected_ids": expected_ids,
            }
            
        return documents, questions, qa_pairs
    
    elif mode == "retrieval":
        retrieval_eval = []
        
        for item in raw_data[:data_volume]:
            expected_ids = []
            
            for title, sentences in item['context']:
                sent_id = 0
                for sentence in sentences:
                    full_text = "".join(sentence)
                    doc_id = f"{item['_id']}_{title}_{sent_id}"
                    doc = Document(text=full_text, doc_id=doc_id)
                    sent_id += 1
                    documents.append(doc)
            
            for title, fact_idx in item["supporting_facts"]:
                doc_id = f"{item['_id']}_{title}_{fact_idx}"
                expected_ids.append(doc_id)
                
            retrieval_eval.append({
                "question_id": item['_id'],
                "query": item['question'],
                "expected_ids": expected_ids,
                "answer": item.get('answer')
            })
            
        return documents, retrieval_eval
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_musique_data(file_path: str, data_volume: int, mode: str = "qa"):
    """
    Load data from MuSiQue dataset.
    
    Args:
        file_path (str): Path to the MuSiQue data file.
        data_volume (int): Number of examples to load.
        mode (str): Mode for loading data - "qa" for question-answering or "retrieval" for retrieval evaluation.
        
    Returns:
        Tuple containing loaded documents and additional data depending on the mode.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = []
        for line in f:
            raw_data.append(json.loads(line.strip()))
            
    documents = []
    
    if mode == "qa":
        questions = []
        qa_pairs = {}
        
        for item in raw_data[1000:1000+data_volume]:
            question_id = item['id']
            questions.append(question_id)
            
            expected_ids = []
            for context in item['paragraphs']:
                full_text = "".join(context['paragraph_text'])
                doc_id = f"{item['id']}_{context['idx']}"
                doc = Document(text=full_text, doc_id=doc_id)
                documents.append(doc)
                if context["is_supporting"]:
                    expected_ids.append(doc_id)
                    
            qa_pairs[question_id] = {
                'question': item['question'],
                'answer': item.get('answer'),
                'expected_ids': expected_ids,
            }
            
        return documents, questions, qa_pairs
    
    elif mode == "retrieval":
        retrieval_eval = []
        
        for item in raw_data[:data_volume]:
            expected_ids = []
            
            for context in item['paragraphs']:
                full_text = "".join(context['paragraph_text'])
                doc_id = f"{item['id']}_{context['idx']}"
                doc = Document(text=full_text, doc_id=doc_id)
                documents.append(doc)
                if context["is_supporting"]:
                    expected_ids.append(doc_id)
                    
            retrieval_eval.append({
                "question_id": item['id'],
                "query": item['question'],
                "expected_ids": expected_ids,
                "answer": item.get('answer')
            })
            
        return documents, retrieval_eval
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_locomo_data(file_path: str, data_volume: int, mode: str = "qa"):
    """
    Load data from Locomo dataset.
    
    Args:
        file_path (str): Path to the Locomo data file.
        data_volume (int): Number of examples to load.
        mode (str): Mode for loading data - "qa" for question-answering or "retrieval" for retrieval evaluation.
        
    Returns:
        Tuple containing loaded documents and additional data depending on the mode.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    documents = []
    
    if mode == "qa":
        questions = []
        qa_pairs = {}
        
        for data in raw_data[:1]:
            sample_id = data['sample_id']
            session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() 
                           if 'session' in k and 'date_time' not in k]
            
            for item in data['qa']:
                # Skip adversarial answers
                if item.get('category') == 5:
                    continue
                    
                answer = item.get('answer') or item.get('adversarial_answer')
                # Skip answers with quotes
                if isinstance(answer, str) and ('"' in answer or "'" in answer):
                    continue
                    
                evidence_list = item.get('evidence', [])
                # Skip questions without evidence
                if not evidence_list:
                    continue
                    
                question_id = f"q_{uuid.uuid4().hex[:8]}"
                questions.append(question_id)
                expected_ids = [evidence for evidence in item['evidence']]
                
                qa_pairs[question_id] = {
                    'question': item['question'],
                    'answer': answer,
                    "category": item.get('category'),
                    'expected_ids': expected_ids,
                }
                
            for i in range(min(session_nums), max(session_nums) + 1):
                session = data['conversation'][f'session_{i}']
                date_time = data['conversation'][f'session_{i}_date_time']
                
                for dialog in session:
                    conv = ''
                    conv += f'At{date_time}, '
                    conv += f"{dialog['speaker']} said, \"{dialog['text']}\""
                    if 'blip_caption' in dialog:
                        conv += f" and shared {dialog['blip_caption']}."
                    conv += '\n'
                    doc_id = f"{dialog['dia_id']}"
                    doc = Document(text=conv, doc_id=doc_id)
                    documents.append(doc)
                    
        return documents, questions[:data_volume], qa_pairs
    
    elif mode == "retrieval":
        retrieval_eval = []
        
        for data in raw_data[:1]:
            sample_id = data['sample_id']
            session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() 
                           if 'session' in k and 'date_time' not in k]
            
            for item in data['qa']:
                question_id = f"q_{uuid.uuid4().hex[:8]}"
                evidence_list = item.get('evidence', [])
                
                # Skip questions without evidence
                if not evidence_list:
                    continue
                    
                expected_ids = [evidence for evidence in item['evidence']]
                
                converted_item = {
                    "question_id": question_id,
                    "query": item['question'],
                    "expected_ids": expected_ids,
                    "answer": item.get('answer') or item.get('adversarial_answer'),
                    "category": item.get('category'),
                }
                retrieval_eval.append(converted_item)
                
            for i in range(min(session_nums), max(session_nums) + 1):
                session = data['conversation'][f'session_{i}']
                date_time = data['conversation'][f'session_{i}_date_time']
                
                for dialog in session:
                    conv = ''
                    conv += f'At{date_time}, '
                    conv += f"{dialog['speaker']} said, \"{dialog['text']}\""
                    if 'blip_caption' in dialog:
                        conv += f" and shared {dialog['blip_caption']}."
                    conv += '\n'
                    doc_id = f"{dialog['dia_id']}"
                    doc = Document(text=conv, doc_id=doc_id)
                    documents.append(doc)
                    
        return documents, retrieval_eval
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_arxiv_data(data_path: str, data_volume: int, mode: str = "qa"):
    """
    Load data from the AI-ArXiv dataset with benchmark questions.
    
    Args:
        data_path (str): Path to the benchmark questions file.
        data_volume (int): Number of questions to load.
        
    Returns:
        Tuple containing loaded documents, questions, and QA pairs.
    """
    if mode == "qa":
        dataset = load_dataset("jamescalam/ai-arxiv")
        df = pd.DataFrame(dataset['train'])
        
        # Specify the titles of the required papers
        required_paper_titles = [
            'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
            'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter',
            'HellaSwag: Can a Machine Really Finish Your Sentence?',
            'LLaMA: Open and Efficient Foundation Language Models',
            'Measuring Massive Multitask Language Understanding',
            'CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks',
            'Task2Vec: Task Embedding for Meta-Learning',
            'GLM-130B: An Open Bilingual Pre-trained Model',
            'SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems',
            "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
            "PAL: Program-aided Language Models",
            "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
            "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"
        ]
        
        # Filter the DataFrame to include only the required papers
        required_papers = df[df['title'].isin(required_paper_titles)]
        
        # Exclude the already selected papers and randomly sample additional papers
        remaining_papers = df[~df['title'].isin(required_paper_titles)].sample(n=40, random_state=123)
        
        # Concatenate the two DataFrames
        final_df = pd.concat([required_papers, remaining_papers], ignore_index=True)
        
        documents = [Document(text=content) for content in final_df['content']]
        
        questions = []
        qa_pairs = {}
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get questions and answers lists
        queries = data.get('questions', [])
        answers = data.get('ground_truths', [])
        
        # Create QA pairs
        for i, (query, answer) in enumerate(zip(queries, answers), 1):
            qa_pairs[i] = {
                'question': query,
                'answer': answer
            }
            questions.append(i)
        
    return documents, questions[:data_volume], qa_pairs


def load_data(dataset_name: str, data_path: str, data_volume: int, mode: str = "qa"):
    """
    Generic data loading function that routes to the appropriate dataset loader.
    
    Args:
        dataset_name (str): Name of the dataset to load.
        data_path (str): Path to the data file.
        data_volume (int): Number of examples to load.
        mode (str): Mode for loading data.
        
    Returns:
        Tuple containing loaded data.
    """
    dataset_loaders = {
        "hotpotqa": load_hotpot_data,
        "musique": load_musique_data,
        "locomo": load_locomo_data,
        "arxiv": load_arxiv_data
    }
    
    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    # For arxiv dataset, mode is always "qa"
    if dataset_name == "arxiv" and mode != "qa":
        mode = "qa"
    
    return dataset_loaders[dataset_name](data_path, data_volume, mode)