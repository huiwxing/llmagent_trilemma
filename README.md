# LLM Agent Energy

LLM Agent Energy is the experimental implementation for the paper ["ADDRESSING THE SUSTAINABLE AI TRILEMMA: A CASE STUDY ON LLM AGENTS AND RAG"](https://arxiv.org/abs/2501.08262). This comprehensive framework focuses on evaluating and optimizing energy efficiency in Retrieval-Augmented Generation (RAG) systems, addressing the tension between AI capability, digital equity, and environmental sustainability.

## Paper Abstract

Large language models (LLMs) have demonstrated significant capabilities, but their widespread deployment and more advanced applications raise critical sustainability challenges, particularly in inference energy consumption. We propose the concept of the Sustainable AI Trilemma, highlighting the tensions between AI capability, digital equity, and environmental sustainability. Through a systematic case study of LLM agents and retrieval-augmented generation (RAG), we analyze the energy costs embedded in memory module designs and introduce novel metrics to quantify the trade-offs between energy consumption and system performance. Our experimental results reveal significant energy inefficiencies in current memory-augmented frameworks and demonstrate that resource-constrained environments face disproportionate efficiency penalties. Our findings challenge the prevailing LLM-centric paradigm in agent design and provide practical insights for developing more sustainable AI systems.

## Core Features

- **Energy Modeling**: Build and fit LLM energy consumption models to quantify inference costs
- **Multi-method Retrieval**: Support for various indexing and retrieval methods (vector stores, knowledge graphs, keyword tables, document summary index)
- **Retrieval Optimization**: Query classification, optimization, reranking, and compression
- **Comprehensive Evaluation**: Assessment of retrieval quality, generation quality, and energy efficiency
- **GEOR Metric**: A novel Generative Energy Optimization Ratio for measuring RAG energy efficiency

## Installation Requirements

### Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Environment Configuration

Create an `env_config.txt` file in the project root with your API keys:

```
HF_TOKEN=your_huggingface_token (for HuggingFace LLM Models, default)
SATURN_TOKEN=your_saturn_token (for NVIDIA NIM LLM Models)
OPENAI_API_KEY=your_openai_api_key (for Evaluators)
```

## Quick Start

The project provides a convenient startup script `quick_start.sh` to run predefined example tasks:

```bash
chmod +x quick_start.sh
./quick_start.sh
```

After running the script, select a task number:

1. **Fit Energy Model (Llama-3-8B)**: Measure and fit the energy consumption model to quantify the relationship between token counts and energy use
2. **Retrieval Evaluation (HotpotQA, vector index)**: Evaluate vector-based retrieval efficiency and quality on HotpotQA
3. **Retrieval Evaluation (MuSiQue, KnowledgeGraphIndex)**: Test knowledge graph-based indexing on MuSiQue dataset
4. **Retrieval Evaluation (MuSiQue, KeywordTableIndex)**: Evaluate keyword-based retrieval on MuSiQue
5. **Retrieval Evaluation (MuSiQue, DocumentSummaryIndex)**: Assess document summary-based retrieval on MuSiQue
6. **Generation Evaluation (Llama-3-8B)**: Measure generation quality and energy efficiency with Llama-3-8B
7. **Generation Evaluation (Llama-3-70B)**: Evaluate larger model generation performance and energy use
8. **GEOR Evaluation (HotpotQA)**: Calculate Generative Energy Optimization Ratio to assess overall RAG system efficiency, this task can only be run after task 1.
9. **Custom Configuration (All Options)**: Configure a custom evaluation setup
10. Exit

For more detailed analysis and research findings, please refer to the [full paper](https://arxiv.org/abs/2501.08262).

## Detailed Usage

### Running with Python

You can directly run `main.py` with appropriate parameters:

```bash
python src/main.py --task [task_type] --model [model_name] --dataset [dataset_name] [additional_options]
```

### Key Parameters

- `--task`: Task type, options: `fit_energy`, `retrieval`, `generation`, `geor`
- `--llm-mode`: LLM mode, options: `hf` (HuggingFace) or `nim` (NVIDIA NIM)
- `--model`: Model name/path, default: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `--dataset`: Dataset, options: `hotpotqa`, `musique`, `locomo`, `arxiv`
- `--data-volume`: Number of samples to use
- `--index-method`: Indexing method, options: `vector`, `KnowledgeGraphIndex`, `KeywordTableIndex`, `DocumentSummaryIndex`

### Retrieval Optimization Parameters

- `--search-k`: Number of documents to retrieve initially
- `--top-k`: Number of documents to return after reranking
- `--disable-classification`: Disable retrieval classification
- `--disable-query-optimization`: Disable query optimization
- `--disable-rerank`: Disable reranking
- `--disable-compress`: Disable compression

### Quantization Parameters

- `--use-4bit`: Use 4-bit quantization
- `--use-8bit`: Use 8-bit quantization

### Example Commands

Fit energy model:
```bash
python src/main.py --task fit_energy --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --data-volume 50
```

Evaluate retrieval performance:
```bash
python src/main.py --task retrieval --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --data-volume 50 --index-method vector --force-rebuild
```

Evaluate generation performance:
```bash
python src/main.py --task generation --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --data-volume 50
```

Evaluate GEOR metric:
```bash
python src/main.py --task geor --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --data-volume 50
```

## Project Structure

Key files:

- `main.py`: Entry point, parses command-line arguments and executes tasks
- `energy_modeling.py`: Energy consumption modeling code
- `data_loaders.py`: Data loading functionality
- `retrieval.py`: Joint retrieval system implementation
- `evaluation.py`: Evaluation framework
- `llm_interface.py`: Unified LLM interface implementation
- `utils.py`: Utility functions
- `requirements.txt`: Project dependencies
- `quick_start.sh`: Quick start script
- `env_config.txt`: Environment variable configuration

## Test Datasets

1. **hotpotqa**: Multi-hop reasoning QA dataset
2. **musique**: Multi-step question answering dataset
3. **locomo**: Multimodal conversation dataset
4. **arxiv**: AI papers dataset

## Citation

If you use this project in your research, please cite our paper:

```
@misc{wu2025addressingsustainableaitrilemma,
      title={Addressing the sustainable AI trilemma: a case study on LLM agents and RAG}, 
      author={Hui Wu and Xiaoyang Wang and Zhong Fan},
      year={2025},
      eprint={2501.08262},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2501.08262}, 
}
```
