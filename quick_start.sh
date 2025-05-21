#!/bin/bash

# RAG Energy - Integrated Launch Script

# Set colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Load environment variables if config file exists
if [ -f env_config.txt ]; then
  echo "Loading environment variables..."
  # Load environment variables from env_config.txt
  export $(grep -v '^#' env_config.txt | xargs)
else
  echo "Warning: env_config.txt file not found."
fi

# Initialize default values for additional parameters
EMBED_MODEL="BAAI/bge-base-en-v1.5"
USE_4BIT=""
USE_8BIT=""
INDEX_METHOD="vector"
SEARCH_K=5
TOP_K=5
FORCE_REBUILD="--force-rebuild"
DISABLE_CLASSIFICATION=""
DISABLE_QUERY_OPTIMIZATION=""
DISABLE_RERANK=""
DISABLE_COMPRESS=""
OUTPUT_DIR="outputs"
OUTPUT_FILE_NAME=""
PERSIST_DIR="outputs/indices"
ADDITIONAL_ARGS=""

# Prepare token arguments
if [ ! -z "$HF_TOKEN" ]; then
  HF_TOKEN_ARG="--hf-token $HF_TOKEN"
else
  HF_TOKEN_ARG=""
fi

if [ ! -z "$SATURN_TOKEN" ]; then
  SATURN_TOKEN_ARG="--saturn-token $SATURN_TOKEN"
else
  SATURN_TOKEN_ARG=""
fi

if [ ! -z "$OPENAI_API_KEY" ]; then
  OPENAI_API_KEY_ARG="--openai-key $OPENAI_API_KEY"
else
  OPENAI_API_KEY_ARG=""
fi

# Function to configure all options (replaces both custom task and advanced options)
configure_all_options() {
  echo -e "\n${BLUE}======== Custom Configuration ========${NC}"
  
  # Task selection
  echo -e "${YELLOW}Select task type:${NC}"
  echo "1) fit_energy"
  echo "2) retrieval"
  echo "3) generation"
  echo "4) geor"
  
  local task_selection
  while true; do
    read -p "Enter selection (1-4): " task_selection
    case $task_selection in
      1) TASK="fit_energy"; break;;
      2) TASK="retrieval"; break;;
      3) TASK="generation"; break;;
      4) TASK="geor"; break;;
      *) echo "Invalid selection. Please try again.";;
    esac
  done
  echo "Selected task: $TASK"
  
  # Model selection
  echo -e "\n${YELLOW}Select model:${NC}"
  echo "1) meta-llama/Meta-Llama-3.1-8B-Instruct"
  echo "2) meta-llama/Meta-Llama-3.1-70B-Instruct"
  echo "3) other (custom entry)"
  
  local model_selection
  while true; do
    read -p "Enter selection (1-3): " model_selection
    case $model_selection in
      1) MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"; break;;
      2) MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"; break;;
      3) read -p "Enter custom model name/path: " MODEL; break;;
      *) echo "Invalid selection. Please try again.";;
    esac
  done
  echo "Selected model: $MODEL"
  
# Dataset selection based on task type
  echo -e "\n${YELLOW}Select dataset:${NC}"
  
  if [ "$TASK" = "retrieval" ]; then
    # For retrieval task, arxiv is not applicable
    echo "1) hotpotqa"
    echo "2) musique"
    echo "3) locomo"
    
    local dataset_selection
    while true; do
      read -p "Enter selection (1-3): " dataset_selection
      case $dataset_selection in
        1) DATASET="hotpotqa"; break;;
        2) DATASET="musique"; break;;
        3) DATASET="locomo"; break;;
        *) echo "Invalid selection. Please try again.";;
      esac
    done
  else
    # For non-retrieval tasks, all datasets are available
    echo "1) hotpotqa"
    echo "2) musique"
    echo "3) locomo"
    echo "4) arxiv"
    
    local dataset_selection
    while true; do
      read -p "Enter selection (1-4): " dataset_selection
      case $dataset_selection in
        1) DATASET="hotpotqa"; break;;
        2) DATASET="musique"; break;;
        3) DATASET="locomo"; break;;
        4) DATASET="arxiv"; break;;
        *) echo "Invalid selection. Please try again.";;
      esac
    done
  fi
  
  echo "Selected dataset: $DATASET"
  
  # Data volume
  read -p "Data volume [default: 50]: " DATA_VOLUME
  DATA_VOLUME=${DATA_VOLUME:-50}
  
  # LLM mode
  echo -e "\n${YELLOW}Select LLM mode:${NC}"
  echo "1) hf (HuggingFace) [default]"
  echo "2) nim (NVIDIA NIM)"
  
  local llm_selection
  read -p "Enter selection (1-2) [default: 1]: " llm_selection
  llm_selection=${llm_selection:-1}
  
  case $llm_selection in
    1) LLM_MODE="hf";;
    2) LLM_MODE="nim";;
    *) echo "Invalid selection. Using default: hf"; LLM_MODE="hf";;
  esac
  echo "Selected LLM mode: $LLM_MODE"
  
  echo -e "\n${YELLOW}Advanced Options Configuration${NC}"
  echo "Press Enter to keep default values."
  
  # Index method options based on task
  if [ "$TASK" = "retrieval" ] || [ "$TASK" = "generation" ] || [ "$TASK" = "geor" ]; then
    echo -e "\n${YELLOW}Select index method:${NC}"
    echo "1) vector"
    echo "2) KnowledgeGraphIndex"
    echo "3) KeywordTableIndex"
    echo "4) DocumentSummaryIndex"
    
    local index_selection
    while true; do
      read -p "Enter selection (1-4): " index_selection
      case $index_selection in
        1) INDEX_METHOD="vector"; break;;
        2) INDEX_METHOD="KnowledgeGraphIndex"; break;;
        3) INDEX_METHOD="KeywordTableIndex"; break;;
        4) INDEX_METHOD="DocumentSummaryIndex"; break;;
        *) echo "Invalid selection. Please try again.";;
      esac
    done
    echo "Selected index method: $INDEX_METHOD"
    
      read -p "Force rebuild index? (y/n) [default: y]: " input
      if [ "$input" = "n" ] || [ "$input" = "N" ]; then
        FORCE_REBUILD=""
      else
        FORCE_REBUILD="--force-rebuild"
      fi
  fi
  
  # Embedding model
  read -p "Embedding model [default: $EMBED_MODEL]: " input
  if [ ! -z "$input" ]; then
    EMBED_MODEL=$input
  fi
  
  # Quantization options
  read -p "Use 4-bit quantization? (y/n) [default: n]: " input
  if [ "$input" = "y" ] || [ "$input" = "Y" ]; then
    USE_4BIT="--use-4bit"
  fi
  
  read -p "Use 8-bit quantization? (y/n) [default: n]: " input
  if [ "$input" = "y" ] || [ "$input" = "Y" ]; then
    USE_8BIT="--use-8bit"
  fi
  
  # Retrieval parameters (if relevant)
  if [ "$TASK" = "retrieval" ] || [ "$TASK" = "generation" ] || [ "$TASK" = "geor" ]; then
    read -p "Search K (docs to retrieve initially) [default: $SEARCH_K]: " input
    if [ ! -z "$input" ]; then
      SEARCH_K=$input
    fi
    
    read -p "Top K (docs after reranking) [default: $TOP_K]: " input
    if [ ! -z "$input" ]; then
      TOP_K=$input
    fi
    
    # Feature toggles
    read -p "Disable classification? (y/n) [default: n]: " input
      if [ "$input" = "y" ] || [ "$input" = "Y" ]; then
        DISABLE_CLASSIFICATION="--disable-classification"
      fi
      
      read -p "Disable query optimization? (y/n) [default: n]: " input
      if [ "$input" = "y" ] || [ "$input" = "Y" ]; then
        DISABLE_QUERY_OPTIMIZATION="--disable-query-optimization"
      fi
      
      read -p "Disable reranking? (y/n) [default: n]: " input
      if [ "$input" = "y" ] || [ "$input" = "Y" ]; then
        DISABLE_RERANK="--disable-rerank"
      fi
      
      read -p "Disable compression? (y/n) [default: n]: " input
      if [ "$input" = "y" ] || [ "$input" = "Y" ]; then
        DISABLE_COMPRESS="--disable-compress"
      fi
  fi
  
  # Output configuration
  read -p "Output directory [default: $OUTPUT_DIR]: " input
  if [ ! -z "$input" ]; then
    OUTPUT_DIR=$input
  fi
  
  read -p "Output file name [default: auto-generated]: " input
  if [ ! -z "$input" ]; then
    OUTPUT_FILE_NAME=$input
  fi
  
  read -p "Persist directory for indices [default: $PERSIST_DIR]: " input
  if [ ! -z "$input" ]; then
    PERSIST_DIR=$input
  fi
  
  # Any additional arguments
  read -p "Any additional arguments (format: --arg1 value1 --arg2 value2): " ADDITIONAL_ARGS
  
  echo -e "${GREEN}All options configured successfully!${NC}"
}

# Function to run the task
run_task() {
  local TASK=$1
  local MODEL=$2
  local DATASET=$3
  local DATA_VOLUME=$4
  local LLM_MODE=$5

  # Generate dynamic output file name if not specified
  if [ -z "$OUTPUT_FILE_NAME" ]; then
    # Extract model's short name (remove path prefix)
    MODEL_SHORT=$(echo $MODEL | sed 's|.*/||')
    # Create timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    # Generate filename
    OUTPUT_FILE_NAME="${TASK}_${DATASET}_${MODEL_SHORT}_${TIMESTAMP}.json"
  fi
  
  # Ensure output directory exists
  mkdir -p "$OUTPUT_DIR"
  
  # Complete output file path
  FULL_OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_FILE_NAME}"
  
  # Display configuration information
  echo "Starting, configuration is as follows:"
  echo "- Task: $TASK"
  echo "- Model: $MODEL"
  echo "- Dataset: $DATASET"
  echo "- Data volume: $DATA_VOLUME"
  echo "- LLM mode: $LLM_MODE"
  
  # Show advanced options if any are set
  if [ ! -z "$USE_4BIT" ] || [ ! -z "$USE_8BIT" ] || [ ! -z "$FORCE_REBUILD" ] || [ ! -z "$DISABLE_CLASSIFICATION" ] || [ ! -z "$DISABLE_QUERY_OPTIMIZATION" ] || [ ! -z "$DISABLE_RERANK" ] || [ ! -z "$DISABLE_COMPRESS" ]; then
    echo -e "${YELLOW}Advanced options:${NC}"
    [ ! -z "$USE_4BIT" ] && echo "- 4-bit quantization enabled"
    [ ! -z "$USE_8BIT" ] && echo "- 8-bit quantization enabled"
    [ ! -z "$FORCE_REBUILD" ] && echo "- Force rebuilding index enabled"
    [ ! -z "$DISABLE_CLASSIFICATION" ] && echo "- Classification disabled"
    [ ! -z "$DISABLE_QUERY_OPTIMIZATION" ] && echo "- Query optimization disabled"
    [ ! -z "$DISABLE_RERANK" ] && echo "- Reranking disabled"
    [ ! -z "$DISABLE_COMPRESS" ] && echo "- Compression disabled"
    echo "- Embed model: $EMBED_MODEL"
    echo "- Index method: $INDEX_METHOD"
    echo "- Search K: $SEARCH_K"
    echo "- Top K: $TOP_K"
    echo "- Output file: $FULL_OUTPUT_PATH"
    echo "- Persist directory: $PERSIST_DIR"
    [ ! -z "$ADDITIONAL_ARGS" ] && echo "- Additional args: $ADDITIONAL_ARGS"
  fi

  # Execute the Python script with all parameters
  python src/main.py \
    --task $TASK \
    --model $MODEL \
    --dataset $DATASET \
    --data-volume $DATA_VOLUME \
    --llm-mode $LLM_MODE \
    $HF_TOKEN_ARG \
    $SATURN_TOKEN_ARG \
    $OPENAI_API_KEY_ARG \
    --embed-model "$EMBED_MODEL" \
    --index-method $INDEX_METHOD \
    --search-k $SEARCH_K \
    --top-k $TOP_K \
    --output-file "$FULL_OUTPUT_PATH" \
    --persist-dir "$PERSIST_DIR" \
    $USE_4BIT \
    $USE_8BIT \
    $FORCE_REBUILD \
    $DISABLE_CLASSIFICATION \
    $DISABLE_QUERY_OPTIMIZATION \
    $DISABLE_RERANK \
    $DISABLE_COMPRESS \
    $ADDITIONAL_ARGS

  echo -e "${GREEN}Task completed. Results saved to: $FULL_OUTPUT_PATH${NC}"
}

# Display main menu
echo -e "${BLUE}=================================${NC}"
echo -e "${GREEN}RAG Energy - Example Tasks${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""
echo "Please select a task to run:"
echo "1) Fit Energy Model (Llama-3-8B)"
echo "2) Retrieval Evaluation (HotpotQA, vector index)"
echo "3) Retrieval Evaluation (MuSiQue, KnowledgeGraphIndex)"
echo "4) Retrieval Evaluation (MuSiQue, KeywordTableIndex)"
echo "5) Retrieval Evaluation (MuSiQue, DocumentSummaryIndex)"
echo "6) Generation Evaluation (Llama-3-8B)"
echo "7) Generation Evaluation (Llama-3-70B)"
echo "8) GEOR Evaluation (HotpotQA)"
echo "9) Custom Configuration (All Options)"
echo "10) Exit"
echo ""
read -p "Please enter option (1-10): " choice

# Handle custom configuration
if [ "$choice" = "9" ]; then
  configure_all_options
  echo "Starting custom configuration task..."
  run_task $TASK $MODEL $DATASET $DATA_VOLUME $LLM_MODE
  exit 0
fi

# Handle exit
if [ "$choice" = "10" ]; then
  echo "Exiting program"
  exit 0
fi

# Handle preset configurations
read -p "Select LLM mode (hf/nim) [default: hf]: " llm_mode
llm_mode=${llm_mode:-hf}

case $choice in
  1)
    echo "Starting energy model fitting..."
    FORCE_REBUILD="--force-rebuild"
    INDEX_METHOD="vector"
    run_task fit_energy meta-llama/Meta-Llama-3.1-8B-Instruct hotpotqa 50 $llm_mode
    ;;
  2)
    echo "Starting HotpotQA retrieval evaluation..."
    FORCE_REBUILD="--force-rebuild"
    INDEX_METHOD="vector"
    run_task retrieval meta-llama/Meta-Llama-3.1-8B-Instruct hotpotqa 50 $llm_mode
    ;;
  3)
    echo "Starting MuSiQue retrieval evaluation with KnowledgeGraphIndex..."
    FORCE_REBUILD="--force-rebuild"
    INDEX_METHOD="KnowledgeGraphIndex"
    run_task retrieval meta-llama/Meta-Llama-3.1-8B-Instruct musique 10 $llm_mode
    ;;
  4)
    echo "Starting MuSiQue retrieval evaluation with KeywordTableIndex..."
    FORCE_REBUILD="--force-rebuild"
    INDEX_METHOD="KeywordTableIndex"
    run_task retrieval meta-llama/Meta-Llama-3.1-8B-Instruct musique 10 $llm_mode
    ;;
  5)
    echo "Starting MuSiQue retrieval evaluation with DocumentSummaryIndex..."
    FORCE_REBUILD="--force-rebuild"
    INDEX_METHOD="DocumentSummaryIndex"
    run_task retrieval meta-llama/Meta-Llama-3.1-8B-Instruct musique 10 $llm_mode
    ;;
  6)
    echo "Starting generation evaluation (Llama-3-8B)..."
    FORCE_REBUILD="--force-rebuild"
    INDEX_METHOD="vector"
    run_task generation meta-llama/Meta-Llama-3.1-8B-Instruct hotpotqa 50 $llm_mode
    ;;
  7)
    echo "Starting generation evaluation (Llama-3-70B)..."
    FORCE_REBUILD="--force-rebuild"
    INDEX_METHOD="vector"
    run_task generation meta-llama/Meta-Llama-3.1-70B-Instruct hotpotqa 50 $llm_mode
    ;;
  8)
    echo "Starting GEOR evaluation..."
    FORCE_REBUILD="--force-rebuild"
    INDEX_METHOD="vector"
    run_task geor meta-llama/Meta-Llama-3.1-8B-Instruct hotpotqa 300 $llm_mode
    ;;
  *)
    echo "Invalid option, please run the script again and choose a number between 1-10."
    exit 1
    ;;
esac

echo "Script execution completed."