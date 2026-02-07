---
name: LLM Finetuning Pipeline
overview: Build a complete end-to-end instruction tuning pipeline for medium-sized models (1B-7B params) supporting multiple finetuning techniques (Full, LoRA, QLoRA, Prefix Tuning) with data preparation, training, evaluation, and deployment capabilities.
todos:
  - id: project-setup
    content: Initialize project structure, pyproject.toml, Makefile, Docker config
    status: pending
  - id: config-management
    content: Create Pydantic Settings for models, training, and monitoring
    status: pending
  - id: domain-models
    content: Implement model config and loader with quantization support
    status: pending
    dependencies:
      - config-management
  - id: training-strategies
    content: Build all finetuning strategies (Full, LoRA, QLoRA, Prefix)
    status: pending
    dependencies:
      - domain-models
  - id: evaluation-logic
    content: Implement metrics and evaluation orchestration
    status: pending
    dependencies:
      - domain-models
  - id: data-preparation
    content: Create data formatting and validation for instruction tuning
    status: pending
    dependencies:
      - config-management
  - id: training-service
    content: Build main training orchestration with callbacks
    status: pending
    dependencies:
      - training-strategies
      - data-preparation
  - id: evaluation-service
    content: Create benchmark runner and comparison tools
    status: pending
    dependencies:
      - evaluation-logic
  - id: inference-service
    content: Implement inference and adapter merging
    status: pending
    dependencies:
      - domain-models
  - id: infrastructure
    content: Build data loaders, model registry, monitoring, storage
    status: pending
    dependencies:
      - project-setup
  - id: testing
    content: Create comprehensive end-to-end test suite
    status: pending
    dependencies:
      - training-service
      - evaluation-service
      - inference-service
  - id: documentation
    content: Write README.md, Quickstart.md with examples
    status: pending
    dependencies:
      - testing
  - id: notebooks
    content: Create educational Jupyter notebooks for each technique
    status: pending
    dependencies:
      - testing
---

# LLM Finetuning Pipeline - Complete Implementation

## Project Structure

Following Clean Architecture principles, we'll create:

```javascript
llm_finetuning/
├── src/llm_finetuning/
│   ├── domain/              # Core business logic (never imports from app/infra)
│   │   ├── models/          # Model architectures & configs
│   │   ├── training/        # Training strategies (Full, LoRA, QLoRA, Prefix)
│   │   └── evaluation/      # Evaluation metrics & logic
│   ├── application/         # Use cases & orchestration (depends only on domain)
│   │   ├── data_preparation_service/
│   │   ├── training_service/
│   │   ├── evaluation_service/
│   │   └── inference_service/
│   └── infrastructure/      # External integrations (can depend on app/domain)
│       ├── api/             # API endpoints (if needed)
│       ├── db/              # Database integrations (if needed)
│       ├── llm_providers/   # LLM provider integrations
│       ├── data_loaders/    # Data loading infrastructure
│       ├── model_registry/  # Model registry (HF Hub, local storage)
│       ├── monitoring/      # WandB/TensorBoard integration
│       └── storage/         # Checkpoints & artifacts
├── data/                    # Datasets
├── notebooks/               # Experimentation notebooks
├── tests/                   # Testing layer
│   ├── conftest.py          # Pytest fixtures
│   └── test_end_to_end.py   # Single comprehensive E2E test file
├── .github/                 # CI/CD workflows
├── docker-compose.yaml
├── Dockerfile
├── Makefile
├── pyproject.toml           # All dependencies (no requirements.txt)
├── .env.example             # Environment variable template
├── README.md                # Main documentation
└── Quickstart.md            # Step-by-step tutorial
```



## Implementation Flow

### Phase 1: Foundation Setup

**1.1 Project Initialization**

- Create project structure following user's Clean Architecture template (exact folder hierarchy)
- Use `uv` for project initialization and dependency management
- Setup `pyproject.toml` with pinned exact dependency versions:
- `torch`, `transformers`, `peft`, `bitsandbytes`
- `datasets`, `accelerate`, `wandb`
- `loguru`, `pydantic-settings`
- Development tools: `ruff`, `pytest`, `pre-commit`
- Create comprehensive `Makefile` for automation:
- `setup`: Create environment using uv, install dependencies
- `lint`: Run ruff for linting and formatting
- `test`: Run end-to-end tests
- `train-full`, `train-lora`, `train-qlora`: Training commands
- `evaluate`: Run evaluation suite
- `inference`: Interactive inference
- Setup `.env.example` for configuration management (all configs via Pydantic Settings)
- Create `tests/conftest.py` for pytest fixtures
- Docker configuration (Dockerfile + docker-compose.yaml) for reproducible environment
- Create `.github` folder structure for CI/CD workflows

**1.2 Configuration Management**

- Centralize all configuration in `.env` file (never hardcode)
- Pydantic Settings classes for:
- Model configuration (size, architecture, quantization)
- Training hyperparameters (batch size, learning rate, epochs)
- Finetuning technique selection (Full/LoRA/QLoRA/Prefix)
- Data paths and preprocessing options
- Monitoring & logging configuration (WandB API keys, etc.)
- All environment variables loaded and validated via Pydantic Settings
- Create `.env.example` with all required variables documented

### Phase 2: Domain Layer - Core Logic

**2.1 Model Configuration (`src/llm_finetuning/domain/models/`)**

- `model_config.py`: Model specifications and supported architectures
- Support for: LLaMA-2 (1B-7B), Mistral 7B, Phi-2, GPT-2 variants
- Quantization configs (4-bit, 8-bit)
- `model_loader.py`: Unified model loading with proper device mapping

**2.2 Training Strategies (`src/llm_finetuning/domain/training/`)**

- `base_strategy.py`: Abstract training strategy interface
- `full_finetuning.py`: Standard full-parameter training
- `lora_strategy.py`: LoRA implementation using PEFT
- Configurable rank, alpha, target modules
- `qlora_strategy.py`: 4-bit quantized LoRA
- BitsAndBytes integration for memory efficiency
- `prefix_tuning.py`: Prefix tuning strategy
- `training_config.py`: Unified training configuration

**2.3 Evaluation Logic (`src/llm_finetuning/domain/evaluation/`)**

- `metrics.py`: Loss, perplexity, ROUGE, BLEU metrics
- `evaluator.py`: Evaluation orchestration
- `instruction_eval.py`: Instruction-following quality assessment

### Phase 3: Application Layer - Use Cases

**3.1 Data Preparation Service (`src/llm_finetuning/application/data_preparation_service/`)**

- `dataset_formatter.py`: Convert datasets to instruction format
- Templates: Alpaca, ShareGPT, custom formats
- Tokenization with proper padding/truncation
- `data_validator.py`: Quality checks and filtering
- Support for popular datasets:
- Alpaca, Dolly, OpenAssistant, custom JSON/JSONL
- All functions must have comprehensive docstrings (Google or NumPy style)
- Include: Purpose, Arguments (with types), Return Type

**3.2 Training Service (`src/llm_finetuning/application/training_service/`)**

- `trainer.py`: Main training orchestration
- Strategy pattern for switching between techniques
- Gradient accumulation, mixed precision (fp16/bf16)
- Checkpointing and recovery
- `callbacks.py`: Custom training callbacks
- Learning rate scheduling
- Early stopping
- Metric logging

**3.3 Evaluation Service (`src/llm_finetuning/application/evaluation_service/`)**

- `benchmark_runner.py`: Run evaluation on test sets
- `comparison.py`: Compare different finetuning techniques
- Generate evaluation reports with metrics

**3.4 Inference Service (`src/llm_finetuning/application/inference_service/`)**

- `predictor.py`: Load finetuned model and generate completions
- `batch_inference.py`: Batch processing support
- `merge_adapters.py`: Merge LoRA/QLoRA adapters into base model

### Phase 4: Infrastructure Layer

**4.1 LLM Providers (`src/llm_finetuning/infrastructure/llm_providers/`)**

- `huggingface_provider.py`: Hugging Face model loading and integration
- `model_loader.py`: Unified model loading interface
- Support for multiple model architectures and quantization

**4.2 Data Loaders (`src/llm_finetuning/infrastructure/data_loaders/`)**

- `instruction_loader.py`: Load and preprocess instruction datasets
- `collator.py`: Custom data collators for instruction tuning
- Efficient DataLoader with caching

**4.3 Model Registry (`src/llm_finetuning/infrastructure/model_registry/`)**

- `huggingface_hub.py`: Integration with HF Hub
- `local_storage.py`: Local model and checkpoint management
- Version tracking

**4.4 Monitoring (`src/llm_finetuning/infrastructure/monitoring/`)**

- `wandb_logger.py`: Weights & Biases integration (using loguru for internal logging)
- `tensorboard_logger.py`: TensorBoard support
- Real-time training metrics visualization
- All logging via loguru (DEBUG, INFO, WARNING, ERROR levels)

**4.5 Storage (`src/llm_finetuning/infrastructure/storage/`)**

- `checkpoint_manager.py`: Save/load checkpoints
- `artifact_manager.py`: Manage training artifacts
- Cloud storage integration (S3-compatible) if needed

### Phase 5: Testing & Documentation

**5.1 Testing**

- Create `tests/conftest.py` with pytest fixtures
- Single comprehensive test file: `tests/test_end_to_end.py`
- Test all finetuning techniques with tiny model
- Validate data preprocessing pipeline
- Check inference and adapter merging
- Verify metric computation
- Tests must remain isolated and not contain production logic

**5.2 Documentation**

- `README.md`: Must include all required sections:
- ✅ Title of the Project
- ✅ Objective / Problem Statement
- ✅ Summary of the Implementation Plan
- ✅ Input Data Description
- ✅ Tech Stack Used
- ✅ Output Format (with sample outputs or structures)
- ✅ Setup & Run Instructions (beginner-friendly)
- ✅ Description of Project Files and Folder Hierarchy
- Quick comparison table of finetuning techniques
- Hardware requirements per technique
- Example commands for each strategy
- `Quickstart.md`: Step-by-step tutorial
- Dataset preparation walkthrough
- Training your first model
- Evaluation and inference examples
- NO other .md files (only README.md and Quickstart.md in root)

**5.3 Example Notebooks**

- `notebooks/01_data_exploration.ipynb`: Dataset analysis
- `notebooks/02_full_finetuning.ipynb`: Full finetuning walkthrough
- `notebooks/03_lora_qlora_comparison.ipynb`: Compare techniques
- `notebooks/04_inference_demo.ipynb`: Use finetuned models

### Phase 6: Automation & Deployment

**6.1 Makefile Targets**

```makefile
setup          # Create environment using uv, install dependencies
lint           # Run ruff for linting and formatting
test           # Run end-to-end tests
prepare-data   # Download and prepare instruction dataset
train-full     # Full finetuning
train-lora     # LoRA finetuning
train-qlora    # QLoRA finetuning (most memory efficient)
evaluate       # Run evaluation suite
inference      # Interactive inference
clean          # Clean build artifacts and cache
```

**6.2 Pre-commit Hooks**

- Setup pre-commit hooks for linting, formatting, and style consistency
- Use ruff for code quality checks

**6.3 Docker Setup**

- Multi-stage Dockerfile with CUDA support
- docker-compose.yaml with GPU configuration
- Pre-configured for training and inference
- Optimized for low-latency execution

## Key Features

### Memory Optimization

- Gradient checkpointing for large models
- QLoRA with 4-bit quantization (fit 7B models on 16GB GPU)
- Flash Attention 2 support
- CPU offloading strategies

### Flexibility

- Easy switching between finetuning techniques via config
- Support for multiple model architectures
- Custom dataset integration
- Extensible evaluation metrics

### Production Ready

- Comprehensive logging with loguru (no print statements)
- Proper log levels: DEBUG, INFO, WARNING, ERROR
- Error handling and recovery
- Model versioning
- Deployment-ready inference API
- All code follows snake_case for functions/variables, PascalCase for classes
- Descriptive, explicit naming (no abbreviations or single-letter variables)

### Educational Value

- Clear code structure showing how each technique works
- Comparative analysis tools
- Documented best practices
- Resource usage profiling

## Technical Specifications

- **Supported Models**: LLaMA-2 (1B-7B), Mistral 7B, Phi-2, GPT-2 variants
- **Finetuning Techniques**: Full, LoRA, QLoRA, Prefix Tuning
- **Training Features**: Mixed precision, gradient accumulation, distributed training (DDP)
- **Evaluation**: Perplexity, ROUGE, BLEU, custom instruction metrics
- **Monitoring**: WandB, TensorBoard
- **Hardware**: Optimized for single GPU (16GB-40GB), multi-GPU support

## Dependencies Highlights

All dependencies in `pyproject.toml` with pinned exact versions for reproducibility:

```toml
torch = "2.0.0"  # Exact version pinned
transformers = "4.36.0"
peft = "0.8.0"
bitsandbytes = "0.42.0"
datasets = "2.16.0"
accelerate = "0.26.0"
loguru = "0.7.0"
pydantic-settings = "2.0.0"
wandb = "0.16.0"
ruff = "0.1.0"  # For linting
pytest = "7.4.0"  # For testing
```



## Architecture Principles

- **Domain Layer**: Never imports from Application or Infrastructure layers
- **Application Layer**: Depends only on Domain layer
- **Infrastructure Layer**: Can depend on Application and Domain layers
- **Modularity**: Each module performs one clear, well-defined purpose
- **Simplicity**: Start with simplest working version, improve iteratively
- **Performance**: Prioritize low-latency execution in all implementations
- **Testability**: All components designed to run and test independently

## Code Quality Standards

### Documentation

- **Docstrings**: Every function, class, and module must have docstrings
- **Format**: Google-style or NumPy-style docstrings
- **Required Fields**: Purpose, Arguments (with types and descriptions), Return Type
- **Example**:
  ```python
      def load_model(model_name: str, quantization: str = "none") -> PreTrainedModel:
          """
          Load a pre-trained model with optional quantization.
          
          Args:
              model_name: Name of the model to load (e.g., "meta-llama/Llama-2-7b-hf")
              quantization: Quantization type ("none", "4bit", "8bit")
          
          Returns:
              Loaded model instance ready for training or inference
          """
  ```




### Naming Conventions

- **Variables & Functions**: snake_case (e.g., `load_model`, `training_config`)
- **Classes**: PascalCase (e.g., `TrainingStrategy`, `ModelLoader`)
- **Descriptive Names**: Avoid abbreviations, single-letter variables, ambiguous names
- **Examples**:
- ✅ `model_configuration` ❌ `cfg` or `config`
- ✅ `learning_rate_scheduler` ❌ `lr_sched`
- ✅ `training_batch_size` ❌ `bs` or `batch`

### Logging Standards

- **Library**: Use loguru exclusively (no print statements)
- **Log Levels**: 
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for failures
- **Context**: Logs must provide meaningful context for debugging

### Code Organization

- **Reusable Logic**: Place in `utils/` folder with helper functions in `utils.py`