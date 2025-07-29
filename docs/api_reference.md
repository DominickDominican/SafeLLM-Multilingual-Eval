# API Reference - SafeLLM Multilingual Evaluation Framework

This document provides detailed API documentation for all public classes and functions in the SafeLLM Multilingual Evaluation Framework.

## Core Modules

### `safellm_eval.evaluator`

#### `MultilingualEvaluator`

Main evaluator class for running multilingual safety assessments.

```python
class MultilingualEvaluator:
    def __init__(self, config_path: Optional[str] = None)
```

**Parameters:**
- `config_path` (str, optional): Path to configuration file

**Methods:**

##### `load_prompts(file_path: str) -> List[Dict[str, Any]]`

Load prompts from JSONL file.

**Parameters:**
- `file_path` (str): Path to JSONL file containing prompts

**Returns:**
- List of prompt dictionaries with added `prompt_id` field

**Example:**
```python
evaluator = MultilingualEvaluator()
prompts = evaluator.load_prompts("datasets/test_prompts.jsonl")
```

##### `evaluate_dataset(dataset_path: str, models: Optional[List[Dict[str, str]]] = None) -> List[EvaluationResult]`

Evaluate entire dataset across multiple models.

**Parameters:**
- `dataset_path` (str): Path to dataset file
- `models` (List[Dict], optional): List of model configurations

**Returns:**
- List of `EvaluationResult` objects

**Example:**
```python
models = [{"name": "gpt-4", "provider": "openai"}]
results = evaluator.evaluate_dataset("test.jsonl", models)
```

##### `save_results(output_path: str, format: str = "jsonl")`

Save evaluation results to file.

**Parameters:**
- `output_path` (str): Output file path
- `format` (str): Output format ("jsonl", "csv", "json")

**Example:**
```python
evaluator.save_results("results.jsonl", format="jsonl")
```

##### `get_summary_stats() -> Dict[str, Any]`

Get summary statistics from evaluation results.

**Returns:**
- Dictionary containing summary statistics

**Example:**
```python
stats = evaluator.get_summary_stats()
print(f"Average safety score: {stats['average_safety_score']}")
```

#### `EvaluationResult`

Data class representing evaluation results.

```python
@dataclass
class EvaluationResult:
    prompt_id: str
    language: str
    domain: str
    prompt_type: str
    risk_level: str
    prompt: str
    model_response: str
    model: str
    provider: str
    safety_score: float
    risk_categories: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: float
    error: Optional[str] = None
```

### `safellm_eval.models`

#### `ModelClient`

Abstract base class for model clients.

```python
class ModelClient(ABC):
    def __init__(self, api_key: str, model_name: str)
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> ModelResponse
```

#### `OpenAIClient`

OpenAI API client implementation.

```python
class OpenAIClient(ModelClient):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4")
```

**Example:**
```python
client = OpenAIClient(api_key="sk-...", model_name="gpt-4")
response = client.generate_response("Hello, world!", temperature=0.7)
```

#### `AnthropicClient`

Anthropic API client implementation.

```python
class AnthropicClient(ModelClient):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-sonnet-20240229")
```

**Example:**
```python
client = AnthropicClient(api_key="sk-ant-...", model_name="claude-3-sonnet-20240229")
response = client.generate_response("Hello, world!", max_tokens=1000)
```

#### `MistralClient`

Mistral AI API client implementation.

```python
class MistralClient(ModelClient):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "mistral-medium")
```

#### `ModelFactory`

Factory class for creating model clients.

```python
class ModelFactory:
    @staticmethod
    def create_client(provider: str, model_name: str, api_key: Optional[str] = None) -> ModelClient
    
    @staticmethod
    def get_available_providers() -> List[str]
```

**Example:**
```python
client = ModelFactory.create_client("openai", "gpt-4", "sk-...")
providers = ModelFactory.get_available_providers()
```

#### `ModelResponse`

Data class for model responses.

```python
@dataclass
class ModelResponse:
    content: str
    model: str
    provider: str
    timestamp: float
    metadata: Dict[str, Any]
    error: Optional[str] = None
```

### `safellm_eval.scoring`

#### `SafetyScorer`

Advanced safety scoring system for LLM responses.

```python
class SafetyScorer:
    def __init__(self)
```

**Methods:**

##### `score_response(response: str, prompt: str, risk_level: str = "medium") -> Dict[str, Any]`

Score a model response for safety across multiple categories.

**Parameters:**
- `response` (str): The model's response text
- `prompt` (str): The original prompt
- `risk_level` (str): Expected risk level ("low", "medium", "high", "critical")

**Returns:**
- Dictionary containing:
  - `overall_score` (float): Overall safety score (0-1)
  - `categories` (Dict[str, float]): Category-specific risk scores
  - `refusal_score` (float): Model refusal appropriateness
  - `explanation` (str): Human-readable explanation

**Example:**
```python
scorer = SafetyScorer()
result = scorer.score_response(
    response="I cannot provide harmful instructions...",
    prompt="How to make explosives?",
    risk_level="critical"
)
print(f"Safety score: {result['overall_score']}")
```

##### `batch_score(responses: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]`

Score multiple responses in batch.

**Parameters:**
- `responses` (List[Tuple]): List of (response, prompt, risk_level) tuples

**Returns:**
- List of scoring results

**Example:**
```python
responses = [
    ("Safe response", "Safe prompt", "low"),
    ("Risky response", "Risky prompt", "high")
]
results = scorer.batch_score(responses)
```

##### `get_category_definitions() -> Dict[str, str]`

Get definitions of all safety categories.

**Returns:**
- Dictionary mapping category names to descriptions

#### `SafetyCategory`

Data class defining a safety category.

```python
@dataclass
class SafetyCategory:
    name: str
    description: str
    keywords: List[str]
    patterns: List[str]
    weight: float
```

### `safellm_eval.visualizer`

#### `ResultVisualizer`

Advanced visualization system for evaluation results.

```python
class ResultVisualizer:
    def __init__(self, style: str = "whitegrid")
```

**Methods:**

##### `create_safety_overview(results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str`

Create comprehensive safety overview dashboard.

**Parameters:**
- `results` (List[Dict]): Evaluation results
- `output_path` (str, optional): Output file path

**Returns:**
- Path to generated visualization

##### `create_language_comparison(results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str`

Create detailed language comparison visualization.

##### `create_model_comparison(results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str`

Create detailed model comparison visualization.

##### `create_risk_analysis(results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str`

Create detailed risk category analysis.

##### `create_summary_report(results: List[Dict[str, Any]], output_dir: str = "./visualizations/") -> Dict[str, str]`

Create comprehensive summary report with all visualizations.

**Parameters:**
- `results` (List[Dict]): Evaluation results
- `output_dir` (str): Output directory

**Returns:**
- Dictionary mapping visualization types to file paths

**Example:**
```python
visualizer = ResultVisualizer()
viz_paths = visualizer.create_summary_report(results, "./my_viz/")
print(f"Dashboard: {viz_paths['overview']}")
```

### `safellm_eval.config`

#### `ConfigManager`

Configuration management system.

```python
class ConfigManager:
    def __init__(self, config_path: Optional[str] = None)
```

**Methods:**

##### `load_config(config_path: str) -> None`

Load configuration from file.

##### `save_config(output_path: str, format: str = "yaml") -> None`

Save current configuration to file.

##### `validate_config() -> Dict[str, List[str]]`

Validate configuration and return validation results.

**Returns:**
- Dictionary with "errors", "warnings", and "info" lists

##### `get_enabled_models() -> List[ModelConfig]`

Get list of enabled models with valid API keys.

##### `create_template_config(output_path: str) -> None`

Create a template configuration file.

**Example:**
```python
config_manager = ConfigManager()
issues = config_manager.validate_config()
if not issues['errors']:
    print("Configuration is valid!")
```

#### Configuration Data Classes

##### `ModelConfig`

```python
@dataclass
class ModelConfig:
    name: str
    provider: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    retry_attempts: int = 3
    enabled: bool = True
```

##### `EvaluationConfig`

```python
@dataclass
class EvaluationConfig:
    batch_size: int = 10
    max_workers: int = 5
    timeout: int = 30
    save_responses: bool = True
    save_format: str = "jsonl"
    output_dir: str = "./results"
    log_level: str = "INFO"
```

##### `SafetyConfig`

```python
@dataclass
class SafetyConfig:
    enabled_categories: List[str]
    custom_keywords: Dict[str, List[str]]
    custom_patterns: Dict[str, List[str]]
    score_threshold: float = 0.5
    strict_mode: bool = False
```

## Command Line Interface

The framework provides a comprehensive CLI through the `safellm-eval` command.

### Global Options

- `--config, -c`: Path to configuration file
- `--verbose, -v`: Enable verbose logging

### Commands

#### `init`

Initialize a new evaluation project.

```bash
safellm-eval init [OPTIONS]
```

**Options:**
- `--output, -o`: Output path for template configuration (default: ./config_template.yaml)

#### `validate`

Validate current configuration.

```bash
safellm-eval validate [OPTIONS]
```

#### `evaluate`

Run multilingual safety evaluation.

```bash
safellm-eval evaluate [OPTIONS] DATASETS...
```

**Options:**
- `--models, -m`: Specific models to evaluate
- `--output, -o`: Output directory for results
- `--format`: Output format (jsonl, csv, json)
- `--visualize/--no-visualize`: Generate visualizations
- `--parallel, -p`: Number of parallel workers

#### `visualize`

Generate visualizations from existing results.

```bash
safellm-eval visualize [OPTIONS] RESULTS_FILE
```

**Options:**
- `--output-dir, -o`: Output directory for visualizations
- `--format`: Visualization formats to generate

#### `list-models`

List available models and their status.

```bash
safellm-eval list-models [OPTIONS]
```

#### `inspect`

Inspect a dataset file and show statistics.

```bash
safellm-eval inspect [OPTIONS] DATASET_PATH
```

#### `info`

Show system information and configuration summary.

```bash
safellm-eval info [OPTIONS]
```

## Error Handling

### Common Exceptions

#### `ConfigurationError`

Raised when configuration is invalid.

```python
try:
    evaluator = MultilingualEvaluator("invalid_config.yaml")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

#### `ModelClientError`

Raised when model client encounters an error.

```python
try:
    response = client.generate_response("test prompt")
except ModelClientError as e:
    print(f"Model error: {e}")
```

#### `DatasetError`

Raised when dataset loading fails.

```python
try:
    prompts = evaluator.load_prompts("invalid_dataset.jsonl")
except DatasetError as e:
    print(f"Dataset error: {e}")
```

## Best Practices

### 1. Configuration Management

```python
# Use environment variables for API keys
import os
from safellm_eval import ConfigManager

config_manager = ConfigManager()
config_manager.config.models[0].api_key = os.getenv("OPENAI_API_KEY")
```

### 2. Error Handling

```python
# Always handle potential errors
try:
    results = evaluator.evaluate_dataset("dataset.jsonl")
except Exception as e:
    logger.error(f"Evaluation failed: {e}")
    # Handle gracefully
```

### 3. Resource Management

```python
# Use context managers for file operations
import jsonlines

with jsonlines.open("results.jsonl") as reader:
    for result in reader:
        process_result(result)
```

### 4. Batch Processing

```python
# Process large datasets in batches
def process_large_dataset(dataset_path, batch_size=100):
    prompts = evaluator.load_prompts(dataset_path)
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        # Process batch
        results = process_batch(batch)
        save_batch_results(results, f"batch_{i}.jsonl")
```

## Type Hints

The framework uses comprehensive type hints. Import common types:

```python
from typing import Dict, List, Optional, Any, Union, Tuple
from safellm_eval.evaluator import EvaluationResult
from safellm_eval.models import ModelResponse
from safellm_eval.config import ModelConfig, SafeLLMConfig
```

## Examples

### Complete Evaluation Workflow

```python
from safellm_eval import MultilingualEvaluator, ResultVisualizer
import os

# Set up API keys
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Initialize evaluator with custom config
evaluator = MultilingualEvaluator("my_config.yaml")

# Run evaluation
results = evaluator.evaluate_dataset("datasets/test_prompts.jsonl")

# Save results
evaluator.save_results("results/my_evaluation.jsonl")

# Generate visualizations
visualizer = ResultVisualizer()
viz_paths = visualizer.create_summary_report(results)

# Print summary
summary = evaluator.get_summary_stats()
print(f"Evaluated {summary['total_evaluations']} prompts")
print(f"Average safety score: {summary['average_safety_score']:.3f}")
```

### Custom Safety Scoring

```python
from safellm_eval.scoring import SafetyScorer, SafetyCategory

# Create custom scorer
scorer = SafetyScorer()

# Add custom category
custom_category = SafetyCategory(
    name="Custom Risk",
    description="Domain-specific risk category",
    keywords=["custom_keyword1", "custom_keyword2"],
    patterns=[r"\bcustom_pattern\b"],
    weight=2.0
)

scorer.categories["custom_risk"] = custom_category

# Use custom scorer
result = scorer.score_response(
    response="Model response with custom keywords",
    prompt="Test prompt",
    risk_level="medium"
)

print(f"Custom category score: {result['categories']['custom_risk']}")
```

---

This API reference provides comprehensive documentation for all public interfaces in the SafeLLM Multilingual Evaluation Framework. For usage examples and tutorials, see the [User Guide](user_guide.md).