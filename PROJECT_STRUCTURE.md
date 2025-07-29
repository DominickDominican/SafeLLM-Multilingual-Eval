# Project Structure Overview

This document provides a comprehensive overview of the SafeLLM Multilingual Evaluation Framework project structure.

## 📁 Directory Structure

```
SafeLLM-Multilingual-Eval/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT license
├── 📄 CHANGELOG.md                 # Version history and changes
├── 📄 AUTHORS.md                   # Contributors and maintainers
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 MANIFEST.in                  # Package distribution files
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package installation script
├── 📄 pyproject.toml              # Build system configuration
├── 📄 pytest.ini                  # Test configuration
├── 📄 .flake8                     # Code linting configuration
├── 📄 .pre-commit-config.yaml     # Pre-commit hooks
├── 📄 .gitignore                  # Git ignore patterns
├── 📄 .env.example                # Environment variables template
├── 📄 config.yaml                 # Default configuration
├── 📄 Makefile                    # Development commands
│
├── 📁 safellm_eval/               # Main package source code
│   ├── 📄 __init__.py            # Package initialization
│   ├── 📄 cli.py                 # Command-line interface
│   ├── 📄 config.py              # Configuration management
│   ├── 📄 evaluator.py           # Core evaluation logic
│   ├── 📄 models.py              # Model client implementations
│   ├── 📄 scoring.py             # Safety scoring system
│   └── 📄 visualizer.py          # Visualization components
│
├── 📁 tests/                      # Test suite
│   ├── 📄 conftest.py            # Test configuration and fixtures
│   ├── 📄 test_safellm_eval.py   # Core functionality tests
│   └── 📄 test_cli.py            # CLI testing
│
├── 📁 datasets/                   # Evaluation datasets
│   ├── 📄 comprehensive_prompts.jsonl  # Main multilingual test set
│   └── 📄 benign_prompts.jsonl         # Safe baseline prompts
│
├── 📁 docs/                       # Documentation
│   ├── 📄 user_guide.md          # Comprehensive user guide
│   ├── 📄 api_reference.md       # API documentation
│   └── 📄 developer_guide.md     # Developer documentation
│
├── 📁 examples/                   # Usage examples
│   ├── 📄 quick_start.py         # Basic usage example
│   └── 📄 advanced_example.py    # Advanced features demo
│
├── 📁 .github/                    # GitHub configuration
│   └── 📁 workflows/             # CI/CD workflows
│       └── 📄 ci.yml             # Main CI/CD pipeline
│
├── 📁 results/                    # Evaluation results (created at runtime)
└── 📁 visualizations/             # Generated visualizations (created at runtime)
```

## 🏗️ Architecture Components

### Core Package (`safellm_eval/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `__init__.py` | Package entry point | Exports main classes |
| `evaluator.py` | Main evaluation engine | `MultilingualEvaluator`, `EvaluationResult` |
| `models.py` | LLM API clients | `ModelClient`, `OpenAIClient`, `AnthropicClient`, `MistralClient` |
| `scoring.py` | Safety assessment | `SafetyScorer`, `SafetyCategory` |
| `visualizer.py` | Analysis dashboards | `ResultVisualizer` |
| `config.py` | Configuration management | `ConfigManager`, `SafeLLMConfig` |
| `cli.py` | Command-line interface | CLI commands and argument parsing |

### Testing (`tests/`)

| File | Purpose | Coverage |
|------|---------|----------|
| `conftest.py` | Test configuration | Fixtures, mocks, test utilities |
| `test_safellm_eval.py` | Core functionality | Unit and integration tests |
| `test_cli.py` | CLI testing | Command-line interface tests |

### Documentation (`docs/`)

| File | Audience | Content |
|------|----------|---------|
| `user_guide.md` | End users | Installation, usage, examples |
| `api_reference.md` | Developers | Complete API documentation |
| `developer_guide.md` | Contributors | Architecture, extending framework |

### Datasets (`datasets/`)

| File | Content | Languages | Prompts |
|------|---------|-----------|---------|
| `comprehensive_prompts.jsonl` | Adversarial test cases | 15+ | ~45 |
| `benign_prompts.jsonl` | Safe baseline prompts | 10+ | ~25 |

### Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| `config.yaml` | Default settings | YAML |
| `.env.example` | Environment template | Shell variables |
| `requirements.txt` | Dependencies | pip format |
| `setup.py` | Package metadata | Python setuptools |
| `pyproject.toml` | Build configuration | TOML |

## 🚀 Key Features by Component

### Evaluator (`evaluator.py`)
- ✅ Multilingual prompt processing
- ✅ Parallel evaluation with configurable workers
- ✅ Comprehensive error handling and logging
- ✅ Multiple output formats (JSONL, CSV, JSON)
- ✅ Progress tracking and statistics

### Model Clients (`models.py`)
- ✅ Unified interface for multiple LLM providers
- ✅ Rate limiting and retry mechanisms
- ✅ Response caching and metadata tracking
- ✅ Extensible architecture for new providers

### Safety Scoring (`scoring.py`)
- ✅ Multi-dimensional risk assessment
- ✅ 6 built-in safety categories
- ✅ Multilingual keyword and pattern matching
- ✅ Configurable scoring weights and thresholds
- ✅ Human-readable explanations

### Visualization (`visualizer.py`)
- ✅ Interactive HTML dashboards
- ✅ Static plots with matplotlib/seaborn
- ✅ Multi-format output (HTML, PNG, SVG)
- ✅ Automated report generation
- ✅ Customizable styling and themes

### Configuration (`config.py`)
- ✅ Hierarchical configuration system
- ✅ Environment variable integration
- ✅ Validation and error reporting
- ✅ Template generation
- ✅ Runtime configuration updates

### CLI (`cli.py`)
- ✅ Intuitive command structure
- ✅ Rich help and documentation
- ✅ Progress bars and status updates
- ✅ Configuration validation
- ✅ Batch processing support

## 📊 Supported Formats and Standards

### Input Formats
- **Datasets**: JSONL (JSON Lines)
- **Configuration**: YAML, Environment variables
- **API Keys**: Environment variables, .env files

### Output Formats
- **Results**: JSONL, CSV, JSON
- **Visualizations**: HTML (interactive), PNG, SVG
- **Reports**: HTML with embedded visualizations
- **Logs**: Structured logging with timestamps

### Language Support
- **High-resource**: English, Chinese, Spanish, French, German, Russian, Portuguese, Italian
- **Medium-resource**: Arabic, Hindi, Vietnamese, Thai, Japanese, Korean, Turkish
- **Low-resource**: Swahili, Bengali, Urdu, Hausa, Malay, Indonesian, Dutch

### Domain Coverage
- 🏥 **Healthcare**: Medical advice, treatment recommendations
- ⚖️ **Legal**: Legal guidance, compliance advice
- 🎓 **Education**: Academic integrity, learning assistance
- 💰 **Finance**: Financial advice, investment guidance
- 🛡️ **Safety**: Harm prevention, dangerous instructions
- 📰 **Misinformation**: Information accuracy, fact-checking
- 🌐 **General**: Everyday queries and conversations

## 🔧 Development Workflow

### Setup
```bash
git clone https://github.com/DominickDominican/SafeLLM-Multilingual-Eval.git
cd SafeLLM-Multilingual-Eval
make install-dev
```

### Testing
```bash
make test          # Run all tests
make test-cov      # Run with coverage
make lint          # Code quality checks
make format        # Auto-format code
```

### Building
```bash
make build         # Build package
make docs          # Generate documentation
```

### Quality Assurance
- **Linting**: flake8 for code quality
- **Formatting**: black for consistent style
- **Type Checking**: mypy for type safety
- **Testing**: pytest with comprehensive coverage
- **Security**: bandit for security scanning
- **Dependencies**: safety for vulnerability checking

## 📈 Performance Characteristics

### Scalability
- **Parallel Processing**: Configurable worker threads (1-20)
- **Batch Processing**: Adjustable batch sizes (1-100)
- **Memory Management**: Efficient data structures, garbage collection
- **Rate Limiting**: Automatic throttling for API compliance

### Throughput
- **Small datasets** (<100 prompts): ~5-10 seconds
- **Medium datasets** (100-1000 prompts): ~1-5 minutes
- **Large datasets** (1000+ prompts): ~10-30 minutes
- **Factors**: Model provider, prompt complexity, network latency

### Resource Usage
- **Memory**: ~50-200MB base + ~1-5MB per 100 prompts
- **CPU**: Minimal, primarily I/O bound
- **Storage**: ~1-10MB per 1000 evaluation results
- **Network**: Dependent on model provider APIs

## 🛡️ Security Considerations

### API Key Management
- Environment variables for secure storage
- No hardcoded credentials in source code
- .env file support with .gitignore protection
- Key validation and error handling

### Data Privacy
- No persistent storage of sensitive prompts
- Configurable data retention policies
- Local processing with optional cloud APIs
- Audit logging for compliance

### Input Validation
- Schema validation for all inputs
- Path traversal protection
- SQL injection prevention (not applicable)
- Malicious payload detection

## 🚀 Deployment Options

### Local Development
```bash
pip install -e .
safellm-eval init
```

### Production Deployment
```bash
pip install safellm-multilingual-eval
# Configure via environment variables
# Run via CLI or Python API
```

### Docker Deployment
```bash
docker build -t safellm-eval .
docker run -e OPENAI_API_KEY=xyz safellm-eval
```

### CI/CD Integration
- GitHub Actions workflows included
- Automated testing on multiple Python versions
- Continuous deployment to PyPI
- Documentation building and deployment

## 📞 Support and Community

### Getting Help
- **Documentation**: Comprehensive guides and API reference
- **Issues**: GitHub issue tracker for bugs and feature requests
- **Discussions**: GitHub discussions for questions
- **Email**: Direct maintainer contact for sensitive issues

### Contributing
- **Code**: Bug fixes, features, performance improvements
- **Documentation**: User guides, API docs, examples
- **Testing**: Test cases, integration scenarios
- **Datasets**: New languages, domains, evaluation cases
- **Translations**: Localization of documentation

### Community
- Open source MIT license
- Welcoming contributor community
- Regular releases and updates
- Responsive maintainer support

---

This project structure is designed for maintainability, extensibility, and ease of use. Each component has a clear purpose and well-defined interfaces, making it easy to understand, modify, and extend the framework for your specific needs.