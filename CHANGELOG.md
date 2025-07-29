# Changelog

All notable changes to the SafeLLM Multilingual Evaluation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial framework architecture and core functionality
- Multilingual support for 20+ languages including low-resource languages
- Advanced safety scoring system with 6 risk categories
- Support for OpenAI, Anthropic, and Mistral APIs
- Comprehensive visualization system with interactive dashboards
- Command-line interface for all operations
- Configuration management with YAML support
- Parallel processing with configurable batch sizes
- Full test suite with CI/CD pipeline
- Extensive documentation and user guides

### Features

#### Core Evaluation System
- `MultilingualEvaluator` class for orchestrating evaluations
- Support for JSONL dataset format with flexible schema
- Automatic prompt ID generation and metadata tracking
- Error handling and retry mechanisms for API calls
- Batch processing with progress tracking
- Results export in multiple formats (JSONL, CSV, JSON)

#### Model Support
- Abstract `ModelClient` base class for extensibility
- `OpenAIClient` for GPT models (GPT-4, GPT-3.5-turbo)
- `AnthropicClient` for Claude models (Claude-3-Sonnet, Claude-3-Haiku)
- `MistralClient` for Mistral models (Mistral-Medium, Mistral-Small)
- `ModelFactory` for dynamic client creation
- Rate limiting and timeout handling
- Comprehensive error logging and reporting

#### Safety Scoring System
- `SafetyScorer` with multi-dimensional risk assessment
- Six built-in risk categories:
  - Harmful Content: Physical/psychological harm instructions
  - Misinformation: False or misleading information
  - Illegal Activity: Guidance on illegal actions
  - Bias & Discrimination: Prejudiced or discriminatory content
  - Privacy Violation: Personal data misuse
  - Inappropriate Content: Sexually explicit material
- Multilingual keyword and pattern matching
- Configurable scoring thresholds and weights
- Refusal score calculation for model appropriateness
- Risk level adjustment (low, medium, high, critical)
- Human-readable explanations for scores

#### Visualization System
- `ResultVisualizer` for comprehensive analysis reports
- Interactive HTML dashboards with Plotly
- Static visualizations with Matplotlib and Seaborn
- Multiple visualization types:
  - Safety Overview Dashboard: Key metrics and distributions
  - Language Comparison: Cross-lingual performance analysis
  - Model Comparison: Comparative analysis across LLMs
  - Risk Category Analysis: Detailed risk breakdowns
  - Timeline Analysis: Performance trends over time
- Customizable styling and output formats
- Automatic report generation with timestamps

#### Configuration Management
- `ConfigManager` for centralized configuration handling
- YAML-based configuration files with validation
- Environment variable support for sensitive data
- Template configuration generation
- Configuration validation with detailed error reporting
- Support for custom safety categories and keywords
- Model enable/disable functionality
- Flexible output and visualization settings

#### Command-Line Interface
- Comprehensive CLI with intuitive commands:
  - `init`: Initialize new evaluation projects
  - `validate`: Validate configuration files
  - `evaluate`: Run multilingual safety evaluations
  - `visualize`: Generate visualizations from results
  - `list-models`: Show available models and status
  - `inspect`: Analyze dataset statistics
  - `info`: Display system information
- Rich command-line output with progress bars
- Flexible argument parsing and validation
- Integration with configuration system

#### Dataset Support
- Comprehensive multilingual test datasets
- `comprehensive_prompts.jsonl`: Adversarial prompts across domains
- `benign_prompts.jsonl`: Safe baseline prompts for comparison
- Support for custom dataset creation
- Flexible JSONL schema with optional fields
- Automatic dataset validation and statistics
- Domain categorization (Healthcare, Legal, Education, Finance, Safety, General)
- Risk level classification (low, medium, high, critical)
- Prompt type classification (benign, adversarial, biased, harmful, conspiracy, misalignment)

#### Language Support
High-Resource Languages:
- English
- Chinese (Simplified)
- Spanish
- French
- German
- Russian
- Portuguese
- Italian

Medium-Resource Languages:
- Arabic
- Hindi
- Vietnamese
- Thai
- Japanese
- Korean
- Turkish

Low-Resource Languages:
- Swahili
- Bengali
- Urdu
- Hausa
- Malay
- Indonesian
- Dutch

#### Testing and Quality Assurance
- Comprehensive test suite with pytest
- Unit tests for all core components
- Integration tests for end-to-end workflows
- Mock API clients for reliable testing
- Code coverage reporting with codecov
- Continuous integration with GitHub Actions
- Code quality checks with flake8, black, and mypy
- Security scanning with bandit
- Pre-commit hooks for code quality

#### Documentation
- Comprehensive README with quick start guide
- Detailed user guide with examples
- Complete API reference documentation
- Developer guide for contributors
- Contributing guidelines with development workflow
- MIT license for open-source usage
- Code of conduct for inclusive community

#### Development Tools
- Makefile for common development tasks
- Pre-commit configuration for code quality
- GitHub Actions workflows for CI/CD
- Docker support for containerized deployment
- Requirements management with pinned versions
- Development environment setup scripts

### Technical Specifications

#### Performance
- Parallel processing support with configurable worker count
- Batch processing to manage memory usage
- Rate limiting and retry mechanisms for API stability
- Efficient data structures for large dataset handling
- Memory monitoring and garbage collection
- Caching support for repeated evaluations

#### Extensibility
- Plugin architecture for new model providers
- Extensible safety category system
- Custom keyword and pattern support
- Flexible visualization components
- Configurable evaluation pipelines
- Custom dataset format support

#### Reliability
- Comprehensive error handling and logging
- Graceful degradation for API failures
- Data validation at all input points
- Atomic operations for data integrity
- Recovery mechanisms for interrupted evaluations
- Detailed audit trails for all operations

#### Security
- Secure API key management
- Input sanitization and validation
- No sensitive data in logs or outputs
- Secure communication with all APIs
- Regular dependency updates
- Security scanning in CI pipeline

## [0.1.0] - 2024-07-29

### Added
- Initial release of SafeLLM Multilingual Evaluation Framework
- Core evaluation engine with multilingual support
- Safety scoring system with risk categorization
- Multi-provider model client architecture
- Comprehensive visualization and reporting system
- Command-line interface for user-friendly operation
- Extensive documentation and examples
- Full test suite with continuous integration
- Open-source release under MIT license

---

## Release Notes Format

For each release, we document:

- **Added**: New features and capabilities
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes and corrections
- **Security**: Security-related improvements

## Version Numbering

We follow Semantic Versioning (SemVer):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backwards compatible manner
- **PATCH**: Backwards compatible bug fixes

## Contributing to Changelog

When contributing, please:
1. Add entries to the [Unreleased] section
2. Follow the established format and categories
3. Include clear, concise descriptions
4. Reference issue numbers when applicable
5. Update the changelog in your pull request

## Links

- [Repository](https://github.com/DominickDominican/SafeLLM-Multilingual-Eval)
- [Documentation](https://safellm-multilingual-eval.readthedocs.io)
- [Issues](https://github.com/DominickDominican/SafeLLM-Multilingual-Eval/issues)
- [Releases](https://github.com/DominickDominican/SafeLLM-Multilingual-Eval/releases)