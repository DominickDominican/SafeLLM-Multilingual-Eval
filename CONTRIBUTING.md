# Contributing to SafeLLM Multilingual Evaluation Framework

We welcome contributions from researchers, developers, and AI safety enthusiasts! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Types of Contributions

- **🐛 Bug Reports**: Report issues or unexpected behavior
- **✨ Feature Requests**: Suggest new features or improvements
- **📖 Documentation**: Improve or add documentation
- **🧪 Test Cases**: Add test cases or improve test coverage
- **🌍 Multilingual Support**: Add new languages or improve language support
- **📊 Datasets**: Contribute new evaluation datasets
- **🔧 Code Improvements**: Fix bugs, optimize performance, or refactor code

### Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YourUsername/SafeLLM-Multilingual-Eval.git
   cd SafeLLM-Multilingual-Eval
   ```

2. **Set Up Development Environment**
   ```bash
   make install-dev
   pre-commit install
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**
   - Write clean, well-documented code
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation as needed

5. **Test Your Changes**
   ```bash
   make test
   make lint
   make format
   ```

6. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description of your changes
   - Link any related issues
   - Include screenshots for UI changes

## 📋 Development Guidelines

### Code Style

- **Python**: Follow PEP 8 guidelines
- **Line Length**: Maximum 88 characters (Black formatter)
- **Imports**: Use `isort` for import organization
- **Type Hints**: Use type hints for all functions and methods
- **Docstrings**: Use Google-style docstrings

Example:
```python
def evaluate_model_safety(
    model_response: str, 
    prompt: str, 
    risk_level: str = "medium"
) -> Dict[str, Any]:
    """Evaluate the safety of a model response.
    
    Args:
        model_response: The response text from the model.
        prompt: The original prompt that generated the response.
        risk_level: Expected risk level ('low', 'medium', 'high', 'critical').
        
    Returns:
        Dictionary containing safety scores and analysis.
        
    Raises:
        ValueError: If risk_level is not recognized.
    """
    # Implementation here
    pass
```

### Testing

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Coverage**: Aim for >90% test coverage

```python
def test_safety_scorer_multilingual():
    """Test safety scorer with multilingual content."""
    scorer = SafetyScorer()
    
    # Test cases for different languages
    test_cases = [
        ("English harmful content", "high"),
        ("中文有害内容", "high"),
        ("محتوى ضار بالعربية", "high")
    ]
    
    for content, expected_risk in test_cases:
        result = scorer.score_response(content, "test prompt", expected_risk)
        assert result['overall_score'] < 0.5
```

### Documentation

- **README**: Keep the main README comprehensive but concise
- **API Documentation**: Document all public functions and classes
- **User Guides**: Provide step-by-step tutorials
- **Examples**: Include practical examples

## 🌍 Adding New Languages

To add support for a new language:

1. **Add Language Configuration**
   ```yaml
   # config.yaml
   languages:
     - NewLanguage
   ```

2. **Create Language-Specific Test Data**
   ```json
   {
     "language": "NewLanguage",
     "domain": "Healthcare",
     "prompt_type": "adversarial", 
     "risk_level": "high",
     "prompt": "Harmful prompt in new language"
   }
   ```

3. **Update Safety Scorer Keywords**
   ```python
   # In scoring.py
   keywords=[
       # Existing keywords...
       "harmful_word_in_new_language",
   ]
   ```

4. **Add Language Tests**
   ```python
   def test_new_language_support():
       """Test evaluation with new language."""
       # Test implementation
   ```

## 📊 Contributing Datasets

When contributing new datasets:

### Dataset Requirements

- **Format**: Use JSONL format
- **Schema**: Follow the standard schema
- **Quality**: Ensure high-quality, diverse examples
- **Ethics**: Avoid genuinely harmful content
- **Balance**: Include both safe and risky examples

### Dataset Schema
```json
{
  "language": "Language name",
  "domain": "Domain category", 
  "prompt_type": "benign|adversarial|biased|harmful|conspiracy|misalignment",
  "risk_level": "low|medium|high|critical",
  "prompt": "The actual prompt text",
  "metadata": {
    "source": "Dataset source",
    "created_by": "Creator name",
    "reviewed": true
  }
}
```

### Ethical Guidelines

- **No Real Harm**: Don't include content that could cause actual harm
- **Research Purpose**: Focus on evaluation and safety research
- **Cultural Sensitivity**: Be respectful of cultural differences
- **Legal Compliance**: Ensure compliance with local laws

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Environment**: Python version, OS, package versions
- **Steps to Reproduce**: Clear, step-by-step instructions
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Include full error messages and stack traces
- **Code Sample**: Minimal code to reproduce the issue

Example bug report:
```markdown
## Bug Description
SafetyScorer fails on empty responses

## Environment
- Python 3.10.2
- safellm-eval 0.1.0
- Ubuntu 22.04

## Steps to Reproduce
1. Initialize SafetyScorer
2. Call score_response with empty string
3. Method crashes with KeyError

## Expected Behavior
Should return score of 0.0 with appropriate error message

## Actual Behavior
Crashes with KeyError: 'content'

## Error Message
```
KeyError: 'content'
  File "scoring.py", line 123, in score_response
```

### Feature Requests

For feature requests, please provide:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Any alternative approaches considered?
- **Impact**: Who would benefit from this feature?

## 🏷️ Commit Message Format

We use conventional commit messages:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `style:` Code style changes
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Examples:
```
feat: add support for Gemini API
fix: handle empty responses in safety scorer
docs: update installation instructions  
test: add integration tests for CLI
```

## 🔍 Code Review Process

### Pull Request Reviews

All pull requests require:

- **Automated Tests**: All CI checks must pass
- **Code Review**: At least one maintainer review
- **Documentation**: Updated if functionality changes
- **Changelog**: Entry in CHANGELOG.md for user-facing changes

### Review Criteria

- **Functionality**: Does it work as intended?
- **Code Quality**: Is it well-written and maintainable?
- **Testing**: Are there adequate tests?
- **Documentation**: Is it properly documented?
- **Performance**: Does it impact performance?
- **Security**: Are there any security concerns?

## 🏆 Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: All contributors listed
- **Release Notes**: Significant contributions highlighted
- **Documentation**: Credits in relevant sections

## 📞 Getting Help

- **GitHub Discussions**: For questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Email**: dominickdominican47@gmail.com for sensitive issues

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- **Be Respectful**: Treat all participants with respect
- **Be Inclusive**: Welcome contributors from all backgrounds
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Patient**: Help newcomers learn and contribute

## ⚖️ Legal

By contributing, you agree that:

- Your contributions will be licensed under the MIT License
- You have the right to contribute the code/content
- Your contributions don't violate any third-party rights

---

Thank you for contributing to SafeLLM Multilingual Evaluation Framework! 🚀