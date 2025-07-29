"""
Main evaluator class for running multilingual safety assessments.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time

from .models import ModelFactory, ModelResponse
from .scoring import SafetyScorer


@dataclass
class EvaluationResult:
    """Results from evaluating a single prompt."""
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


class MultilingualEvaluator:
    """Main evaluator for multilingual LLM safety assessment."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the evaluator."""
        self.logger = logging.getLogger(__name__)
        self.safety_scorer = SafetyScorer()
        self.results: List[EvaluationResult] = []
        
        # Default configuration
        self.config = {
            "models": [
                {"name": "mock-gpt-4", "provider": "mock"},
                {"name": "mock-claude", "provider": "mock"}
            ],
            "evaluation": {
                "batch_size": 10,
                "max_workers": 2,
                "timeout": 30,
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
    
    def load_prompts(self, file_path: str) -> List[Dict[str, Any]]:
        """Load prompts from JSONL file."""
        prompts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            obj = json.loads(line.strip())
                            obj['prompt_id'] = f"{os.path.basename(file_path)}_{i}"
                            prompts.append(obj)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON on line {i+1}: {e}")
                            continue
            
            self.logger.info(f"Loaded {len(prompts)} prompts from {file_path}")
            return prompts
            
        except Exception as e:
            self.logger.error(f"Error loading prompts from {file_path}: {e}")
            return []
    
    def evaluate_single_prompt(self, prompt_data: Dict[str, Any], model_config: Dict[str, str]) -> EvaluationResult:
        """Evaluate a single prompt with a specific model."""
        try:
            # Create model client
            client = ModelFactory.create_client(
                provider=model_config["provider"],
                model_name=model_config["name"]
            )
            
            # Generate response
            response = client.generate_response(
                prompt=prompt_data["prompt"],
                temperature=self.config["evaluation"]["temperature"],
                max_tokens=self.config["evaluation"]["max_tokens"]
            )
            
            if response.error:
                return EvaluationResult(
                    prompt_id=prompt_data["prompt_id"],
                    language=prompt_data.get("language", "unknown"),
                    domain=prompt_data.get("domain", "unknown"),
                    prompt_type=prompt_data.get("prompt_type", "unknown"),
                    risk_level=prompt_data.get("risk_level", "unknown"),
                    prompt=prompt_data["prompt"],
                    model_response="",
                    model=model_config["name"],
                    provider=model_config["provider"],
                    safety_score=0.0,
                    risk_categories={},
                    metadata=response.metadata,
                    timestamp=response.timestamp,
                    error=response.error
                )
            
            # Calculate safety score
            safety_result = self.safety_scorer.score_response(
                response.content,
                prompt_data["prompt"],
                prompt_data.get("risk_level", "medium")
            )
            
            return EvaluationResult(
                prompt_id=prompt_data["prompt_id"],
                language=prompt_data.get("language", "unknown"),
                domain=prompt_data.get("domain", "unknown"),
                prompt_type=prompt_data.get("prompt_type", "unknown"),
                risk_level=prompt_data.get("risk_level", "unknown"),
                prompt=prompt_data["prompt"],
                model_response=response.content,
                model=model_config["name"],
                provider=model_config["provider"],
                safety_score=safety_result["overall_score"],
                risk_categories=safety_result["categories"],
                metadata=response.metadata,
                timestamp=response.timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating prompt {prompt_data.get('prompt_id', 'unknown')}: {e}")
            return EvaluationResult(
                prompt_id=prompt_data.get("prompt_id", "unknown"),
                language=prompt_data.get("language", "unknown"),
                domain=prompt_data.get("domain", "unknown"),
                prompt_type=prompt_data.get("prompt_type", "unknown"),
                risk_level=prompt_data.get("risk_level", "unknown"),
                prompt=prompt_data.get("prompt", ""),
                model_response="",
                model=model_config["name"],
                provider=model_config["provider"],
                safety_score=0.0,
                risk_categories={},
                metadata={},
                timestamp=time.time(),
                error=str(e)
            )
    
    def evaluate_dataset(self, dataset_path: str, models: Optional[List[Dict[str, str]]] = None) -> List[EvaluationResult]:
        """Evaluate entire dataset across multiple models."""
        prompts = self.load_prompts(dataset_path)
        if not prompts:
            return []
        
        models = models or self.config["models"]
        
        self.logger.info(f"Starting evaluation of {len(prompts)} prompts with {len(models)} models")
        
        results = []
        
        # Simple sequential processing for reliability
        for prompt_data in prompts:
            for model_config in models:
                result = self.evaluate_single_prompt(prompt_data, model_config)
                results.append(result)
                
                # Simple progress logging
                if len(results) % 5 == 0:
                    self.logger.info(f"Completed {len(results)} evaluations...")
        
        self.results.extend(results)
        self.logger.info(f"Completed evaluation. Total results: {len(results)}")
        return results
    
    def save_results(self, output_path: str, format: str = "jsonl"):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            if format.lower() == "jsonl":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for result in self.results:
                        f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
            
            elif format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([asdict(result) for result in self.results], f, indent=2, ensure_ascii=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from evaluation results."""
        if not self.results:
            return {"total_evaluations": 0}
        
        successful_results = [r for r in self.results if r.error is None]
        failed_results = [r for r in self.results if r.error is not None]
        
        summary = {
            "total_evaluations": len(self.results),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(failed_results)
        }
        
        if successful_results:
            safety_scores = [r.safety_score for r in successful_results]
            summary.update({
                "average_safety_score": sum(safety_scores) / len(safety_scores),
                "min_safety_score": min(safety_scores),
                "max_safety_score": max(safety_scores)
            })
            
            # Language breakdown
            language_scores = {}
            for result in successful_results:
                if result.language not in language_scores:
                    language_scores[result.language] = []
                language_scores[result.language].append(result.safety_score)
            
            summary["safety_score_by_language"] = {
                lang: sum(scores) / len(scores) 
                for lang, scores in language_scores.items()
            }
            
            # Domain breakdown  
            domain_scores = {}
            for result in successful_results:
                if result.domain not in domain_scores:
                    domain_scores[result.domain] = []
                domain_scores[result.domain].append(result.safety_score)
            
            summary["safety_score_by_domain"] = {
                domain: sum(scores) / len(scores)
                for domain, scores in domain_scores.items()
            }
        
        return summary