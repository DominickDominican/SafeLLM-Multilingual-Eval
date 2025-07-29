"""
Main evaluation engine for SafeLLM Multilingual Evaluation.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .models import ModelClient, create_model_client
from .scoring import SafetyScorer
from .config import ConfigManager


@dataclass
class EvaluationResult:
    """Results of evaluating a single prompt."""
    prompt: str
    language: str
    domain: str
    model: str
    provider: str
    response: Optional[str]
    safety_score: float
    safety_details: Dict[str, Any]
    timestamp: float
    error: Optional[str] = None


class MultilingualEvaluator:
    """Main evaluation engine for multilingual LLM safety testing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the evaluator."""
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager(config_path)
        self.safety_scorer = SafetyScorer()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load prompts from JSONL dataset file."""
        prompts = []
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        prompt_data = json.loads(line)
                        prompts.append(prompt_data)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
            
            self.logger.info(f"Loaded {len(prompts)} prompts from {dataset_path}")
            return prompts
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def evaluate_single_prompt(self, prompt_data: Dict[str, Any], 
                             model_client: ModelClient) -> EvaluationResult:
        """Evaluate a single prompt with a single model."""
        
        prompt = prompt_data.get('prompt', '')
        language = prompt_data.get('language', 'unknown')
        domain = prompt_data.get('domain', 'unknown')
        risk_level = prompt_data.get('risk_level', 'medium')
        
        try:
            # Get model response
            response = model_client.generate_response(prompt)
            
            if response.error:
                return EvaluationResult(
                    prompt=prompt,
                    language=language,
                    domain=domain,
                    model=response.model,
                    provider=response.provider,
                    response=None,
                    safety_score=0.0,
                    safety_details={},
                    timestamp=time.time(),
                    error=response.error
                )
            
            # Score the response for safety
            safety_result = self.safety_scorer.score_response(
                response.content, prompt, risk_level
            )
            
            return EvaluationResult(
                prompt=prompt,
                language=language,
                domain=domain,
                model=response.model,
                provider=response.provider,
                response=response.content,
                safety_score=safety_result['overall_score'],
                safety_details=safety_result,
                timestamp=time.time(),
                error=None
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating prompt: {e}")
            return EvaluationResult(
                prompt=prompt,
                language=language,
                domain=domain,
                model="unknown",
                provider="unknown",
                response=None,
                safety_score=0.0,
                safety_details={},
                timestamp=time.time(),
                error=str(e)
            )
    
    def run_evaluation(self, dataset_path: str, models: Optional[List[str]] = None) -> List[EvaluationResult]:
        """Run evaluation on a dataset with specified models."""
        
        # Load dataset
        prompts = self.load_dataset(dataset_path)
        if not prompts:
            self.logger.warning("No prompts found in dataset")
            return []
        
        # Get enabled models
        enabled_models = self.config_manager.get_enabled_models()
        if models:
            # Filter to requested models
            model_names = {model.name for model in enabled_models}
            enabled_models = [m for m in enabled_models if m.name in models]
        
        if not enabled_models:
            self.logger.error("No enabled models found")
            return []
        
        self.logger.info(f"Evaluating {len(prompts)} prompts with {len(enabled_models)} models")
        
        # Run evaluation
        results = []
        total_evaluations = len(prompts) * len(enabled_models)
        current_evaluation = 0
        
        for model_config in enabled_models:
            self.logger.info(f"Starting evaluation with model: {model_config.name}")
            
            # Create model client
            try:
                model_client = create_model_client(
                    model_config.provider, 
                    model_config.name,
                    model_config.api_key
                )
            except Exception as e:
                self.logger.error(f"Failed to create client for {model_config.name}: {e}")
                continue
            
            # Evaluate each prompt
            for prompt_data in prompts:
                current_evaluation += 1
                self.logger.info(f"Progress: {current_evaluation}/{total_evaluations}")
                
                result = self.evaluate_single_prompt(prompt_data, model_client)
                results.append(result)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
        
        self.logger.info(f"Evaluation completed. {len(results)} results generated.")
        return results

def main():
    """Main CLI entry point for the evaluator."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='SafeLLM Multilingual Evaluation')
    parser.add_argument('--dataset', '-d', required=True, help='Path to JSONL dataset file')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--output', '-o', default='./results', help='Output directory')
    parser.add_argument('--models', '-m', nargs='+', help='Models to evaluate')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = MultilingualEvaluator(args.config)
        
        print(f"Starting evaluation with dataset: {args.dataset}")
        
        # Run evaluation
        results = evaluator.run_evaluation(
            dataset_path=args.dataset,
            models=args.models
        )
        
        print(f"\n✅ Evaluation completed! {len(results)} results generated.")
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        results_file = os.path.join(args.output, f"evaluation_results_{int(time.time())}.jsonl")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            for result in results:
                result_dict = {
                    'prompt': result.prompt,
                    'language': result.language,
                    'domain': result.domain,
                    'model': result.model,
                    'provider': result.provider,
                    'response': result.response,
                    'safety_score': result.safety_score,
                    'safety_details': result.safety_details,
                    'timestamp': result.timestamp,
                    'error': result.error
                }
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        
        print(f"💾 Results saved to: {results_file}")
        
        # Create visualizations if results exist
        if results:
            try:
                from .visualizer import ResultVisualizer
                visualizer = ResultVisualizer()
                
                # Convert results to dict format for visualization
                result_dicts = []
                for result in results:
                    result_dict = {
                        'language': result.language,
                        'domain': result.domain,
                        'model': result.model,
                        'safety_score': result.safety_score,
                        'response': result.response,
                        'error': result.error
                    }
                    result_dicts.append(result_dict)
                
                # Generate reports
                report_info = visualizer.create_summary_report(result_dicts, args.output)
                print(f"📊 Reports generated in: {report_info.get('output_directory', args.output)}")
                
            except ImportError as e:
                print(f"⚠️  Could not create visualizations: {e}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()