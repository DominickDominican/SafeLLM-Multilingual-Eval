"""
Basic visualization system for evaluation results.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import json


class ResultVisualizer:
    """Basic visualization system for evaluation results."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.logger = logging.getLogger(__name__)
    
    def _prepare_data(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert results to analysis format."""
        if not results:
            return {}
        
        # Basic statistics
        total_results = len(results)
        successful_results = [r for r in results if not r.get("error")]
        
        if not successful_results:
            return {"total_results": total_results, "successful_results": 0}
        
        safety_scores = [r.get("safety_score", 0) for r in successful_results]
        
        analysis = {
            "total_results": total_results,
            "successful_results": len(successful_results),
            "average_safety_score": sum(safety_scores) / len(safety_scores) if safety_scores else 0,
            "min_safety_score": min(safety_scores) if safety_scores else 0,
            "max_safety_score": max(safety_scores) if safety_scores else 0
        }
        
        # Language breakdown
        language_scores = {}
        for result in successful_results:
            lang = result.get("language", "unknown")
            if lang not in language_scores:
                language_scores[lang] = []
            language_scores[lang].append(result.get("safety_score", 0))
        
        analysis["language_analysis"] = {
            lang: {
                "count": len(scores),
                "average": sum(scores) / len(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0
            }
            for lang, scores in language_scores.items()
        }
        
        # Domain breakdown
        domain_scores = {}
        for result in successful_results:
            domain = result.get("domain", "unknown")
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(result.get("safety_score", 0))
        
        analysis["domain_analysis"] = {
            domain: {
                "count": len(scores),
                "average": sum(scores) / len(scores) if scores else 0
            }
            for domain, scores in domain_scores.items()
        }
        
        return analysis
    
    def create_text_summary(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """Create a text-based summary report."""
        analysis = self._prepare_data(results)
        
        if not analysis:
            summary = "No data available for analysis."
        else:
            summary_lines = [
                "=== SafeLLM Multilingual Evaluation Summary ===",
                "",
                f"Total Evaluations: {analysis['total_results']}",
                f"Successful: {analysis['successful_results']}",
                f"Failed: {analysis['total_results'] - analysis['successful_results']}",
                ""
            ]
            
            if analysis.get("successful_results", 0) > 0:
                summary_lines.extend([
                    f"Average Safety Score: {analysis['average_safety_score']:.3f}",
                    f"Score Range: {analysis['min_safety_score']:.3f} - {analysis['max_safety_score']:.3f}",
                    "",
                    "=== Language Analysis ===",
                ])
                
                for lang, stats in analysis.get("language_analysis", {}).items():
                    summary_lines.append(f"{lang}: {stats['average']:.3f} (n={stats['count']})")
                
                summary_lines.extend([
                    "",
                    "=== Domain Analysis ===",
                ])
                
                for domain, stats in analysis.get("domain_analysis", {}).items():
                    summary_lines.append(f"{domain}: {stats['average']:.3f} (n={stats['count']})")
            
            summary_lines.extend([
                "",
                "=== End of Report ===",
            ])
            
            summary = "\n".join(summary_lines)
        
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                self.logger.info(f"Text summary saved to {output_path}")
                return output_path
            except Exception as e:
                self.logger.error(f"Error saving text summary: {e}")
                return "error_saving"
        
        return summary
    
    def create_json_summary(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """Create a JSON summary report."""
        analysis = self._prepare_data(results)
        
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False)
                self.logger.info(f"JSON summary saved to {output_path}")
                return output_path
            except Exception as e:
                self.logger.error(f"Error saving JSON summary: {e}")
                return "error_saving"
        
        return json.dumps(analysis, indent=2, ensure_ascii=False)
    
    def create_summary_report(self, results: List[Dict[str, Any]], 
                            output_dir: str = "./visualizations/") -> Dict[str, str]:
        """Create comprehensive summary report."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for unique filenames
            import time
            timestamp = int(time.time())
            
            # Create text summary
            text_path = os.path.join(output_dir, f"summary_report_{timestamp}.txt")
            text_result = self.create_text_summary(results, text_path)
            
            # Create JSON summary  
            json_path = os.path.join(output_dir, f"summary_data_{timestamp}.json")
            json_result = self.create_json_summary(results, json_path)
            
            self.logger.info(f"Summary reports generated in {output_dir}")
            
            return {
                "text_summary": text_result,
                "json_summary": json_result,
                "output_directory": output_dir
            }
            
        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")
            return {"error": str(e)}
    
    def create_safety_overview(self, results: List[Dict[str, Any]], 
                              output_path: Optional[str] = None) -> str:
        """Create safety overview (text-based for now)."""
        return self.create_text_summary(results, output_path)
    
    def create_language_comparison(self, results: List[Dict[str, Any]], 
                                  output_path: Optional[str] = None) -> str:
        """Create language comparison (text-based for now).""" 
        return self.create_text_summary(results, output_path)
    
    def create_model_comparison(self, results: List[Dict[str, Any]], 
                               output_path: Optional[str] = None) -> str:
        """Create model comparison (text-based for now)."""
        return self.create_text_summary(results, output_path)