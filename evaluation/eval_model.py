
# eval_model.py - Updated script to query LLMs with modern APIs

import openai
import json
import os
from typing import Dict, Any, Optional

# Initialize OpenAI client with API key from environment
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")
)

def evaluate_prompt(prompt: str, model: str = "gpt-4") -> Dict[str, Any]:
    """
    Evaluate a prompt using OpenAI's API.
    
    Args:
        prompt: The input prompt to evaluate
        model: The model to use for evaluation
        
    Returns:
        Dictionary containing response and metadata
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        return {
            "response": response.choices[0].message.content,
            "model": model,
            "tokens_used": response.usage.total_tokens,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "response": None,
            "model": model,
            "tokens_used": 0,
            "success": False,
            "error": str(e)
        }

def load_prompts_from_jsonl(filepath: str) -> list:
    """Load prompts from JSONL file."""
    prompts = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    prompts.append(json.loads(line))
        return prompts
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return []

def save_results(results: list, output_path: str = "results/evaluation_results.jsonl"):
    """Save evaluation results to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Load prompts from dataset
    dataset_path = "datasets/sample_prompts.jsonl"
    prompts = load_prompts_from_jsonl(dataset_path)
    
    if not prompts:
        print(f"No prompts found in {dataset_path}")
        exit(1)
    
    print(f"Loaded {len(prompts)} prompts for evaluation")
    
    # Evaluate each prompt
    results = []
    for i, item in enumerate(prompts):
        print(f"\n--- Evaluating prompt {i+1}/{len(prompts)} ---")
        print(f"Language: {item.get('language', 'unknown')}")
        print(f"Domain: {item.get('domain', 'unknown')}")
        print(f"Prompt: {item.get('prompt', '')[:100]}...")
        
        # Evaluate the prompt
        result = evaluate_prompt(item.get('prompt', ''))
        
        # Combine original metadata with evaluation result
        combined_result = {
            **item,  # Original prompt metadata
            **result  # Evaluation results
        }
        
        results.append(combined_result)
        
        if result['success']:
            print(f"✅ Success - {result['tokens_used']} tokens used")
            print(f"Response: {result['response'][:150]}...")
        else:
            print(f"❌ Error: {result['error']}")
    
    # Save results
    save_results(results)
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n=== Evaluation Summary ===")
    print(f"Total prompts: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
