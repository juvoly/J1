import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any
from .medqa import MedQAEvaluator
from .pubmedqa import PubMedQAEvaluator

EVALUATOR_MAP = {
    "medqa": MedQAEvaluator,
    "pubmedqa": PubMedQAEvaluator
}

async def main():
    # Load configuration
    config_path = Path("configs/medqa_eval.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Get evaluator class from config
    benchmark_type = config.get("benchmark_type")
    if not benchmark_type or benchmark_type not in EVALUATOR_MAP:
        raise ValueError(f"benchmark_type must be one of: {list(EVALUATOR_MAP.keys())}")
    
    evaluator_class = EVALUATOR_MAP[benchmark_type]
    
    # Create evaluators
    evaluators = [
        evaluator_class(
            label=model["label"],
            model=model["model"],
            api_base=model["api_base"],
            api_key=model["api_key"],
            max_concurrent_requests=model.get("max_concurrent_requests", 5),
            model_params=model.get("params", {}),
            use_chain_of_thought=model.get("use_chain_of_thought", False),
            custom_prompt=model.get("custom_prompt")
        )
        for model in config["models"]
    ]
    
    # Run evaluations
    results = {}
    for evaluator in evaluators:
        print(f"\nEvaluating {evaluator.label}...")
        results[evaluator.label] = await evaluator.evaluate_dataset(
            config["dataset_path"],
            limit=config.get("limit")
        )
    
    # Print results
    for label, result in results.items():
        print(f"\nResults for {label}:")
        print(f"Accuracy: {result['accuracy']:.2%}")
        print(f"Total tokens: {result['total_tokens']}")
        print(f"Total cost: ${result['total_cost']:.4f}")

if __name__ == "__main__":
    asyncio.run(main()) 