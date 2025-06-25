"""
Main Pipeline Orchestrator

This module provides the main PromptEngineeringPipeline class that orchestrates
the entire multi-path reasoning pipeline with Tree-of-Thought, Self-Consistency,
and automated prompt optimization.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .llm_interface import LLMManager, LLMRequest, LLMResponse, create_llm_manager_from_config
from .tot_reasoning import TreeOfThought, WeightedScoringEvaluator, ConfidenceBasedEvaluator
from .self_consistency import SelfConsistency, WeightedConsensusAggregator, MajorityVoteAggregator
from .prompt_optimizer import (
    PromptOptimizer, PromptEvaluator, PromptCandidate, OptimizationResult,
    OPROOptimizer, TextGradOptimizer, OptimizationStrategy
)
from .utils import (
    setup_logging, load_config, save_results, calculate_metrics, 
    format_duration, create_unique_id, ensure_directory
)


class PromptEngineeringPipeline:
    """Main pipeline for multi-path reasoning with automated optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """Initialize the pipeline with configuration."""
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = self._get_default_config()
        
        # Setup logging
        log_config = self.config.get("logging", {})
        self.logger = setup_logging(
            level=log_config.get("level", "INFO"),
            log_to_file=log_config.get("log_to_file", True),
            log_file=log_config.get("log_file", "logs/pipeline.log")
        )
        
        # Initialize components
        self.llm_manager = None
        self.tot_reasoning = None
        self.self_consistency = None
        self.prompt_optimizer = None
        self.prompt_evaluator = None
        
        # Initialize pipeline
        self._initialize_pipeline()
        
        # Pipeline state
        self.session_id = create_unique_id("session")
        self.results_history = []
        
        self.logger.info(f"Pipeline initialized with session ID: {self.session_id}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "llm_interfaces": {
                "primary": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "api_key": "${OPENAI_API_KEY}",
                    "default": True
                }
            },
            "tot_config": {
                "num_paths": 5,
                "max_depth": 4,
                "branching_factor": 3,
                "pruning_threshold": 0.3,
                "evaluation_strategy": "weighted_scoring"
            },
            "consistency_config": {
                "aggregation_method": "weighted_consensus",
                "similarity_threshold": 0.8,
                "min_agreement": 0.6,
                "confidence_threshold": 0.7
            },
            "optimization_config": {
                "strategy": "opro",
                "max_iterations": 3,
                "improvement_threshold": 0.02,
                "validation_split": 0.2
            },
            "logging": {
                "level": "INFO",
                "log_to_file": True,
                "log_file": "logs/pipeline.log"
            }
        }
    
    def _initialize_pipeline(self):
        """Initialize all pipeline components."""
        # Initialize LLM manager
        self.llm_manager = create_llm_manager_from_config(self.config)
        
        # Initialize Tree-of-Thought
        tot_config = self.config.get("tot_config", {})
        evaluator_strategy = tot_config.get("evaluation_strategy", "weighted_scoring")
        
        if evaluator_strategy == "weighted_scoring":
            evaluator = WeightedScoringEvaluator(self.llm_manager.get_interface())
        elif evaluator_strategy == "confidence_based":
            evaluator = ConfidenceBasedEvaluator(self.llm_manager.get_interface())
        else:
            evaluator = WeightedScoringEvaluator(self.llm_manager.get_interface())
        
        self.tot_reasoning = TreeOfThought(
            llm_interface=self.llm_manager.get_interface(),
            evaluator=evaluator,
            config=tot_config
        )
        
        # Initialize Self-Consistency
        consistency_config = self.config.get("consistency_config", {})
        aggregation_method = consistency_config.get("aggregation_method", "weighted_consensus")
        
        if aggregation_method == "majority_vote":
            aggregator = MajorityVoteAggregator(
                similarity_threshold=consistency_config.get("similarity_threshold", 0.8)
            )
        else:
            aggregator = WeightedConsensusAggregator(
                similarity_threshold=consistency_config.get("similarity_threshold", 0.8),
                min_consensus=consistency_config.get("min_agreement", 0.6)
            )
        
        self.self_consistency = SelfConsistency(
            aggregator=aggregator,
            config=consistency_config
        )
        
        # Initialize prompt optimization components
        self.prompt_evaluator = PromptEvaluator(
            llm_interface=self.llm_manager.get_interface(),
            consistency_aggregator=self.self_consistency
        )
        
        optimization_config = self.config.get("optimization_config", {})
        strategy = optimization_config.get("strategy", "opro")
        
        if strategy == "opro":
            self.prompt_optimizer = OPROOptimizer(
                llm_interface=self.llm_manager.get_interface(),
                evaluator=self.prompt_evaluator
            )
        elif strategy == "textgrad":
            self.prompt_optimizer = TextGradOptimizer(
                llm_interface=self.llm_manager.get_interface(),
                evaluator=self.prompt_evaluator
            )
        else:
            self.prompt_optimizer = OPROOptimizer(
                llm_interface=self.llm_manager.get_interface(),
                evaluator=self.prompt_evaluator
            )
    
    async def run_single_problem(self, problem: str, num_paths: Optional[int] = None) -> Dict[str, Any]:
        """Run the pipeline on a single problem."""
        start_time = time.time()
        run_id = create_unique_id("run")
        
        self.logger.info(f"Starting single problem run {run_id}: {problem[:100]}...")
        
        # Use configured number of paths or default
        if num_paths is None:
            num_paths = self.config.get("tot_config", {}).get("num_paths", 5)
        
        try:
            # Generate reasoning paths using Tree-of-Thought
            self.logger.info("Generating reasoning paths with Tree-of-Thought")
            reasoning_paths = await self.tot_reasoning.reason(problem, num_paths)
            
            if not reasoning_paths:
                return {
                    "run_id": run_id,
                    "problem": problem,
                    "success": False,
                    "error": "No reasoning paths generated",
                    "runtime": time.time() - start_time
                }
            
            self.logger.info(f"Generated {len(reasoning_paths)} reasoning paths")
            
            # Aggregate paths using Self-Consistency
            self.logger.info("Aggregating paths with Self-Consistency")
            aggregation_result = await self.self_consistency.aggregate_paths(reasoning_paths, problem)
            
            # Compile results
            total_time = time.time() - start_time
            
            result = {
                "run_id": run_id,
                "session_id": self.session_id,
                "problem": problem,
                "success": True,
                "final_answer": aggregation_result["final_answer"],
                "confidence": aggregation_result["confidence"],
                "num_reasoning_paths": len(reasoning_paths),
                "aggregation_method": aggregation_result["method"],
                "agreement_ratio": aggregation_result.get("agreement_ratio", 0.0),
                "consistency_metrics": aggregation_result.get("consistency_metrics", {}),
                "reasoning_paths": [
                    {
                        "path_id": path.id,
                        "reasoning_steps": path.reasoning_steps,
                        "final_answer": path.final_answer,
                        "total_score": path.total_score,
                        "average_confidence": path.average_confidence,
                        "is_complete": path.is_complete,
                        "metadata": path.metadata
                    }
                    for path in reasoning_paths
                ],
                "tree_statistics": self.tot_reasoning.get_tree_statistics(),
                "aggregation_details": aggregation_result,
                "runtime": total_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results_history.append(result)
            self.logger.info(f"Single problem run completed in {format_duration(total_time)}")
            
            return result
            
        except Exception as e:
            error_result = {
                "run_id": run_id,
                "problem": problem,
                "success": False,
                "error": str(e),
                "runtime": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.error(f"Error in single problem run: {e}")
            return error_result
    
    async def run_batch(self, problems: List[Dict[str, Any]], 
                       evaluate: bool = True) -> Dict[str, Any]:
        """Run the pipeline on a batch of problems."""
        start_time = time.time()
        batch_id = create_unique_id("batch")
        
        self.logger.info(f"Starting batch run {batch_id} with {len(problems)} problems")
        
        results = []
        successful_runs = 0
        total_runtime = 0.0
        
        for i, problem_data in enumerate(problems):
            problem = problem_data.get("problem", "") if isinstance(problem_data, dict) else str(problem_data)
            
            self.logger.info(f"Processing problem {i+1}/{len(problems)}")
            
            try:
                result = await self.run_single_problem(problem)
                results.append(result)
                
                if result["success"]:
                    successful_runs += 1
                
                total_runtime += result["runtime"]
                
            except Exception as e:
                self.logger.error(f"Error processing problem {i+1}: {e}")
                results.append({
                    "problem": problem,
                    "success": False,
                    "error": str(e),
                    "runtime": 0.0
                })
        
        # Calculate batch metrics
        batch_runtime = time.time() - start_time
        success_rate = successful_runs / len(problems) if problems else 0.0
        
        batch_result = {
            "batch_id": batch_id,
            "session_id": self.session_id,
            "total_problems": len(problems),
            "successful_runs": successful_runs,
            "success_rate": success_rate,
            "total_runtime": batch_runtime,
            "average_runtime_per_problem": total_runtime / len(problems) if problems else 0.0,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add evaluation metrics if requested
        if evaluate and successful_runs > 0:
            batch_result["evaluation_metrics"] = self._calculate_batch_metrics(problems, results)
        
        self.logger.info(f"Batch run completed: {successful_runs}/{len(problems)} successful in {format_duration(batch_runtime)}")
        
        return batch_result
    
    async def run_with_optimization(self, problems: List[Dict[str, Any]], 
                                  initial_prompt: Optional[str] = None,
                                  max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Run the pipeline with automated prompt optimization."""
        start_time = time.time()
        optimization_id = create_unique_id("optimization")
        
        self.logger.info(f"Starting optimization run {optimization_id}")
        
        # Prepare prompt candidate
        if initial_prompt is None:
            initial_prompt = self._get_default_prompt()
        
        prompt_candidate = PromptCandidate(
            id=create_unique_id("prompt"),
            prompt=initial_prompt,
            generation=0
        )
        
        # Split problems into train and validation sets
        split_ratio = self.config.get("optimization_config", {}).get("validation_split", 0.2)
        split_index = int(len(problems) * (1 - split_ratio))
        
        train_problems = problems[:split_index]
        val_problems = problems[split_index:]
        
        if not val_problems:
            val_problems = train_problems[:min(3, len(train_problems))]
        
        self.logger.info(f"Using {len(train_problems)} training and {len(val_problems)} validation problems")
        
        # Configure optimization
        optimization_config = self.config.get("optimization_config", {}).copy()
        if max_iterations is not None:
            optimization_config["max_iterations"] = max_iterations
        
        try:
            # Run optimization
            self.logger.info("Starting prompt optimization")
            optimization_result = await self.prompt_optimizer.optimize(
                initial_prompt=prompt_candidate,
                train_problems=train_problems,
                val_problems=val_problems,
                config=optimization_config
            )
            
            self.logger.info(f"Optimization completed with {optimization_result.total_iterations} iterations")
            
            # Test optimized prompt on full problem set
            self.logger.info("Testing optimized prompt on full problem set")
            final_batch_result = await self._run_batch_with_prompt(
                problems, optimization_result.best_prompt
            )
            
            # Compile optimization results
            total_time = time.time() - start_time
            
            result = {
                "optimization_id": optimization_id,
                "session_id": self.session_id,
                "total_problems": len(problems),
                "train_problems": len(train_problems),
                "val_problems": len(val_problems),
                "optimization_strategy": self.prompt_optimizer.get_strategy_name(),
                "initial_prompt": initial_prompt,
                "optimized_prompt": optimization_result.best_prompt.prompt,
                "optimization_iterations": optimization_result.total_iterations,
                "optimization_time": optimization_result.total_time,
                "improvement_ratio": optimization_result.improvement_ratio,
                "convergence_metrics": optimization_result.convergence_metrics,
                "final_metrics": optimization_result.final_metrics,
                "optimization_history": [
                    {
                        "generation": candidate.generation,
                        "prompt": candidate.prompt,
                        "performance_metrics": candidate.performance_metrics,
                        "overall_score": candidate.get_overall_score()
                    }
                    for candidate in optimization_result.optimization_history
                ],
                "final_batch_result": final_batch_result,
                "total_runtime": total_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Optimization run completed in {format_duration(total_time)}")
            return result
            
        except Exception as e:
            error_result = {
                "optimization_id": optimization_id,
                "success": False,
                "error": str(e),
                "runtime": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.error(f"Error in optimization run: {e}")
            return error_result
    
    async def _run_batch_with_prompt(self, problems: List[Dict[str, Any]], 
                                   prompt_candidate: PromptCandidate) -> Dict[str, Any]:
        """Run batch with specific prompt candidate."""
        # This is a simplified version - in practice, you'd integrate the prompt
        # more deeply into the ToT reasoning process
        return await self.run_batch(problems, evaluate=True)
    
    def _get_default_prompt(self) -> str:
        """Get default reasoning prompt."""
        return """
        Solve the following problem step by step. Think carefully about each step and show your reasoning clearly.
        
        For each step:
        1. Identify what you need to find
        2. Determine what information you have
        3. Choose an appropriate method or approach
        4. Execute the method step by step
        5. Check your work and verify the answer
        
        Provide your final answer clearly at the end.
        """
    
    def _calculate_batch_metrics(self, problems: List[Dict[str, Any]], 
                               results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate evaluation metrics for batch results."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {"success_rate": 0.0}
        
        # Calculate basic metrics
        metrics = {
            "success_rate": len(successful_results) / len(results),
            "average_confidence": sum(r.get("confidence", 0.0) for r in successful_results) / len(successful_results),
            "average_agreement_ratio": sum(r.get("agreement_ratio", 0.0) for r in successful_results) / len(successful_results),
            "average_reasoning_paths": sum(r.get("num_reasoning_paths", 0) for r in successful_results) / len(successful_results)
        }
        
        # Calculate accuracy if expected answers are provided
        if all(isinstance(p, dict) and "expected" in p for p in problems):
            correct_count = 0
            total_evaluated = 0
            
            for i, result in enumerate(successful_results):
                if i < len(problems) and "expected" in problems[i]:
                    expected = problems[i]["expected"]
                    predicted = result.get("final_answer", "")
                    
                    # Simple accuracy check - could be enhanced
                    if self._check_answer_correctness(predicted, expected):
                        correct_count += 1
                    total_evaluated += 1
            
            if total_evaluated > 0:
                metrics["accuracy"] = correct_count / total_evaluated
        
        return metrics
    
    def _check_answer_correctness(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected answer."""
        # Simple exact match - could be enhanced with more sophisticated matching
        return predicted.strip().lower() == expected.strip().lower()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics."""
        return {
            "session_id": self.session_id,
            "total_runs": len(self.results_history),
            "successful_runs": sum(1 for r in self.results_history if r.get("success", False)),
            "llm_usage_stats": self.llm_manager.get_combined_usage_stats(),
            "tot_config": self.config.get("tot_config", {}),
            "consistency_config": self.config.get("consistency_config", {}),
            "optimization_config": self.config.get("optimization_config", {}),
            "pipeline_components": {
                "tot_reasoning": self.tot_reasoning.__class__.__name__,
                "self_consistency": self.self_consistency.__class__.__name__,
                "prompt_optimizer": self.prompt_optimizer.__class__.__name__ if self.prompt_optimizer else None
            }
        }
    
    def save_session(self, output_path: str, include_full_history: bool = True) -> None:
        """Save current session data."""
        session_data = {
            "session_id": self.session_id,
            "config": self.config,
            "pipeline_status": self.get_pipeline_status(),
            "timestamp": datetime.now().isoformat()
        }
        
        if include_full_history:
            session_data["results_history"] = self.results_history
        
        save_results(session_data, output_path)
        self.logger.info(f"Session saved to {output_path}")
    
    def load_session(self, session_path: str) -> None:
        """Load session data from file."""
        from .utils import load_results
        
        session_data = load_results(session_path)
        
        if "session_id" in session_data:
            self.session_id = session_data["session_id"]
        
        if "results_history" in session_data:
            self.results_history = session_data["results_history"]
        
        self.logger.info(f"Session loaded from {session_path}")
    
    async def close(self):
        """Clean up pipeline resources."""
        # Close any open connections or resources
        self.logger.info(f"Closing pipeline session {self.session_id}")


# Command-line interface
async def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Engineering Pipeline")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--tasks", type=str, default="all", help="Task domain to run")
    parser.add_argument("--optimize", action="store_true", help="Enable prompt optimization")
    parser.add_argument("--iterations", type=int, default=3, help="Optimization iterations")
    parser.add_argument("--paths", type=int, default=5, help="Number of reasoning paths")
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    config = None
    if args.config:
        config = load_config(args.config)
    
    pipeline = PromptEngineeringPipeline(config=config)
    
    # Load tasks
    problems = []
    if args.tasks == "all" or args.tasks == "math":
        from ..tasks.math_word_problems import MathWordProblems
        math_tasks = MathWordProblems()
        problems.extend(math_tasks.get_problems())
    
    if args.tasks == "all" or args.tasks == "logic":
        from ..tasks.logic_puzzles import LogicPuzzles
        logic_tasks = LogicPuzzles()
        problems.extend(logic_tasks.get_problems())
    
    if args.tasks == "all" or args.tasks == "code":
        from ..tasks.code_debugging import CodeDebugging
        code_tasks = CodeDebugging()
        problems.extend(code_tasks.get_problems())
    
    # Run pipeline
    try:
        if args.optimize:
            print(f"Running optimization with {len(problems)} problems...")
            result = await pipeline.run_with_optimization(
                problems=problems,
                max_iterations=args.iterations
            )
        else:
            print(f"Running batch with {len(problems)} problems...")
            result = await pipeline.run_batch(problems)
        
        # Save results
        if args.output:
            output_path = args.output
        else:
            output_path = f"results/pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        ensure_directory(Path(output_path).parent)
        save_results(result, output_path)
        
        print(f"Results saved to {output_path}")
        
        # Print summary
        if args.optimize:
            print(f"Optimization completed:")
            print(f"  Iterations: {result.get('optimization_iterations', 0)}")
            print(f"  Improvement: {result.get('improvement_ratio', 0):.2%}")
            print(f"  Final metrics: {result.get('final_metrics', {})}")
        else:
            print(f"Batch completed:")
            print(f"  Success rate: {result.get('success_rate', 0):.2%}")
            print(f"  Total runtime: {format_duration(result.get('total_runtime', 0))}")
    
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())