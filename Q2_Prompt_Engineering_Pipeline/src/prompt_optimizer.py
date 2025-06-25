"""
Prompt Optimizer Module

This module implements automated prompt optimization using OPRO (Optimization by PROmpting)
and TextGrad-style approaches for iterative prompt improvement.
"""

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .llm_interface import LLMInterface, LLMRequest, LLMResponse
from .tot_reasoning import ReasoningPath
from .self_consistency import SelfConsistency


class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""
    OPRO = "opro"
    TEXTGRAD = "textgrad"
    EVOLUTIONARY = "evolutionary"
    HILL_CLIMBING = "hill_climbing"
    RANDOM_SEARCH = "random_search"


class OptimizationObjective(Enum):
    """Enumeration of optimization objectives."""
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    CONFIDENCE = "confidence"
    REASONING_QUALITY = "reasoning_quality"
    EFFICIENCY = "efficiency"


@dataclass
class PromptCandidate:
    """Represents a prompt candidate with its performance metrics."""
    id: str
    prompt: str
    system_prompt: Optional[str] = None
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_id: Optional[str] = None
    mutation_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def get_overall_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted overall score."""
        if not self.performance_metrics:
            return 0.0
        
        if weights is None:
            weights = {metric: 1.0 for metric in self.performance_metrics.keys()}
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in self.performance_metrics.items():
            weight = weights.get(metric, 0.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


@dataclass
class OptimizationResult:
    """Results from prompt optimization."""
    best_prompt: PromptCandidate
    optimization_history: List[PromptCandidate]
    convergence_metrics: Dict[str, List[float]]
    total_iterations: int
    total_time: float
    final_metrics: Dict[str, float]
    improvement_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptEvaluator:
    """Evaluates prompt performance on given tasks."""
    
    def __init__(self, llm_interface: LLMInterface, consistency_aggregator: SelfConsistency):
        self.llm_interface = llm_interface
        self.consistency_aggregator = consistency_aggregator
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def evaluate_prompt(self, prompt_candidate: PromptCandidate,
                            test_problems: List[Dict[str, Any]],
                            config: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate a prompt candidate on test problems."""
        config = config or {}
        num_paths = config.get("num_paths", 3)
        
        results = []
        total_time = 0.0
        
        for problem in test_problems:
            start_time = time.time()
            
            # Generate multiple reasoning paths
            reasoning_paths = []
            for _ in range(num_paths):
                response = await self._generate_response(prompt_candidate, problem, config)
                if response and not response.error:
                    path = self._create_reasoning_path(response, problem)
                    reasoning_paths.append(path)
            
            # Aggregate paths if multiple exist
            if reasoning_paths:
                aggregation_result = await self.consistency_aggregator.aggregate_paths(
                    reasoning_paths, problem.get("problem", "")
                )
                
                # Evaluate against expected answer
                evaluation = self._evaluate_answer(
                    aggregation_result["final_answer"],
                    problem.get("expected", ""),
                    problem
                )
                
                results.append({
                    "correct": evaluation["correct"],
                    "confidence": aggregation_result["confidence"],
                    "consistency": aggregation_result.get("agreement_ratio", 0.0),
                    "reasoning_quality": aggregation_result.get("consistency_metrics", {}).get("average_score", 0.0)
                })
            else:
                results.append({
                    "correct": False,
                    "confidence": 0.0,
                    "consistency": 0.0,
                    "reasoning_quality": 0.0
                })
            
            total_time += time.time() - start_time
        
        # Calculate aggregate metrics
        if not results:
            return {"accuracy": 0.0, "confidence": 0.0, "consistency": 0.0, "reasoning_quality": 0.0, "efficiency": 0.0}
        
        accuracy = sum(r["correct"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_consistency = sum(r["consistency"] for r in results) / len(results)
        avg_reasoning_quality = sum(r["reasoning_quality"] for r in results) / len(results)
        efficiency = len(results) / total_time if total_time > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "confidence": avg_confidence,
            "consistency": avg_consistency,
            "reasoning_quality": avg_reasoning_quality,
            "efficiency": efficiency
        }
    
    async def _generate_response(self, prompt_candidate: PromptCandidate,
                               problem: Dict[str, Any], config: Dict[str, Any]) -> Optional[LLMResponse]:
        """Generate response using the prompt candidate."""
        # Format the prompt
        formatted_prompt = self._format_prompt(prompt_candidate, problem)
        
        request = LLMRequest(
            prompt=formatted_prompt,
            system_prompt=prompt_candidate.system_prompt,
            model=config.get("model", "gpt-3.5-turbo"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000)
        )
        
        try:
            response = await self.llm_interface.generate(request)
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return None
    
    def _format_prompt(self, prompt_candidate: PromptCandidate, problem: Dict[str, Any]) -> str:
        """Format the prompt with few-shot examples and problem."""
        formatted_parts = []
        
        # Add base prompt
        formatted_parts.append(prompt_candidate.prompt)
        
        # Add few-shot examples
        if prompt_candidate.few_shot_examples:
            formatted_parts.append("\nHere are some examples:")
            for i, example in enumerate(prompt_candidate.few_shot_examples):
                formatted_parts.append(f"\nExample {i+1}:")
                formatted_parts.append(f"Problem: {example.get('problem', '')}")
                formatted_parts.append(f"Solution: {example.get('solution', '')}")
        
        # Add current problem
        formatted_parts.append(f"\nNow solve this problem:")
        formatted_parts.append(f"Problem: {problem.get('problem', '')}")
        formatted_parts.append("Solution:")
        
        return "\n".join(formatted_parts)
    
    def _create_reasoning_path(self, response: LLMResponse, problem: Dict[str, Any]) -> ReasoningPath:
        """Create a reasoning path from LLM response."""
        from .tot_reasoning import ReasoningPath
        import uuid
        
        path = ReasoningPath(
            id=str(uuid.uuid4()),
            node_ids=[],
            total_score=0.8,  # Default score
            average_confidence=0.7,  # Default confidence
            is_complete=True,
            final_answer=response.content,
            reasoning_steps=[response.content],
            metadata={"response_time": response.latency}
        )
        
        return path
    
    def _evaluate_answer(self, predicted: str, expected: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate predicted answer against expected answer."""
        # Simple string similarity for now - could be enhanced with task-specific evaluation
        predicted_clean = predicted.strip().lower()
        expected_clean = expected.strip().lower()
        
        # Check for exact match
        exact_match = predicted_clean == expected_clean
        
        # Check for numerical match
        numerical_match = self._check_numerical_match(predicted, expected)
        
        # Check for semantic similarity (basic)
        semantic_match = self._check_semantic_match(predicted_clean, expected_clean)
        
        correct = exact_match or numerical_match or semantic_match
        
        return {
            "correct": correct,
            "exact_match": exact_match,
            "numerical_match": numerical_match,
            "semantic_match": semantic_match
        }
    
    def _check_numerical_match(self, predicted: str, expected: str) -> bool:
        """Check if numerical values match."""
        import re
        
        pred_numbers = re.findall(r'-?\d+\.?\d*', predicted)
        exp_numbers = re.findall(r'-?\d+\.?\d*', expected)
        
        if not pred_numbers or not exp_numbers:
            return False
        
        try:
            pred_nums = [float(n) for n in pred_numbers]
            exp_nums = [float(n) for n in exp_numbers]
            
            # Check if any predicted number matches any expected number
            for pred_num in pred_nums:
                for exp_num in exp_nums:
                    if abs(pred_num - exp_num) < 1e-6:
                        return True
        except ValueError:
            pass
        
        return False
    
    def _check_semantic_match(self, predicted: str, expected: str) -> bool:
        """Check for basic semantic similarity."""
        # Simple word overlap check
        pred_words = set(predicted.split())
        exp_words = set(expected.split())
        
        if not pred_words or not exp_words:
            return False
        
        intersection = pred_words.intersection(exp_words)
        union = pred_words.union(exp_words)
        
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity >= 0.5


class PromptOptimizer(ABC):
    """Abstract base class for prompt optimizers."""
    
    def __init__(self, llm_interface: LLMInterface, evaluator: PromptEvaluator):
        self.llm_interface = llm_interface
        self.evaluator = evaluator
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def optimize(self, initial_prompt: PromptCandidate,
                      train_problems: List[Dict[str, Any]],
                      val_problems: List[Dict[str, Any]],
                      config: Dict[str, Any]) -> OptimizationResult:
        """Optimize the prompt using the specific strategy."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this optimization strategy."""
        pass


class OPROOptimizer(PromptOptimizer):
    """OPRO-style optimizer that uses LLM to improve prompts."""
    
    def __init__(self, llm_interface: LLMInterface, evaluator: PromptEvaluator):
        super().__init__(llm_interface, evaluator)
        self.optimization_prompt_template = """
You are an expert prompt engineer. Your task is to improve the given prompt to achieve better performance on reasoning tasks.

Current prompt performance:
Accuracy: {accuracy:.2f}
Confidence: {confidence:.2f}
Consistency: {consistency:.2f}

Current prompt:
{current_prompt}

Failed examples:
{failed_examples}

Please provide an improved version of this prompt that:
1. Addresses the specific failure modes shown in the failed examples
2. Maintains the core reasoning approach
3. Is clearer and more specific in its instructions
4. Helps the model provide more accurate and consistent responses

Improved prompt:
"""
    
    async def optimize(self, initial_prompt: PromptCandidate,
                      train_problems: List[Dict[str, Any]],
                      val_problems: List[Dict[str, Any]],
                      config: Dict[str, Any]) -> OptimizationResult:
        """Optimize using OPRO approach."""
        max_iterations = config.get("max_iterations", 5)
        improvement_threshold = config.get("improvement_threshold", 0.02)
        
        history = []
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = 0.0
        
        convergence_metrics = {
            "accuracy": [],
            "confidence": [],
            "consistency": [],
            "reasoning_quality": []
        }
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            self.logger.info(f"OPRO iteration {iteration + 1}/{max_iterations}")
            
            # Evaluate current prompt
            metrics = await self.evaluator.evaluate_prompt(current_prompt, val_problems, config)
            current_prompt.performance_metrics = metrics
            history.append(current_prompt)
            
            # Update convergence metrics
            for metric, value in metrics.items():
                if metric in convergence_metrics:
                    convergence_metrics[metric].append(value)
            
            # Check if this is the best prompt so far
            current_score = current_prompt.get_overall_score()
            if current_score > best_score:
                best_prompt = current_prompt
                best_score = current_score
                self.logger.info(f"New best score: {best_score:.3f}")
            
            # Check convergence
            if iteration > 0 and abs(current_score - history[-2].get_overall_score()) < improvement_threshold:
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            # Generate improved prompt
            if iteration < max_iterations - 1:  # Don't generate on last iteration
                current_prompt = await self._generate_improved_prompt(
                    current_prompt, train_problems, config
                )
                current_prompt.generation = iteration + 1
                current_prompt.parent_id = history[-1].id
        
        total_time = time.time() - start_time
        
        return OptimizationResult(
            best_prompt=best_prompt,
            optimization_history=history,
            convergence_metrics=convergence_metrics,
            total_iterations=len(history),
            total_time=total_time,
            final_metrics=best_prompt.performance_metrics,
            improvement_ratio=best_score / initial_prompt.get_overall_score() if initial_prompt.get_overall_score() > 0 else 1.0,
            metadata={"strategy": "opro", "converged": len(history) < max_iterations}
        )
    
    async def _generate_improved_prompt(self, current_prompt: PromptCandidate,
                                      train_problems: List[Dict[str, Any]],
                                      config: Dict[str, Any]) -> PromptCandidate:
        """Generate improved prompt using OPRO approach."""
        # Identify failed examples
        failed_examples = await self._identify_failed_examples(current_prompt, train_problems, config)
        
        # Create optimization prompt
        optimization_prompt = self.optimization_prompt_template.format(
            accuracy=current_prompt.performance_metrics.get("accuracy", 0.0),
            confidence=current_prompt.performance_metrics.get("confidence", 0.0),
            consistency=current_prompt.performance_metrics.get("consistency", 0.0),
            current_prompt=current_prompt.prompt,
            failed_examples=self._format_failed_examples(failed_examples)
        )
        
        # Generate improved prompt
        request = LLMRequest(
            prompt=optimization_prompt,
            model=config.get("meta_model", "gpt-4"),
            temperature=0.3,
            max_tokens=1000
        )
        
        try:
            response = await self.llm_interface.generate(request)
            
            if response.error:
                self.logger.warning(f"Failed to generate improved prompt: {response.error}")
                return self._create_mutated_prompt(current_prompt)
            
            # Extract improved prompt
            improved_prompt_text = self._extract_improved_prompt(response.content)
            
            # Create new prompt candidate
            import uuid
            new_prompt = PromptCandidate(
                id=str(uuid.uuid4()),
                prompt=improved_prompt_text,
                system_prompt=current_prompt.system_prompt,
                few_shot_examples=current_prompt.few_shot_examples.copy(),
                generation=current_prompt.generation + 1,
                parent_id=current_prompt.id,
                mutation_type="opro_improvement",
                metadata={"optimization_method": "opro"}
            )
            
            return new_prompt
            
        except Exception as e:
            self.logger.error(f"Error generating improved prompt: {e}")
            return self._create_mutated_prompt(current_prompt)
    
    async def _identify_failed_examples(self, prompt: PromptCandidate,
                                      problems: List[Dict[str, Any]],
                                      config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify examples where the current prompt fails."""
        failed = []
        
        for problem in problems[:5]:  # Limit to first 5 problems for efficiency
            response = await self.evaluator._generate_response(prompt, problem, config)
            if response and not response.error:
                evaluation = self.evaluator._evaluate_answer(
                    response.content, problem.get("expected", ""), problem
                )
                
                if not evaluation["correct"]:
                    failed.append({
                        "problem": problem.get("problem", ""),
                        "expected": problem.get("expected", ""),
                        "predicted": response.content,
                        "evaluation": evaluation
                    })
        
        return failed
    
    def _format_failed_examples(self, failed_examples: List[Dict[str, Any]]) -> str:
        """Format failed examples for the optimization prompt."""
        if not failed_examples:
            return "No failed examples identified."
        
        formatted = []
        for i, example in enumerate(failed_examples[:3]):  # Limit to 3 examples
            formatted.append(f"Failed Example {i+1}:")
            formatted.append(f"Problem: {example['problem']}")
            formatted.append(f"Expected: {example['expected']}")
            formatted.append(f"Predicted: {example['predicted']}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _extract_improved_prompt(self, response_content: str) -> str:
        """Extract the improved prompt from the response."""
        lines = response_content.strip().split('\n')
        
        # Look for the improved prompt after "Improved prompt:"
        start_idx = -1
        for i, line in enumerate(lines):
            if "improved prompt:" in line.lower():
                start_idx = i + 1
                break
        
        if start_idx == -1:
            # If no explicit marker, use the entire response
            return response_content.strip()
        
        # Extract everything after the marker
        improved_lines = lines[start_idx:]
        return '\n'.join(improved_lines).strip()
    
    def _create_mutated_prompt(self, current_prompt: PromptCandidate) -> PromptCandidate:
        """Create a mutated version of the current prompt as fallback."""
        import uuid
        
        # Simple mutation: add emphasis or clarification
        mutations = [
            "Please think step by step and show your reasoning clearly.",
            "Make sure to double-check your answer before providing it.",
            "Consider multiple approaches and choose the most appropriate one.",
            "Explain your reasoning process in detail."
        ]
        
        selected_mutation = random.choice(mutations)
        mutated_prompt = f"{current_prompt.prompt}\n\n{selected_mutation}"
        
        return PromptCandidate(
            id=str(uuid.uuid4()),
            prompt=mutated_prompt,
            system_prompt=current_prompt.system_prompt,
            few_shot_examples=current_prompt.few_shot_examples.copy(),
            generation=current_prompt.generation + 1,
            parent_id=current_prompt.id,
            mutation_type="random_mutation",
            metadata={"mutation": selected_mutation}
        )
    
    def get_strategy_name(self) -> str:
        return "opro"


class TextGradOptimizer(PromptOptimizer):
    """TextGrad-style optimizer that treats prompts as differentiable parameters."""
    
    def __init__(self, llm_interface: LLMInterface, evaluator: PromptEvaluator):
        super().__init__(llm_interface, evaluator)
        self.gradient_prompt_template = """
You are analyzing the performance of a prompt for reasoning tasks. Based on the feedback, provide specific suggestions for improving the prompt.

Current prompt:
{current_prompt}

Performance feedback:
{feedback}

Please provide specific, actionable modifications to improve the prompt. Focus on:
1. Clarity of instructions
2. Better reasoning guidance
3. Addressing specific failure modes
4. Improving consistency

Suggestions for improvement:
"""
    
    async def optimize(self, initial_prompt: PromptCandidate,
                      train_problems: List[Dict[str, Any]],
                      val_problems: List[Dict[str, Any]],
                      config: Dict[str, Any]) -> OptimizationResult:
        """Optimize using TextGrad approach."""
        max_iterations = config.get("max_iterations", 5)
        learning_rate = config.get("learning_rate", 0.1)
        
        history = []
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = 0.0
        
        convergence_metrics = {
            "accuracy": [],
            "confidence": [],
            "consistency": [],
            "reasoning_quality": []
        }
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            self.logger.info(f"TextGrad iteration {iteration + 1}/{max_iterations}")
            
            # Evaluate current prompt
            metrics = await self.evaluator.evaluate_prompt(current_prompt, val_problems, config)
            current_prompt.performance_metrics = metrics
            history.append(current_prompt)
            
            # Update convergence metrics
            for metric, value in metrics.items():
                if metric in convergence_metrics:
                    convergence_metrics[metric].append(value)
            
            # Update best prompt
            current_score = current_prompt.get_overall_score()
            if current_score > best_score:
                best_prompt = current_prompt
                best_score = current_score
            
            # Generate gradient-based improvement
            if iteration < max_iterations - 1:
                current_prompt = await self._apply_gradient_update(
                    current_prompt, train_problems, config, learning_rate
                )
                current_prompt.generation = iteration + 1
                current_prompt.parent_id = history[-1].id
        
        total_time = time.time() - start_time
        
        return OptimizationResult(
            best_prompt=best_prompt,
            optimization_history=history,
            convergence_metrics=convergence_metrics,
            total_iterations=len(history),
            total_time=total_time,
            final_metrics=best_prompt.performance_metrics,
            improvement_ratio=best_score / initial_prompt.get_overall_score() if initial_prompt.get_overall_score() > 0 else 1.0,
            metadata={"strategy": "textgrad", "learning_rate": learning_rate}
        )
    
    async def _apply_gradient_update(self, current_prompt: PromptCandidate,
                                   train_problems: List[Dict[str, Any]],
                                   config: Dict[str, Any],
                                   learning_rate: float) -> PromptCandidate:
        """Apply gradient-like update to the prompt."""
        # Compute "gradient" by analyzing performance feedback
        feedback = await self._compute_performance_feedback(current_prompt, train_problems, config)
        
        # Generate improvement suggestions
        gradient_prompt = self.gradient_prompt_template.format(
            current_prompt=current_prompt.prompt,
            feedback=feedback
        )
        
        request = LLMRequest(
            prompt=gradient_prompt,
            model=config.get("meta_model", "gpt-4"),
            temperature=0.3,
            max_tokens=800
        )
        
        try:
            response = await self.llm_interface.generate(request)
            
            if response.error:
                return self._create_random_variant(current_prompt)
            
            # Apply suggestions to create updated prompt
            updated_prompt = await self._apply_suggestions(
                current_prompt, response.content, learning_rate
            )
            
            return updated_prompt
            
        except Exception as e:
            self.logger.error(f"Error in gradient update: {e}")
            return self._create_random_variant(current_prompt)
    
    async def _compute_performance_feedback(self, prompt: PromptCandidate,
                                          problems: List[Dict[str, Any]],
                                          config: Dict[str, Any]) -> str:
        """Compute detailed performance feedback."""
        feedback_parts = []
        
        # Add performance metrics
        metrics = prompt.performance_metrics
        feedback_parts.append(f"Accuracy: {metrics.get('accuracy', 0.0):.2f}")
        feedback_parts.append(f"Confidence: {metrics.get('confidence', 0.0):.2f}")
        feedback_parts.append(f"Consistency: {metrics.get('consistency', 0.0):.2f}")
        
        # Add specific failure analysis
        failures = await self._analyze_failures(prompt, problems[:3], config)
        if failures:
            feedback_parts.append("\nFailure Analysis:")
            feedback_parts.extend(failures)
        
        return "\n".join(feedback_parts)
    
    async def _analyze_failures(self, prompt: PromptCandidate,
                              problems: List[Dict[str, Any]],
                              config: Dict[str, Any]) -> List[str]:
        """Analyze specific failure modes."""
        failures = []
        
        for problem in problems:
            response = await self.evaluator._generate_response(prompt, problem, config)
            if response and not response.error:
                evaluation = self.evaluator._evaluate_answer(
                    response.content, problem.get("expected", ""), problem
                )
                
                if not evaluation["correct"]:
                    failures.append(f"Failed on: {problem.get('problem', '')[:100]}...")
                    failures.append(f"Expected: {problem.get('expected', '')}")
                    failures.append(f"Got: {response.content[:100]}...")
        
        return failures
    
    async def _apply_suggestions(self, current_prompt: PromptCandidate,
                               suggestions: str, learning_rate: float) -> PromptCandidate:
        """Apply improvement suggestions to create updated prompt."""
        import uuid
        
        # Parse suggestions and modify prompt
        # This is a simplified implementation - in practice, you'd have more sophisticated
        # methods to apply textual "gradients"
        
        modification_prompt = f"""
Current prompt:
{current_prompt.prompt}

Improvement suggestions:
{suggestions}

Please provide an improved version of the prompt that incorporates these suggestions while maintaining the core structure and intent. Apply the suggestions with intensity {learning_rate:.2f} (where 1.0 means full application and 0.0 means no change).

Improved prompt:
"""
        
        request = LLMRequest(
            prompt=modification_prompt,
            model="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=1000
        )
        
        try:
            response = await self.llm_interface.generate(request)
            
            if response.error:
                return self._create_random_variant(current_prompt)
            
            # Extract improved prompt
            improved_text = response.content.strip()
            
            return PromptCandidate(
                id=str(uuid.uuid4()),
                prompt=improved_text,
                system_prompt=current_prompt.system_prompt,
                few_shot_examples=current_prompt.few_shot_examples.copy(),
                generation=current_prompt.generation + 1,
                parent_id=current_prompt.id,
                mutation_type="textgrad_update",
                metadata={"learning_rate": learning_rate}
            )
            
        except Exception as e:
            self.logger.error(f"Error applying suggestions: {e}")
            return self._create_random_variant(current_prompt)
    
    def _create_random_variant(self, current_prompt: PromptCandidate) -> PromptCandidate:
        """Create a random variant as fallback."""
        import uuid
        
        variants = [
            f"{current_prompt.prompt}\n\nPlease be extra careful with your reasoning.",
            f"Let's think about this step by step.\n\n{current_prompt.prompt}",
            f"{current_prompt.prompt}\n\nDouble-check your work before providing the final answer.",
            f"Take a deep breath and work through this problem methodically.\n\n{current_prompt.prompt}"
        ]
        
        selected_variant = random.choice(variants)
        
        return PromptCandidate(
            id=str(uuid.uuid4()),
            prompt=selected_variant,
            system_prompt=current_prompt.system_prompt,
            few_shot_examples=current_prompt.few_shot_examples.copy(),
            generation=current_prompt.generation + 1,
            parent_id=current_prompt.id,
            mutation_type="random_variant",
            metadata={"variant_type": "random"}
        )
    
    def get_strategy_name(self) -> str:
        return "textgrad"


def create_optimizer(strategy: OptimizationStrategy, llm_interface: LLMInterface,
                    evaluator: PromptEvaluator) -> PromptOptimizer:
    """Factory function to create optimizers."""
    if strategy == OptimizationStrategy.OPRO:
        return OPROOptimizer(llm_interface, evaluator)
    elif strategy == OptimizationStrategy.TEXTGRAD:
        return TextGradOptimizer(llm_interface, evaluator)
    else:
        raise ValueError(f"Unsupported optimization strategy: {strategy}")