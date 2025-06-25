"""
Tree-of-Thought Reasoning Module

This module implements the Tree-of-Thought (ToT) reasoning methodology,
allowing for structured exploration of multiple reasoning paths.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from .llm_interface import LLMInterface, LLMRequest, LLMResponse


class NodeState(Enum):
    """Enumeration of possible reasoning node states."""
    ACTIVE = "active"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationStrategy(Enum):
    """Enumeration of evaluation strategies for reasoning paths."""
    WEIGHTED_SCORING = "weighted_scoring"
    CONFIDENCE_BASED = "confidence_based"
    MAJORITY_VOTE = "majority_vote"
    EXPERT_HEURISTIC = "expert_heuristic"


@dataclass
class ReasoningNode:
    """Represents a single node in the reasoning tree."""
    id: str
    content: str
    depth: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    state: NodeState = NodeState.ACTIVE
    score: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def add_child(self, child_id: str):
        """Add a child node ID to this node."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children_ids) == 0
    
    def is_root(self) -> bool:
        """Check if this node is the root (has no parent)."""
        return self.parent_id is None


@dataclass
class ReasoningPath:
    """Represents a complete path from root to leaf in the reasoning tree."""
    id: str
    node_ids: List[str]
    total_score: float = 0.0
    average_confidence: float = 0.0
    is_complete: bool = False
    final_answer: Optional[str] = None
    reasoning_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node_id: str):
        """Add a node to this path."""
        self.node_ids.append(node_id)
    
    def get_depth(self) -> int:
        """Get the depth of this path."""
        return len(self.node_ids)


class NodeEvaluator(ABC):
    """Abstract base class for node evaluation strategies."""
    
    @abstractmethod
    async def evaluate_node(self, node: ReasoningNode, context: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate a node and return (score, confidence)."""
        pass
    
    @abstractmethod
    def get_evaluation_criteria(self) -> List[str]:
        """Get the criteria used for evaluation."""
        pass


class WeightedScoringEvaluator(NodeEvaluator):
    """Evaluator that uses weighted scoring across multiple criteria."""
    
    def __init__(self, llm_interface: LLMInterface, weights: Dict[str, float] = None):
        self.llm_interface = llm_interface
        self.weights = weights or {
            "logical_coherence": 0.3,
            "factual_accuracy": 0.25,
            "relevance": 0.2,
            "completeness": 0.15,
            "clarity": 0.1
        }
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def evaluate_node(self, node: ReasoningNode, context: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate a node using weighted scoring."""
        evaluation_prompt = self._create_evaluation_prompt(node, context)
        
        request = LLMRequest(
            prompt=evaluation_prompt,
            model=context.get("model", "gpt-3.5-turbo"),
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=500
        )
        
        try:
            response = await self.llm_interface.generate(request)
            
            if response.error:
                self.logger.warning(f"Evaluation failed for node {node.id}: {response.error}")
                return 0.5, 0.3  # Default moderate scores
            
            scores = self._parse_evaluation_response(response.content)
            weighted_score = sum(scores.get(criterion, 0.5) * weight 
                               for criterion, weight in self.weights.items())
            
            # Calculate confidence based on score consistency
            score_variance = np.var(list(scores.values()))
            confidence = max(0.1, 1.0 - score_variance)
            
            return weighted_score, confidence
            
        except Exception as e:
            self.logger.error(f"Error evaluating node {node.id}: {e}")
            return 0.5, 0.3
    
    def _create_evaluation_prompt(self, node: ReasoningNode, context: Dict[str, Any]) -> str:
        """Create evaluation prompt for the node."""
        problem = context.get("problem", "")
        parent_content = context.get("parent_content", "")
        
        return f"""
Evaluate the following reasoning step for solving this problem:

Problem: {problem}

Previous reasoning: {parent_content}

Current reasoning step: {node.content}

Please evaluate this reasoning step on the following criteria (score 0.0-1.0):

1. Logical Coherence: How logically sound and consistent is this step?
2. Factual Accuracy: How factually accurate is the information presented?
3. Relevance: How relevant is this step to solving the problem?
4. Completeness: How complete is this step in addressing the required aspects?
5. Clarity: How clear and understandable is the reasoning?

Provide your evaluation in this exact format:
LOGICAL_COHERENCE: [score]
FACTUAL_ACCURACY: [score] 
RELEVANCE: [score]
COMPLETENESS: [score]
CLARITY: [score]

Brief justification: [explanation]
"""
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, float]:
        """Parse the evaluation response to extract scores."""
        scores = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace('_', ' ')
                    try:
                        score = float(parts[1].strip())
                        if key in ['logical coherence', 'factual accuracy', 'relevance', 'completeness', 'clarity']:
                            scores[key.replace(' ', '_')] = max(0.0, min(1.0, score))
                    except ValueError:
                        continue
        
        # Fill in missing scores with default values
        for criterion in self.weights.keys():
            if criterion not in scores:
                scores[criterion] = 0.5
        
        return scores
    
    def get_evaluation_criteria(self) -> List[str]:
        """Get the criteria used for evaluation."""
        return list(self.weights.keys())


class ConfidenceBasedEvaluator(NodeEvaluator):
    """Evaluator that focuses on confidence and uncertainty estimation."""
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def evaluate_node(self, node: ReasoningNode, context: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate based on confidence and uncertainty."""
        confidence_prompt = self._create_confidence_prompt(node, context)
        
        request = LLMRequest(
            prompt=confidence_prompt,
            model=context.get("model", "gpt-3.5-turbo"),
            temperature=0.1,
            max_tokens=300
        )
        
        try:
            response = await self.llm_interface.generate(request)
            
            if response.error:
                return 0.5, 0.3
            
            confidence, reasoning_quality = self._parse_confidence_response(response.content)
            return reasoning_quality, confidence
            
        except Exception as e:
            self.logger.error(f"Error in confidence evaluation for node {node.id}: {e}")
            return 0.5, 0.3
    
    def _create_confidence_prompt(self, node: ReasoningNode, context: Dict[str, Any]) -> str:
        """Create confidence evaluation prompt."""
        problem = context.get("problem", "")
        
        return f"""
Analyze the confidence and quality of this reasoning step:

Problem: {problem}
Reasoning step: {node.content}

Please assess:
1. How confident are you that this reasoning step is correct? (0.0-1.0)
2. What is the overall quality of the reasoning? (0.0-1.0)
3. What are the main sources of uncertainty?

Format your response as:
CONFIDENCE: [score]
REASONING_QUALITY: [score]
UNCERTAINTY_SOURCES: [list main concerns]
"""
    
    def _parse_confidence_response(self, response: str) -> Tuple[float, float]:
        """Parse confidence evaluation response."""
        confidence = 0.5
        reasoning_quality = 0.5
        
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':')[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    pass
            elif line.startswith('REASONING_QUALITY:'):
                try:
                    reasoning_quality = float(line.split(':')[1].strip())
                    reasoning_quality = max(0.0, min(1.0, reasoning_quality))
                except:
                    pass
        
        return confidence, reasoning_quality
    
    def get_evaluation_criteria(self) -> List[str]:
        """Get evaluation criteria."""
        return ["confidence", "reasoning_quality", "uncertainty_assessment"]


class TreeOfThought:
    """Main Tree-of-Thought reasoning implementation."""
    
    def __init__(self, 
                 llm_interface: LLMInterface,
                 evaluator: Optional[NodeEvaluator] = None,
                 config: Dict[str, Any] = None):
        self.llm_interface = llm_interface
        self.evaluator = evaluator or WeightedScoringEvaluator(llm_interface)
        self.config = config or {}
        
        # Configuration parameters
        self.max_depth = self.config.get("max_depth", 4)
        self.branching_factor = self.config.get("branching_factor", 3)
        self.pruning_threshold = self.config.get("pruning_threshold", 0.3)
        self.max_nodes = self.config.get("max_nodes", 50)
        
        # Internal state
        self.nodes: Dict[str, ReasoningNode] = {}
        self.paths: Dict[str, ReasoningPath] = {}
        self.root_id: Optional[str] = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def reason(self, problem: str, num_paths: int = 5) -> List[ReasoningPath]:
        """Main reasoning method that generates multiple reasoning paths."""
        self.logger.info(f"Starting ToT reasoning for problem with {num_paths} paths")
        
        # Initialize root node
        root_id = self._create_root_node(problem)
        
        # Generate reasoning paths
        completed_paths = []
        active_nodes = [root_id]
        
        for depth in range(self.max_depth):
            if not active_nodes or len(completed_paths) >= num_paths:
                break
            
            self.logger.info(f"Processing depth {depth} with {len(active_nodes)} active nodes")
            
            # Generate children for active nodes
            new_nodes = []
            for node_id in active_nodes:
                children = await self._generate_children(node_id, problem)
                new_nodes.extend(children)
            
            # Evaluate new nodes
            await self._evaluate_nodes(new_nodes, problem)
            
            # Prune low-scoring nodes
            surviving_nodes = self._prune_nodes(new_nodes)
            
            # Check for completed paths
            for node_id in surviving_nodes:
                node = self.nodes[node_id]
                if self._is_solution_complete(node, problem):
                    path = self._extract_path(node_id)
                    if path:
                        completed_paths.append(path)
            
            # Update active nodes for next iteration
            active_nodes = [nid for nid in surviving_nodes 
                          if not self._is_solution_complete(self.nodes[nid], problem)]
            
            # Limit the number of active nodes to control branching
            if len(active_nodes) > self.branching_factor * 2:
                active_nodes = sorted(active_nodes, 
                                    key=lambda nid: self.nodes[nid].score, 
                                    reverse=True)[:self.branching_factor * 2]
        
        # If we don't have enough completed paths, extract best partial paths
        while len(completed_paths) < num_paths and active_nodes:
            best_node_id = max(active_nodes, key=lambda nid: self.nodes[nid].score)
            path = self._extract_path(best_node_id, force_completion=True)
            if path:
                completed_paths.append(path)
            active_nodes.remove(best_node_id)
        
        self.logger.info(f"Generated {len(completed_paths)} reasoning paths")
        return completed_paths[:num_paths]
    
    def _create_root_node(self, problem: str) -> str:
        """Create the root node for the reasoning tree."""
        root_id = str(uuid.uuid4())
        root_node = ReasoningNode(
            id=root_id,
            content=f"Problem: {problem}",
            depth=0,
            state=NodeState.ACTIVE,
            metadata={"is_root": True, "problem": problem}
        )
        
        self.nodes[root_id] = root_node
        self.root_id = root_id
        return root_id
    
    async def _generate_children(self, parent_id: str, problem: str) -> List[str]:
        """Generate child nodes for a given parent node."""
        parent_node = self.nodes[parent_id]
        
        if parent_node.depth >= self.max_depth - 1:
            return []
        
        # Create prompt for generating reasoning steps
        generation_prompt = self._create_generation_prompt(parent_node, problem)
        
        request = LLMRequest(
            prompt=generation_prompt,
            model=self.config.get("model", "gpt-3.5-turbo"),
            temperature=0.8,  # Higher temperature for diversity
            max_tokens=800
        )
        
        try:
            response = await self.llm_interface.generate(request)
            
            if response.error:
                self.logger.warning(f"Failed to generate children for node {parent_id}: {response.error}")
                return []
            
            # Parse the response to extract reasoning steps
            reasoning_steps = self._parse_reasoning_steps(response.content)
            
            # Create child nodes
            child_ids = []
            for i, step in enumerate(reasoning_steps[:self.branching_factor]):
                child_id = str(uuid.uuid4())
                child_node = ReasoningNode(
                    id=child_id,
                    content=step,
                    depth=parent_node.depth + 1,
                    parent_id=parent_id,
                    state=NodeState.ACTIVE,
                    metadata={"step_index": i}
                )
                
                self.nodes[child_id] = child_node
                parent_node.add_child(child_id)
                child_ids.append(child_id)
            
            return child_ids
            
        except Exception as e:
            self.logger.error(f"Error generating children for node {parent_id}: {e}")
            return []
    
    def _create_generation_prompt(self, parent_node: ReasoningNode, problem: str) -> str:
        """Create prompt for generating next reasoning steps."""
        # Get the path to this node
        path_content = self._get_path_content(parent_node.id)
        
        return f"""
Given this problem and the reasoning so far, generate {self.branching_factor} different next steps for solving it.

Problem: {problem}

Current reasoning path:
{path_content}

Generate {self.branching_factor} different approaches for the next reasoning step. Each should:
1. Build logically on the previous reasoning
2. Move toward solving the problem
3. Be distinct from the other options
4. Be specific and actionable

Format your response as:
STEP_1: [detailed reasoning step]
STEP_2: [detailed reasoning step]
STEP_3: [detailed reasoning step]
"""
    
    def _parse_reasoning_steps(self, response: str) -> List[str]:
        """Parse the LLM response to extract reasoning steps."""
        steps = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('STEP_'):
                if ':' in line:
                    step_content = line.split(':', 1)[1].strip()
                    if step_content:
                        steps.append(step_content)
        
        # If parsing failed, try to extract any meaningful content
        if not steps:
            meaningful_lines = [line.strip() for line in lines 
                              if line.strip() and not line.strip().startswith('#')]
            steps = meaningful_lines[:self.branching_factor]
        
        return steps
    
    async def _evaluate_nodes(self, node_ids: List[str], problem: str):
        """Evaluate a list of nodes."""
        evaluation_tasks = []
        
        for node_id in node_ids:
            node = self.nodes[node_id]
            parent_content = ""
            
            if node.parent_id:
                parent_content = self._get_path_content(node.parent_id)
            
            context = {
                "problem": problem,
                "parent_content": parent_content,
                "model": self.config.get("model", "gpt-3.5-turbo")
            }
            
            evaluation_tasks.append(self._evaluate_single_node(node, context))
        
        # Execute evaluations concurrently
        if evaluation_tasks:
            await asyncio.gather(*evaluation_tasks, return_exceptions=True)
    
    async def _evaluate_single_node(self, node: ReasoningNode, context: Dict[str, Any]):
        """Evaluate a single node."""
        try:
            score, confidence = await self.evaluator.evaluate_node(node, context)
            node.score = score
            node.confidence = confidence
            node.state = NodeState.EVALUATED
        except Exception as e:
            self.logger.error(f"Error evaluating node {node.id}: {e}")
            node.score = 0.1
            node.confidence = 0.1
            node.state = NodeState.FAILED
    
    def _prune_nodes(self, node_ids: List[str]) -> List[str]:
        """Prune nodes below the threshold score."""
        surviving_nodes = []
        
        for node_id in node_ids:
            node = self.nodes[node_id]
            if node.score >= self.pruning_threshold:
                surviving_nodes.append(node_id)
            else:
                node.state = NodeState.PRUNED
                self.logger.debug(f"Pruned node {node_id} with score {node.score}")
        
        return surviving_nodes
    
    def _is_solution_complete(self, node: ReasoningNode, problem: str) -> bool:
        """Check if a node represents a complete solution."""
        # Simple heuristic: consider solution complete if at sufficient depth
        # and contains answer indicators
        if node.depth < 2:
            return False
        
        content_lower = node.content.lower()
        answer_indicators = [
            "therefore", "answer is", "result is", "solution is",
            "final answer", "conclusion", "equals", "="
        ]
        
        return any(indicator in content_lower for indicator in answer_indicators)
    
    def _extract_path(self, leaf_node_id: str, force_completion: bool = False) -> Optional[ReasoningPath]:
        """Extract a complete reasoning path from root to leaf."""
        if leaf_node_id not in self.nodes:
            return None
        
        # Trace back to root
        path_nodes = []
        current_id = leaf_node_id
        
        while current_id is not None:
            node = self.nodes[current_id]
            path_nodes.append(current_id)
            current_id = node.parent_id
        
        path_nodes.reverse()  # Root to leaf order
        
        # Create reasoning path
        path_id = str(uuid.uuid4())
        leaf_node = self.nodes[leaf_node_id]
        
        reasoning_steps = []
        total_score = 0.0
        total_confidence = 0.0
        
        for node_id in path_nodes[1:]:  # Skip root
            node = self.nodes[node_id]
            reasoning_steps.append(node.content)
            total_score += node.score
            total_confidence += node.confidence
        
        avg_score = total_score / len(reasoning_steps) if reasoning_steps else 0.0
        avg_confidence = total_confidence / len(reasoning_steps) if reasoning_steps else 0.0
        
        path = ReasoningPath(
            id=path_id,
            node_ids=path_nodes,
            total_score=avg_score,
            average_confidence=avg_confidence,
            is_complete=self._is_solution_complete(leaf_node, "") or force_completion,
            final_answer=leaf_node.content if self._is_solution_complete(leaf_node, "") else None,
            reasoning_steps=reasoning_steps,
            metadata={
                "depth": len(path_nodes) - 1,
                "leaf_node_score": leaf_node.score,
                "forced_completion": force_completion
            }
        )
        
        self.paths[path_id] = path
        return path
    
    def _get_path_content(self, node_id: str) -> str:
        """Get the content of the path from root to the specified node."""
        path_content = []
        current_id = node_id
        
        while current_id is not None:
            node = self.nodes[current_id]
            if not node.is_root():
                path_content.append(node.content)
            current_id = node.parent_id
        
        path_content.reverse()
        return "\n".join(path_content)
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reasoning tree."""
        if not self.nodes:
            return {}
        
        states = {}
        depths = []
        scores = []
        
        for node in self.nodes.values():
            states[node.state.value] = states.get(node.state.value, 0) + 1
            depths.append(node.depth)
            if node.score > 0:
                scores.append(node.score)
        
        return {
            "total_nodes": len(self.nodes),
            "node_states": states,
            "max_depth": max(depths) if depths else 0,
            "average_depth": np.mean(depths) if depths else 0,
            "average_score": np.mean(scores) if scores else 0,
            "total_paths": len(self.paths)
        }
    
    def export_tree(self) -> Dict[str, Any]:
        """Export the complete reasoning tree."""
        return {
            "nodes": {nid: {
                "id": node.id,
                "content": node.content,
                "depth": node.depth,
                "parent_id": node.parent_id,
                "children_ids": node.children_ids,
                "state": node.state.value,
                "score": node.score,
                "confidence": node.confidence,
                "metadata": node.metadata,
                "timestamp": node.timestamp
            } for nid, node in self.nodes.items()},
            "paths": {pid: {
                "id": path.id,
                "node_ids": path.node_ids,
                "total_score": path.total_score,
                "average_confidence": path.average_confidence,
                "is_complete": path.is_complete,
                "final_answer": path.final_answer,
                "reasoning_steps": path.reasoning_steps,
                "metadata": path.metadata
            } for pid, path in self.paths.items()},
            "root_id": self.root_id,
            "statistics": self.get_tree_statistics()
        }