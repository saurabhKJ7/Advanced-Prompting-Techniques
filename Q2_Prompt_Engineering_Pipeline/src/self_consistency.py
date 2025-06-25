"""
Self-Consistency Aggregation Module

This module implements Self-Consistency methodology for aggregating multiple
reasoning paths and answers to improve overall accuracy and reliability.
"""

import asyncio
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .tot_reasoning import ReasoningPath
from .llm_interface import LLMInterface, LLMRequest, LLMResponse


class AggregationMethod(Enum):
    """Enumeration of different aggregation methods."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_CONSENSUS = "weighted_consensus"
    CONFIDENCE_FILTERING = "confidence_filtering"
    SEMANTIC_CLUSTERING = "semantic_clustering"
    HYBRID = "hybrid"


class SimilarityMetric(Enum):
    """Enumeration of similarity metrics for answer comparison."""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    NUMERICAL_SIMILARITY = "numerical_similarity"


@dataclass
class Answer:
    """Represents a single answer from a reasoning path."""
    content: str
    confidence: float
    path_id: str
    reasoning_quality: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_values: List[Any] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.content)
    
    def __eq__(self, other):
        if not isinstance(other, Answer):
            return False
        return self.content == other.content


@dataclass
class AnswerCluster:
    """Represents a cluster of similar answers."""
    id: str
    representative_answer: str
    answers: List[Answer]
    similarity_scores: List[float]
    cluster_confidence: float
    cluster_weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_size(self) -> int:
        return len(self.answers)
    
    def get_average_confidence(self) -> float:
        if not self.answers:
            return 0.0
        return sum(answer.confidence for answer in self.answers) / len(self.answers)
    
    def get_weighted_confidence(self) -> float:
        if not self.answers:
            return 0.0
        total_weight = sum(answer.reasoning_quality for answer in self.answers)
        if total_weight == 0:
            return self.get_average_confidence()
        
        weighted_sum = sum(answer.confidence * answer.reasoning_quality 
                          for answer in self.answers)
        return weighted_sum / total_weight


class AnswerExtractor:
    """Extracts structured answers from reasoning text."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Common patterns for answer extraction
        self.patterns = {
            'numerical': [
                r'(?:answer|result|solution)(?:\s+is)?\s*:?\s*([+-]?\d+(?:\.\d+)?)',
                r'=\s*([+-]?\d+(?:\.\d+)?)',
                r'([+-]?\d+(?:\.\d+)?)\s*(?:is|are)\s+the\s+(?:answer|result|solution)',
                r'therefore\s*,?\s*([+-]?\d+(?:\.\d+)?)'
            ],
            'textual': [
                r'(?:answer|result|solution)(?:\s+is)?\s*:?\s*([^.!?\n]+)',
                r'therefore\s*,?\s*([^.!?\n]+)',
                r'conclusion\s*:?\s*([^.!?\n]+)',
                r'final\s+(?:answer|result)\s*:?\s*([^.!?\n]+)'
            ],
            'boolean': [
                r'\b(yes|no|true|false)\b',
                r'(?:answer|result|solution)(?:\s+is)?\s*:?\s*(yes|no|true|false)',
                r'therefore\s*,?\s*(yes|no|true|false)'
            ]
        }
    
    def extract_answer(self, text: str, expected_type: str = "auto") -> List[Any]:
        """Extract answers from reasoning text."""
        text = text.lower().strip()
        extracted = []
        
        if expected_type == "auto":
            # Try all patterns
            for pattern_type, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]
                        extracted.append(self._normalize_answer(match.strip(), pattern_type))
        else:
            # Use specific pattern type
            patterns = self.patterns.get(expected_type, [])
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    extracted.append(self._normalize_answer(match.strip(), expected_type))
        
        # Remove duplicates while preserving order
        unique_extracted = []
        seen = set()
        for item in extracted:
            if item not in seen:
                unique_extracted.append(item)
                seen.add(item)
        
        return unique_extracted
    
    def _normalize_answer(self, answer: str, answer_type: str) -> Any:
        """Normalize extracted answer based on type."""
        if answer_type == "numerical":
            try:
                if '.' in answer:
                    return float(answer)
                else:
                    return int(answer)
            except ValueError:
                return answer
        elif answer_type == "boolean":
            return answer.lower() in ['yes', 'true']
        else:
            return answer.strip()


class SimilarityCalculator:
    """Calculates similarity between answers using various metrics."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self.sentence_model = SentenceTransformer(model_name)
        except:
            self.logger.warning("Failed to load sentence transformer, using TF-IDF instead")
            self.sentence_model = None
        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    
    def calculate_similarity(self, answer1: str, answer2: str, 
                           metric: SimilarityMetric = SimilarityMetric.SEMANTIC_SIMILARITY) -> float:
        """Calculate similarity between two answers."""
        if metric == SimilarityMetric.EXACT_MATCH:
            return 1.0 if answer1.strip().lower() == answer2.strip().lower() else 0.0
        
        elif metric == SimilarityMetric.FUZZY_MATCH:
            return self._fuzzy_similarity(answer1, answer2)
        
        elif metric == SimilarityMetric.SEMANTIC_SIMILARITY:
            return self._semantic_similarity(answer1, answer2)
        
        elif metric == SimilarityMetric.NUMERICAL_SIMILARITY:
            return self._numerical_similarity(answer1, answer2)
        
        else:
            return 0.0
    
    def _fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy string similarity."""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        except:
            # Fallback to simple Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence embeddings."""
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            except:
                pass
        
        # Fallback to TF-IDF similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            return float(similarity)
        except:
            return self._fuzzy_similarity(text1, text2)
    
    def _numerical_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on numerical values."""
        extractor = AnswerExtractor()
        nums1 = [x for x in extractor.extract_answer(text1, "numerical") if isinstance(x, (int, float))]
        nums2 = [x for x in extractor.extract_answer(text2, "numerical") if isinstance(x, (int, float))]
        
        if not nums1 or not nums2:
            return 0.0
        
        # Find best matching pairs
        max_similarity = 0.0
        for n1 in nums1:
            for n2 in nums2:
                if n1 == n2:
                    max_similarity = 1.0
                elif n1 != 0 and n2 != 0:
                    similarity = 1.0 - abs(n1 - n2) / max(abs(n1), abs(n2))
                    max_similarity = max(max_similarity, similarity)
        
        return max_similarity


class ConsistencyAggregator(ABC):
    """Abstract base class for consistency aggregation strategies."""
    
    @abstractmethod
    async def aggregate(self, reasoning_paths: List[ReasoningPath], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate reasoning paths into a final answer."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of this aggregation method."""
        pass


class MajorityVoteAggregator(ConsistencyAggregator):
    """Aggregates answers using simple majority voting."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.answer_extractor = AnswerExtractor()
        self.similarity_calculator = SimilarityCalculator()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def aggregate(self, reasoning_paths: List[ReasoningPath], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate using majority vote."""
        # Extract answers from all paths
        answers = []
        for path in reasoning_paths:
            answer_text = path.final_answer or (path.reasoning_steps[-1] if path.reasoning_steps else "")
            extracted_values = self.answer_extractor.extract_answer(answer_text)
            
            answer = Answer(
                content=answer_text,
                confidence=path.average_confidence,
                path_id=path.id,
                reasoning_quality=path.total_score,
                extracted_values=extracted_values
            )
            answers.append(answer)
        
        # Group similar answers
        answer_groups = await self._group_similar_answers(answers)
        
        # Find majority group
        majority_group = max(answer_groups, key=lambda g: len(g.answers))
        
        return {
            "final_answer": majority_group.representative_answer,
            "confidence": majority_group.get_average_confidence(),
            "supporting_paths": len(majority_group.answers),
            "total_paths": len(reasoning_paths),
            "agreement_ratio": len(majority_group.answers) / len(reasoning_paths),
            "answer_groups": [
                {
                    "answer": group.representative_answer,
                    "count": len(group.answers),
                    "confidence": group.get_average_confidence()
                }
                for group in answer_groups
            ],
            "method": self.get_method_name(),
            "metadata": {
                "similarity_threshold": self.similarity_threshold,
                "total_answer_groups": len(answer_groups)
            }
        }
    
    async def _group_similar_answers(self, answers: List[Answer]) -> List[AnswerCluster]:
        """Group similar answers into clusters."""
        if not answers:
            return []
        
        clusters = []
        processed = set()
        
        for i, answer in enumerate(answers):
            if i in processed:
                continue
            
            # Create new cluster with this answer
            cluster_answers = [answer]
            processed.add(i)
            
            # Find similar answers
            for j, other_answer in enumerate(answers[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self.similarity_calculator.calculate_similarity(
                    answer.content, other_answer.content, 
                    SimilarityMetric.SEMANTIC_SIMILARITY
                )
                
                if similarity >= self.similarity_threshold:
                    cluster_answers.append(other_answer)
                    processed.add(j)
            
            # Create cluster
            cluster = AnswerCluster(
                id=f"cluster_{len(clusters)}",
                representative_answer=self._select_representative(cluster_answers),
                answers=cluster_answers,
                similarity_scores=[],
                cluster_confidence=sum(a.confidence for a in cluster_answers) / len(cluster_answers),
                cluster_weight=len(cluster_answers)
            )
            clusters.append(cluster)
        
        return clusters
    
    def _select_representative(self, answers: List[Answer]) -> str:
        """Select the most representative answer from a cluster."""
        if not answers:
            return ""
        
        # Select the answer with highest confidence
        best_answer = max(answers, key=lambda a: a.confidence)
        return best_answer.content
    
    def get_method_name(self) -> str:
        return "majority_vote"


class WeightedConsensusAggregator(ConsistencyAggregator):
    """Aggregates answers using weighted consensus based on reasoning quality."""
    
    def __init__(self, similarity_threshold: float = 0.7, min_consensus: float = 0.6):
        self.similarity_threshold = similarity_threshold
        self.min_consensus = min_consensus
        self.answer_extractor = AnswerExtractor()
        self.similarity_calculator = SimilarityCalculator()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def aggregate(self, reasoning_paths: List[ReasoningPath], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate using weighted consensus."""
        # Extract answers with weights
        answers = []
        for path in reasoning_paths:
            answer_text = path.final_answer or (path.reasoning_steps[-1] if path.reasoning_steps else "")
            extracted_values = self.answer_extractor.extract_answer(answer_text)
            
            answer = Answer(
                content=answer_text,
                confidence=path.average_confidence,
                path_id=path.id,
                reasoning_quality=path.total_score,
                extracted_values=extracted_values
            )
            answers.append(answer)
        
        # Group similar answers
        answer_groups = await self._group_similar_answers_weighted(answers)
        
        # Calculate weighted consensus
        best_group = self._find_weighted_consensus(answer_groups)
        
        if best_group is None:
            # No consensus found, return best individual answer
            best_answer = max(answers, key=lambda a: a.confidence * a.reasoning_quality)
            return {
                "final_answer": best_answer.content,
                "confidence": best_answer.confidence,
                "supporting_paths": 1,
                "total_paths": len(reasoning_paths),
                "agreement_ratio": 1 / len(reasoning_paths),
                "consensus_found": False,
                "method": self.get_method_name()
            }
        
        return {
            "final_answer": best_group.representative_answer,
            "confidence": best_group.get_weighted_confidence(),
            "supporting_paths": len(best_group.answers),
            "total_paths": len(reasoning_paths),
            "agreement_ratio": len(best_group.answers) / len(reasoning_paths),
            "consensus_weight": best_group.cluster_weight,
            "consensus_found": True,
            "answer_groups": [
                {
                    "answer": group.representative_answer,
                    "count": len(group.answers),
                    "weight": group.cluster_weight,
                    "confidence": group.get_weighted_confidence()
                }
                for group in answer_groups
            ],
            "method": self.get_method_name(),
            "metadata": {
                "similarity_threshold": self.similarity_threshold,
                "min_consensus": self.min_consensus
            }
        }
    
    async def _group_similar_answers_weighted(self, answers: List[Answer]) -> List[AnswerCluster]:
        """Group similar answers with weighted clustering."""
        if not answers:
            return []
        
        clusters = []
        processed = set()
        
        for i, answer in enumerate(answers):
            if i in processed:
                continue
            
            cluster_answers = [answer]
            processed.add(i)
            
            for j, other_answer in enumerate(answers[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self.similarity_calculator.calculate_similarity(
                    answer.content, other_answer.content,
                    SimilarityMetric.SEMANTIC_SIMILARITY
                )
                
                if similarity >= self.similarity_threshold:
                    cluster_answers.append(other_answer)
                    processed.add(j)
            
            # Calculate cluster weight based on reasoning quality
            total_weight = sum(a.reasoning_quality for a in cluster_answers)
            
            cluster = AnswerCluster(
                id=f"weighted_cluster_{len(clusters)}",
                representative_answer=self._select_weighted_representative(cluster_answers),
                answers=cluster_answers,
                similarity_scores=[],
                cluster_confidence=sum(a.confidence * a.reasoning_quality for a in cluster_answers) / total_weight if total_weight > 0 else 0,
                cluster_weight=total_weight
            )
            clusters.append(cluster)
        
        return clusters
    
    def _select_weighted_representative(self, answers: List[Answer]) -> str:
        """Select representative based on weighted scoring."""
        if not answers:
            return ""
        
        best_answer = max(answers, key=lambda a: a.confidence * a.reasoning_quality)
        return best_answer.content
    
    def _find_weighted_consensus(self, groups: List[AnswerCluster]) -> Optional[AnswerCluster]:
        """Find the group that meets weighted consensus criteria."""
        if not groups:
            return None
        
        # Sort by cluster weight
        sorted_groups = sorted(groups, key=lambda g: g.cluster_weight, reverse=True)
        
        # Check if top group meets consensus threshold
        top_group = sorted_groups[0]
        total_weight = sum(g.cluster_weight for g in groups)
        
        if total_weight == 0:
            return top_group
        
        consensus_ratio = top_group.cluster_weight / total_weight
        
        if consensus_ratio >= self.min_consensus:
            return top_group
        
        return None
    
    def get_method_name(self) -> str:
        return "weighted_consensus"


class ConfidenceFilteringAggregator(ConsistencyAggregator):
    """Aggregates answers by filtering based on confidence thresholds."""
    
    def __init__(self, confidence_threshold: float = 0.7, min_paths: int = 2):
        self.confidence_threshold = confidence_threshold
        self.min_paths = min_paths
        self.answer_extractor = AnswerExtractor()
        self.similarity_calculator = SimilarityCalculator()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def aggregate(self, reasoning_paths: List[ReasoningPath], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate using confidence filtering."""
        # Extract answers
        answers = []
        for path in reasoning_paths:
            answer_text = path.final_answer or (path.reasoning_steps[-1] if path.reasoning_steps else "")
            
            answer = Answer(
                content=answer_text,
                confidence=path.average_confidence,
                path_id=path.id,
                reasoning_quality=path.total_score,
                extracted_values=self.answer_extractor.extract_answer(answer_text)
            )
            answers.append(answer)
        
        # Filter by confidence
        high_confidence_answers = [a for a in answers if a.confidence >= self.confidence_threshold]
        
        if len(high_confidence_answers) < self.min_paths:
            # Not enough high-confidence answers, use all answers
            working_answers = answers
            filtered = False
        else:
            working_answers = high_confidence_answers
            filtered = True
        
        # Find most common answer among high-confidence ones
        if not working_answers:
            return {
                "final_answer": "No valid answers found",
                "confidence": 0.0,
                "supporting_paths": 0,
                "total_paths": len(reasoning_paths),
                "method": self.get_method_name()
            }
        
        # Group similar answers
        answer_groups = await self._group_answers(working_answers)
        best_group = max(answer_groups, key=lambda g: len(g.answers))
        
        return {
            "final_answer": best_group.representative_answer,
            "confidence": best_group.get_average_confidence(),
            "supporting_paths": len(best_group.answers),
            "total_paths": len(reasoning_paths),
            "high_confidence_paths": len(high_confidence_answers),
            "confidence_filtered": filtered,
            "agreement_ratio": len(best_group.answers) / len(working_answers),
            "method": self.get_method_name(),
            "metadata": {
                "confidence_threshold": self.confidence_threshold,
                "min_paths": self.min_paths
            }
        }
    
    async def _group_answers(self, answers: List[Answer]) -> List[AnswerCluster]:
        """Group similar answers."""
        # Similar to MajorityVoteAggregator but simpler
        if not answers:
            return []
        
        clusters = []
        processed = set()
        
        for i, answer in enumerate(answers):
            if i in processed:
                continue
            
            cluster_answers = [answer]
            processed.add(i)
            
            for j, other_answer in enumerate(answers[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self.similarity_calculator.calculate_similarity(
                    answer.content, other_answer.content
                )
                
                if similarity >= 0.8:
                    cluster_answers.append(other_answer)
                    processed.add(j)
            
            cluster = AnswerCluster(
                id=f"conf_cluster_{len(clusters)}",
                representative_answer=max(cluster_answers, key=lambda a: a.confidence).content,
                answers=cluster_answers,
                similarity_scores=[],
                cluster_confidence=sum(a.confidence for a in cluster_answers) / len(cluster_answers),
                cluster_weight=len(cluster_answers)
            )
            clusters.append(cluster)
        
        return clusters
    
    def get_method_name(self) -> str:
        return "confidence_filtering"


class SelfConsistency:
    """Main Self-Consistency implementation."""
    
    def __init__(self, aggregator: Optional[ConsistencyAggregator] = None,
                 config: Dict[str, Any] = None):
        self.config = config or {}
        self.aggregator = aggregator or self._create_default_aggregator()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.min_agreement = self.config.get("min_agreement", 0.6)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
    
    def _create_default_aggregator(self) -> ConsistencyAggregator:
        """Create default aggregator based on configuration."""
        method = self.config.get("aggregation_method", "weighted_consensus")
        
        if method == "majority_vote":
            return MajorityVoteAggregator(self.similarity_threshold)
        elif method == "weighted_consensus":
            return WeightedConsensusAggregator(self.similarity_threshold, self.min_agreement)
        elif method == "confidence_filtering":
            return ConfidenceFilteringAggregator(self.confidence_threshold)
        else:
            return WeightedConsensusAggregator(self.similarity_threshold, self.min_agreement)
    
    async def aggregate_paths(self, reasoning_paths: List[ReasoningPath],
                            problem: str = "") -> Dict[str, Any]:
        """Aggregate multiple reasoning paths into a consistent answer."""
        if not reasoning_paths:
            return {
                "final_answer": "No reasoning paths provided",
                "confidence": 0.0,
                "supporting_paths": 0,
                "total_paths": 0,
                "method": "none"
            }
        
        self.logger.info(f"Aggregating {len(reasoning_paths)} reasoning paths")
        
        context = {
            "problem": problem,
            "config": self.config
        }
        
        start_time = time.time()
        result = await self.aggregator.aggregate(reasoning_paths, context)
        aggregation_time = time.time() - start_time
        
        # Add timing and additional metadata
        result.update({
            "aggregation_time": aggregation_time,
            "input_paths": len(reasoning_paths),
            "consistency_metrics": self._calculate_consistency_metrics(reasoning_paths, result)
        })
        
        self.logger.info(f"Aggregation completed in {aggregation_time:.2f}s")
        return result
    
    def _calculate_consistency_metrics(self, paths: List[ReasoningPath], 
                                     result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate consistency metrics for the aggregation."""
        if not paths:
            return {}
        
        # Calculate diversity metrics
        unique_final_answers = set()
        confidences = []
        scores = []
        
        for path in paths:
            final_answer = path.final_answer or (path.reasoning_steps[-1] if path.reasoning_steps else "")
            unique_final_answers.add(final_answer.strip().lower())
            confidences.append(path.average_confidence)
            scores.append(path.total_score)
        
        answer_diversity = len(unique_final_answers) / len(paths)
        confidence_variance = np.var(confidences) if confidences else 0.0
        score_variance = np.var(scores) if scores else 0.0
        
        return {
            "answer_diversity": answer_diversity,
            "confidence_variance": confidence_variance,
            "score_variance": score_variance,
            "average_confidence": np.mean(confidences) if confidences else 0.0,
            "average_score": np.mean(scores) if scores else 0.0,
            "agreement_strength": result.get("agreement_ratio", 0.0)
        }
    
    def get_aggregator_info(self) -> Dict[str, Any]:
        """Get information about the current aggregator."""
        return {
            "method": self.aggregator.get_method_name(),
            "config": self.config,
            "similarity_threshold": self.similarity_threshold,
            "min_agreement": self.min_agreement,
            "confidence_threshold": self.confidence_threshold
        }