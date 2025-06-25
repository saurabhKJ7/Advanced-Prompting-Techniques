"""
Prompt Engineering Pipeline - Core Package

This package provides the core implementation of the multi-path reasoning pipeline
with Tree-of-Thought (ToT) and Self-Consistency, along with automated prompt optimization.
"""

from .pipeline import PromptEngineeringPipeline
from .tot_reasoning import TreeOfThought, ReasoningPath, ReasoningNode
from .self_consistency import SelfConsistency, ConsistencyAggregator
from .prompt_optimizer import PromptOptimizer, OPROOptimizer, TextGradOptimizer
from .llm_interface import LLMInterface, OpenAIInterface, AnthropicInterface
from .utils import (
    setup_logging,
    load_config,
    save_results,
    calculate_metrics,
    format_prompt,
    extract_answer,
    validate_response
)

__version__ = "1.0.0"
__author__ = "Advanced Prompting Research Team"
__email__ = "research@promptengineering.ai"

__all__ = [
    # Main Pipeline
    "PromptEngineeringPipeline",
    
    # Tree-of-Thought Components
    "TreeOfThought",
    "ReasoningPath", 
    "ReasoningNode",
    
    # Self-Consistency Components
    "SelfConsistency",
    "ConsistencyAggregator",
    
    # Prompt Optimization
    "PromptOptimizer",
    "OPROOptimizer", 
    "TextGradOptimizer",
    
    # LLM Interfaces
    "LLMInterface",
    "OpenAIInterface",
    "AnthropicInterface",
    
    # Utilities
    "setup_logging",
    "load_config",
    "save_results",
    "calculate_metrics",
    "format_prompt",
    "extract_answer",
    "validate_response"
]

# Package metadata
PACKAGE_INFO = {
    "name": "prompt-engineering-pipeline",
    "version": __version__,
    "description": "Multi-Path Reasoning Pipeline with Tree-of-Thought and Automated Prompt Optimization",
    "author": __author__,
    "email": __email__,
    "license": "MIT",
    "keywords": ["prompt-engineering", "tree-of-thought", "self-consistency", "llm", "reasoning"],
    "categories": ["AI/ML", "Natural Language Processing", "Prompt Engineering"],
    "dependencies": [
        "openai>=1.3.0",
        "anthropic>=0.7.0",
        "pydantic>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "asyncio",
        "structlog>=22.1.0"
    ]
}

# Configuration defaults
DEFAULT_CONFIG = {
    "tot_config": {
        "num_paths": 5,
        "max_depth": 4,
        "pruning_threshold": 0.3,
        "branching_factor": 3,
        "evaluation_strategy": "weighted_scoring"
    },
    "consistency_config": {
        "aggregation_method": "weighted_consensus",
        "min_agreement": 0.6,
        "confidence_threshold": 0.7,
        "semantic_similarity_threshold": 0.8
    },
    "optimization_config": {
        "strategy": "opro",
        "max_iterations": 3,
        "improvement_threshold": 0.05,
        "validation_split": 0.2
    },
    "llm_config": {
        "default_model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout": 30,
        "retry_attempts": 3
    },
    "logging_config": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_to_file": True,
        "log_file": "logs/pipeline.log"
    }
}

def get_version():
    """Get the current version of the package."""
    return __version__

def get_package_info():
    """Get complete package information."""
    return PACKAGE_INFO.copy()

def get_default_config():
    """Get the default configuration dictionary."""
    return DEFAULT_CONFIG.copy()