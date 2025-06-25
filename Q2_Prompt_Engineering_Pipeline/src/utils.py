"""
Utilities Module

This module provides utility functions for the prompt engineering pipeline,
including logging setup, configuration management, result handling, and common helpers.
"""

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import yaml

import numpy as np
import pandas as pd
from structlog import configure, get_logger
import structlog


def setup_logging(level: str = "INFO", log_to_file: bool = True, 
                 log_file: str = "logs/pipeline.log") -> logging.Logger:
    """Setup structured logging for the pipeline."""
    
    # Create logs directory if it doesn't exist
    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    
    configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Setup standard logging
    logger = logging.getLogger("prompt_pipeline")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return config
    
    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML or JSON file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    except Exception as e:
        raise ValueError(f"Error saving configuration to {config_path}: {e}")


def save_results(results: Dict[str, Any], output_path: str, 
                format: str = "json") -> None:
    """Save results to file in specified format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to results
    results["timestamp"] = datetime.now().isoformat()
    results["save_format"] = format
    
    try:
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        elif format.lower() == "yaml":
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, indent=2)
        
        elif format.lower() == "csv":
            # Flatten results for CSV format
            flattened = flatten_dict(results)
            df = pd.DataFrame([flattened])
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    except Exception as e:
        raise ValueError(f"Error saving results to {output_path}: {e}")


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from file."""
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    try:
        if results_path.suffix.lower() == '.json':
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif results_path.suffix.lower() in ['.yaml', '.yml']:
            with open(results_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        elif results_path.suffix.lower() == '.csv':
            df = pd.read_csv(results_path)
            return df.to_dict('records')[0] if not df.empty else {}
        
        else:
            raise ValueError(f"Unsupported results file format: {results_path.suffix}")
    
    except Exception as e:
        raise ValueError(f"Error loading results from {results_path}: {e}")


def calculate_metrics(predictions: List[Any], ground_truth: List[Any],
                     metric_types: List[str] = None) -> Dict[str, float]:
    """Calculate various evaluation metrics."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    if not predictions:
        return {}
    
    if metric_types is None:
        metric_types = ["accuracy", "precision", "recall", "f1"]
    
    metrics = {}
    
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Convert to appropriate format for sklearn
        pred_array = np.array(predictions)
        truth_array = np.array(ground_truth)
        
        # Classification metrics
        if "accuracy" in metric_types:
            try:
                metrics["accuracy"] = accuracy_score(truth_array, pred_array)
            except:
                # For non-classification tasks, use exact match
                metrics["accuracy"] = np.mean(pred_array == truth_array)
        
        if any(m in metric_types for m in ["precision", "recall", "f1"]):
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    truth_array, pred_array, average='weighted', zero_division=0
                )
                if "precision" in metric_types:
                    metrics["precision"] = precision
                if "recall" in metric_types:
                    metrics["recall"] = recall
                if "f1" in metric_types:
                    metrics["f1"] = f1
            except:
                # Fallback for non-standard classification
                pass
        
        # Regression metrics (if applicable)
        if "mse" in metric_types:
            try:
                if all(isinstance(x, (int, float)) for x in predictions + ground_truth):
                    metrics["mse"] = mean_squared_error(truth_array, pred_array)
            except:
                pass
        
        if "mae" in metric_types:
            try:
                if all(isinstance(x, (int, float)) for x in predictions + ground_truth):
                    metrics["mae"] = mean_absolute_error(truth_array, pred_array)
            except:
                pass
        
        # Custom metrics
        if "exact_match" in metric_types:
            exact_matches = sum(1 for p, t in zip(predictions, ground_truth) if str(p).strip() == str(t).strip())
            metrics["exact_match"] = exact_matches / len(predictions)
        
        if "fuzzy_match" in metric_types:
            fuzzy_matches = 0
            for p, t in zip(predictions, ground_truth):
                similarity = fuzzy_string_similarity(str(p), str(t))
                if similarity >= 0.8:
                    fuzzy_matches += 1
            metrics["fuzzy_match"] = fuzzy_matches / len(predictions)
    
    except ImportError:
        # Fallback if sklearn is not available
        exact_matches = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
        metrics["accuracy"] = exact_matches / len(predictions)
    
    return metrics


def format_prompt(template: str, variables: Dict[str, Any]) -> str:
    """Format a prompt template with variables."""
    try:
        return template.format(**variables)
    except KeyError as e:
        raise ValueError(f"Missing variable in prompt template: {e}")
    except Exception as e:
        raise ValueError(f"Error formatting prompt template: {e}")


def extract_answer(text: str, patterns: List[str] = None) -> List[str]:
    """Extract answers from text using regex patterns."""
    if patterns is None:
        patterns = [
            r'(?:answer|result|solution)(?:\s+is)?\s*:?\s*([^.\n]+)',
            r'therefore\s*,?\s*([^.\n]+)',
            r'=\s*([^.\n]+)',
            r'the answer is\s*([^.\n]+)',
            r'final answer\s*:?\s*([^.\n]+)'
        ]
    
    extracted = []
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            answer = match.group(1).strip()
            if answer and answer not in extracted:
                extracted.append(answer)
    
    return extracted


def extract_numbers(text: str) -> List[Union[int, float]]:
    """Extract numerical values from text."""
    # Pattern for numbers (including decimals and negatives)
    number_pattern = r'-?\d+\.?\d*'
    matches = re.findall(number_pattern, text)
    
    numbers = []
    for match in matches:
        try:
            if '.' in match:
                numbers.append(float(match))
            else:
                numbers.append(int(match))
        except ValueError:
            continue
    
    return numbers


def validate_response(response: str, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """Validate response against specified rules."""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Length validation
    if "min_length" in validation_rules:
        if len(response) < validation_rules["min_length"]:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Response too short: {len(response)} < {validation_rules['min_length']}")
    
    if "max_length" in validation_rules:
        if len(response) > validation_rules["max_length"]:
            validation_result["warnings"].append(f"Response too long: {len(response)} > {validation_rules['max_length']}")
    
    # Required patterns
    if "required_patterns" in validation_rules:
        for pattern in validation_rules["required_patterns"]:
            if not re.search(pattern, response, re.IGNORECASE):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required pattern: {pattern}")
    
    # Forbidden patterns
    if "forbidden_patterns" in validation_rules:
        for pattern in validation_rules["forbidden_patterns"]:
            if re.search(pattern, response, re.IGNORECASE):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Contains forbidden pattern: {pattern}")
    
    # Numerical validation
    if "expected_numbers" in validation_rules:
        extracted_numbers = extract_numbers(response)
        expected_numbers = validation_rules["expected_numbers"]
        
        if not any(abs(num - exp) < 1e-6 for num in extracted_numbers for exp in expected_numbers):
            validation_result["warnings"].append("No expected numerical values found")
    
    return validation_result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, list_key, sep=sep).items())
                else:
                    items.append((list_key, item))
        else:
            items.append((new_key, v))
    
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflatten a flattened dictionary."""
    result = {}
    
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        
        for part in parts[:-1]:
            if '[' in part and ']' in part:
                # Handle list indices
                list_key, index_str = part.split('[', 1)
                index = int(index_str.rstrip(']'))
                
                if list_key not in current:
                    current[list_key] = []
                
                # Extend list if necessary
                while len(current[list_key]) <= index:
                    current[list_key].append({})
                
                current = current[list_key][index]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Set the final value
        final_key = parts[-1]
        if '[' in final_key and ']' in final_key:
            list_key, index_str = final_key.split('[', 1)
            index = int(index_str.rstrip(']'))
            
            if list_key not in current:
                current[list_key] = []
            
            while len(current[list_key]) <= index:
                current[list_key].append(None)
            
            current[list_key][index] = value
        else:
            current[final_key] = value
    
    return result


def fuzzy_string_similarity(s1: str, s2: str) -> float:
    """Calculate fuzzy similarity between two strings."""
    try:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    except ImportError:
        # Fallback to simple Jaccard similarity
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values."""
    if not values:
        return 0.0, 0.0
    
    try:
        import scipy.stats as stats
        
        mean = np.mean(values)
        sem = stats.sem(values)  # Standard error of the mean
        
        # Calculate confidence interval
        h = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
        
        return mean - h, mean + h
    
    except ImportError:
        # Fallback without scipy
        mean = np.mean(values)
        std = np.std(values)
        margin = 1.96 * std / np.sqrt(len(values))  # Approximate 95% CI
        
        return mean - margin, mean + margin


def create_unique_id(prefix: str = "") -> str:
    """Create a unique identifier."""
    timestamp = int(time.time() * 1000)
    unique_part = str(uuid.uuid4())[:8]
    
    if prefix:
        return f"{prefix}_{timestamp}_{unique_part}"
    else:
        return f"{timestamp}_{unique_part}"


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Get SHA-256 hash of a file."""
    import hashlib
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries, with override taking precedence."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split a list into batches of specified size."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function calls on failure."""
    def decorator(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
                else:
                    break
        
        raise last_exception
    
    return decorator


# Environment and configuration helpers
def get_env_var(var_name: str, default: Any = None, var_type: type = str) -> Any:
    """Get environment variable with type conversion and default."""
    value = os.environ.get(var_name, default)
    
    if value is None:
        return None
    
    try:
        if var_type == bool:
            return str(value).lower() in ('true', '1', 'yes', 'on')
        elif var_type == int:
            return int(value)
        elif var_type == float:
            return float(value)
        elif var_type == list:
            return value.split(',') if isinstance(value, str) else value
        else:
            return var_type(value)
    except (ValueError, TypeError):
        return default


def setup_environment() -> Dict[str, Any]:
    """Setup environment configuration from environment variables."""
    return {
        "openai_api_key": get_env_var("OPENAI_API_KEY"),
        "anthropic_api_key": get_env_var("ANTHROPIC_API_KEY"),
        "google_api_key": get_env_var("GOOGLE_API_KEY"),
        "log_level": get_env_var("LOG_LEVEL", "INFO"),
        "max_concurrent_requests": get_env_var("MAX_CONCURRENT_REQUESTS", 10, int),
        "default_model": get_env_var("DEFAULT_MODEL", "gpt-3.5-turbo"),
        "cache_enabled": get_env_var("CACHE_ENABLED", True, bool),
        "debug_mode": get_env_var("DEBUG_MODE", False, bool)
    }