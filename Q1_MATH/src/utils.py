import re
import json
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class MathEvaluator:
    """Utility class for evaluating mathematical responses"""
    
    @staticmethod
    def extract_final_answer(response: str) -> str:
        """Extract the final numerical answer from a response"""
        # Look for common answer patterns
        patterns = [
            r'(?:answer is|equals?|=)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'(?:x\s*=|y\s*=|z\s*=)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'(?:Therefore|Thus|So),?\s*(?:the answer is)?\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(?:is the answer|is correct)',
            r'final answer:?\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no pattern matches, look for the last number in the response
        numbers = re.findall(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?', response)
        if numbers:
            return numbers[-1]
        
        return ""

    @staticmethod
    def check_mathematical_accuracy(response: str, expected_answer: str) -> bool:
        """Check if the mathematical answer is correct"""
        extracted_answer = MathEvaluator.extract_final_answer(response)
        
        if not extracted_answer:
            return False
        
        try:
            response_value = float(extracted_answer)
            expected_value = float(expected_answer)
            
            # Allow for small floating point differences
            return abs(response_value - expected_value) < 1e-6
        except ValueError:
            # Compare as strings if not numeric
            return extracted_answer.strip().lower() == expected_answer.strip().lower()

    @staticmethod
    def count_solution_steps(response: str) -> int:
        """Count the number of solution steps in a response"""
        step_patterns = [
            r'step\s+\d+',
            r'\d+\.\s+',
            r'first,?\s+',
            r'second,?\s+',
            r'third,?\s+',
            r'next,?\s+',
            r'then,?\s+',
            r'finally,?\s+'
        ]
        
        total_steps = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            total_steps += len(matches)
        
        return max(total_steps, 1)  # At least 1 step

class ResponseAnalyzer:
    """Analyze and score LLM responses"""
    
    @staticmethod
    def analyze_reasoning_clarity(response: str) -> Dict[str, Any]:
        """Analyze the clarity of reasoning in a response"""
        metrics = {
            'has_step_by_step': False,
            'uses_mathematical_notation': False,
            'explains_concepts': False,
            'includes_verification': False,
            'length_appropriate': False,
            'clarity_score': 0
        }
        
        # Check for step-by-step structure
        step_indicators = ['step', 'first', 'second', 'next', 'then', 'finally']
        metrics['has_step_by_step'] = any(word in response.lower() for word in step_indicators)
        
        # Check for mathematical notation
        math_patterns = [r'[=+\-*/]', r'\^', r'\d+', r'[xy]', r'[(){}[\]]']
        metrics['uses_mathematical_notation'] = any(re.search(pattern, response) for pattern in math_patterns)
        
        # Check for concept explanations
        explanation_words = ['because', 'since', 'therefore', 'thus', 'so', 'means', 'represents']
        metrics['explains_concepts'] = any(word in response.lower() for word in explanation_words)
        
        # Check for verification
        verification_words = ['check', 'verify', 'confirm', 'substitute', 'test']
        metrics['includes_verification'] = any(word in response.lower() for word in verification_words)
        
        # Check length appropriateness (between 50 and 1000 words)
        word_count = len(response.split())
        metrics['length_appropriate'] = 50 <= word_count <= 1000
        
        # Calculate overall clarity score
        score_components = [
            metrics['has_step_by_step'],
            metrics['uses_mathematical_notation'],
            metrics['explains_concepts'],
            metrics['includes_verification'],
            metrics['length_appropriate']
        ]
        metrics['clarity_score'] = sum(score_components) / len(score_components)
        
        return metrics

    @staticmethod
    def detect_hallucinations(response: str) -> Dict[str, Any]:
        """Detect potential hallucinations in mathematical responses"""
        issues = {
            'mathematical_errors': [],
            'inconsistent_statements': [],
            'impossible_values': [],
            'hallucination_score': 0
        }
        
        # Check for basic mathematical inconsistencies
        equations = re.findall(r'(\d+)\s*[+\-*/]\s*(\d+)\s*=\s*(\d+)', response)
        for eq in equations:
            try:
                left, op_right = eq[0], eq[1:]
                # This is a simplified check - in reality, you'd need more sophisticated parsing
                pass
            except:
                issues['mathematical_errors'].append(f"Invalid equation: {eq}")
        
        # Check for impossible mathematical values
        if re.search(r'divide by zero|division by zero', response, re.IGNORECASE):
            issues['impossible_values'].append("Division by zero mentioned")
        
        # Check for contradictory statements
        if re.search(r'always.*never|never.*always', response, re.IGNORECASE):
            issues['inconsistent_statements'].append("Contradictory absolute statements")
        
        # Calculate hallucination score (lower is better)
        total_issues = len(issues['mathematical_errors']) + len(issues['inconsistent_statements']) + len(issues['impossible_values'])
        issues['hallucination_score'] = min(total_issues / 10.0, 1.0)  # Normalize to 0-1
        
        return issues

class PromptOptimizer:
    """Utilities for optimizing prompts"""
    
    @staticmethod
    def suggest_prompt_improvements(prompt: str, performance_data: Dict) -> List[str]:
        """Suggest improvements to a prompt based on performance data"""
        suggestions = []
        
        if performance_data.get('accuracy_score', 0) < 0.7:
            suggestions.append("Add more specific instructions for mathematical accuracy")
            suggestions.append("Include examples of correct calculation steps")
        
        if performance_data.get('clarity_score', 0) < 0.6:
            suggestions.append("Emphasize the need for step-by-step explanations")
            suggestions.append("Add instructions to explain mathematical reasoning")
        
        if performance_data.get('hallucination_score', 0) > 0.3:
            suggestions.append("Add warnings against making unsupported claims")
            suggestions.append("Include verification steps in the prompt")
        
        if performance_data.get('consistency_score', 0) < 0.8:
            suggestions.append("Provide more consistent formatting guidelines")
            suggestions.append("Add template structure for responses")
        
        return suggestions

    @staticmethod
    def generate_fallback_response(user_question: str, error_type: str = "unclear") -> str:
        """Generate appropriate fallback responses for ambiguous questions"""
        fallback_templates = {
            "unclear": [
                "I need some clarification to help you better. Could you please:",
                "- Specify what grade level this is for?",
                "- Provide any missing numbers or measurements?",
                "- Clarify what you're trying to find or solve?",
                "",
                "For example, if you're asking about area, please specify the shape and dimensions."
            ],
            "missing_info": [
                "I notice some information might be missing from your question.",
                "To solve this problem, I would need:",
                "- All relevant measurements or values",
                "- The specific mathematical operation you want to perform",
                "- Any constraints or conditions",
                "",
                "Could you please provide these details?"
            ],
            "out_of_scope": [
                "This question seems to be beyond the grade 6-10 mathematics curriculum I'm designed for.",
                "I can help you with:",
                "- Basic algebra and equations",
                "- Geometry (area, perimeter, volume)",
                "- Arithmetic operations",
                "- Word problems",
                "- Basic probability and statistics",
                "",
                "Could you rephrase your question to focus on these topics?"
            ]
        }
        
        return "\n".join(fallback_templates.get(error_type, fallback_templates["unclear"]))

class ConversationManager:
    """Manage conversation context and history"""
    
    def __init__(self):
        self.conversation_context = []
        self.user_preferences = {}
    
    def add_interaction(self, question: str, response: str, prompt_type: str, metadata: Dict = None):
        """Add an interaction to the conversation history"""
        interaction = {
            'question': question,
            'response': response,
            'prompt_type': prompt_type,
            'timestamp': str(Path(__file__).stat().st_mtime),  # Simple timestamp
            'metadata': metadata or {}
        }
        self.conversation_context.append(interaction)
    
    def get_context_summary(self, last_n: int = 3) -> str:
        """Get a summary of recent conversation context"""
        if not self.conversation_context:
            return "No previous conversation context."
        
        recent_interactions = self.conversation_context[-last_n:]
        context_lines = []
        
        for i, interaction in enumerate(recent_interactions, 1):
            context_lines.append(f"Previous Q{i}: {interaction['question'][:100]}...")
            context_lines.append(f"Previous A{i}: {interaction['response'][:100]}...")
        
        return "\n".join(context_lines)
    
    def detect_user_level(self) -> str:
        """Attempt to detect user's mathematical level from conversation history"""
        if not self.conversation_context:
            return "unknown"
        
        # Simple heuristic based on question complexity
        recent_questions = [interaction['question'].lower() for interaction in self.conversation_context[-5:]]
        
        advanced_keywords = ['derivative', 'integral', 'calculus', 'logarithm', 'trigonometry', 'matrix']
        intermediate_keywords = ['quadratic', 'equation', 'variable', 'algebra', 'function']
        basic_keywords = ['add', 'subtract', 'multiply', 'divide', 'area', 'perimeter']
        
        for question in recent_questions:
            if any(keyword in question for keyword in advanced_keywords):
                return "advanced"
            elif any(keyword in question for keyword in intermediate_keywords):
                return "intermediate"
        
        return "basic"

def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.3,
            "max_history": 10
        }

def save_session_data(data: Dict, filepath: str):
    """Save session data to file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Session data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save session data: {e}")