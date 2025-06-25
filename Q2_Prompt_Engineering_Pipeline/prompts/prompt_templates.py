"""
Prompt Templates Collection

This module contains various prompt templates for different reasoning tasks
and optimization strategies used in the pipeline.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """Types of prompts in the system."""
    REASONING = "reasoning"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"
    SYSTEM = "system"


@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata."""
    name: str
    template: str
    prompt_type: PromptType
    description: str
    variables: List[str]
    domain: str = "general"
    difficulty: str = "medium"
    examples: List[Dict[str, str]] = None


class PromptTemplates:
    """Collection of prompt templates for different tasks."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates."""
        templates = {}
        
        # Reasoning Templates
        templates["basic_reasoning"] = PromptTemplate(
            name="basic_reasoning",
            template="""
Solve the following problem step by step:

Problem: {problem}

Please think through this problem carefully and show your reasoning process clearly. Break down the problem into manageable steps and explain each step.

Solution:
""",
            prompt_type=PromptType.REASONING,
            description="Basic step-by-step reasoning template",
            variables=["problem"],
            domain="general"
        )
        
        templates["cot_reasoning"] = PromptTemplate(
            name="cot_reasoning",
            template="""
Let's think step by step to solve this problem.

Problem: {problem}

I need to:
1. Understand what the problem is asking
2. Identify the given information
3. Determine what approach to use
4. Work through the solution step by step
5. Verify my answer

Let me work through this:

Step 1 - Understanding the problem:
{step1_placeholder}

Step 2 - Given information:
{step2_placeholder}

Step 3 - Approach:
{step3_placeholder}

Step 4 - Solution:
{step4_placeholder}

Step 5 - Verification:
{step5_placeholder}

Therefore, the answer is: {answer_placeholder}
""",
            prompt_type=PromptType.REASONING,
            description="Chain-of-thought reasoning with structured steps",
            variables=["problem", "step1_placeholder", "step2_placeholder", "step3_placeholder", "step4_placeholder", "step5_placeholder", "answer_placeholder"],
            domain="general"
        )
        
        templates["mathematical_reasoning"] = PromptTemplate(
            name="mathematical_reasoning",
            template="""
Solve this mathematical problem with clear reasoning:

Problem: {problem}

Mathematical Analysis:
1. What type of mathematical problem is this?
2. What mathematical concepts are involved?
3. What formula or method should I use?
4. What are the given values and unknowns?

Solution Process:
Let me work through this systematically:

Given: {given_info}
Find: {find_what}
Method: {method}

Calculations:
{calculations}

Check: {verification}

Final Answer: {final_answer}
""",
            prompt_type=PromptType.REASONING,
            description="Mathematical problem solving template",
            variables=["problem", "given_info", "find_what", "method", "calculations", "verification", "final_answer"],
            domain="mathematics"
        )
        
        templates["logical_reasoning"] = PromptTemplate(
            name="logical_reasoning",
            template="""
Analyze this logical problem systematically:

Problem: {problem}

Logical Analysis:
1. What are the given statements or conditions?
2. What logical relationships exist?
3. What can I deduce from the given information?
4. What reasoning method should I use?

Reasoning Process:
Given statements:
{given_statements}

Logical deductions:
{deductions}

Step-by-step reasoning:
{reasoning_steps}

Conclusion: {conclusion}
""",
            prompt_type=PromptType.REASONING,
            description="Logical problem solving template",
            variables=["problem", "given_statements", "deductions", "reasoning_steps", "conclusion"],
            domain="logic"
        )
        
        templates["code_debugging"] = PromptTemplate(
            name="code_debugging",
            template="""
Debug this code problem systematically:

Problem Description: {problem}

Code: {code}

Expected Output: {expected_output}
Actual Output: {actual_output}

Debugging Process:
1. Code Analysis:
   - What is this code supposed to do?
   - What is the algorithm or logic?

2. Error Identification:
   - Where is the error occurring?
   - What type of error is this?

3. Root Cause Analysis:
   - Why is this error happening?
   - What is the underlying issue?

4. Solution:
   - How can this be fixed?
   - What changes are needed?

Analysis:
{analysis}

Error Type: {error_type}
Root Cause: {root_cause}
Fix: {fix}
Corrected Code: {corrected_code}
""",
            prompt_type=PromptType.REASONING,
            description="Code debugging and analysis template",
            variables=["problem", "code", "expected_output", "actual_output", "analysis", "error_type", "root_cause", "fix", "corrected_code"],
            domain="programming"
        )
        
        # Evaluation Templates
        templates["answer_evaluation"] = PromptTemplate(
            name="answer_evaluation",
            template="""
Evaluate the quality of this answer:

Original Problem: {problem}
Answer to Evaluate: {answer}
Expected Answer: {expected}

Evaluation Criteria:
1. Correctness: Is the answer factually correct?
2. Completeness: Does it fully address the problem?
3. Clarity: Is the explanation clear and understandable?
4. Reasoning: Is the reasoning sound and logical?
5. Accuracy: Are calculations and facts accurate?

Evaluation:
Correctness (0-1): {correctness_score}
Completeness (0-1): {completeness_score}
Clarity (0-1): {clarity_score}
Reasoning (0-1): {reasoning_score}
Accuracy (0-1): {accuracy_score}

Overall Score: {overall_score}
Feedback: {feedback}
""",
            prompt_type=PromptType.EVALUATION,
            description="Answer quality evaluation template",
            variables=["problem", "answer", "expected", "correctness_score", "completeness_score", "clarity_score", "reasoning_score", "accuracy_score", "overall_score", "feedback"],
            domain="evaluation"
        )
        
        templates["reasoning_path_evaluation"] = PromptTemplate(
            name="reasoning_path_evaluation",
            template="""
Evaluate this reasoning path for the given problem:

Problem: {problem}
Reasoning Path: {reasoning_path}

Evaluation Criteria:
1. Logical Coherence (0-1): Are the steps logically connected?
2. Factual Accuracy (0-1): Are the facts and information correct?
3. Relevance (0-1): How relevant are the steps to solving the problem?
4. Completeness (0-1): Does the reasoning cover all necessary aspects?
5. Clarity (0-1): How clear and understandable is the reasoning?

Detailed Assessment:
Logical Coherence: {logical_coherence}
Explanation: {logical_explanation}

Factual Accuracy: {factual_accuracy}
Explanation: {factual_explanation}

Relevance: {relevance}
Explanation: {relevance_explanation}

Completeness: {completeness}
Explanation: {completeness_explanation}

Clarity: {clarity}
Explanation: {clarity_explanation}

Overall Assessment: {overall_assessment}
Recommended Improvements: {improvements}
""",
            prompt_type=PromptType.EVALUATION,
            description="Reasoning path evaluation template",
            variables=["problem", "reasoning_path", "logical_coherence", "logical_explanation", "factual_accuracy", "factual_explanation", "relevance", "relevance_explanation", "completeness", "completeness_explanation", "clarity", "clarity_explanation", "overall_assessment", "improvements"],
            domain="evaluation"
        )
        
        # Optimization Templates
        templates["opro_optimization"] = PromptTemplate(
            name="opro_optimization",
            template="""
You are an expert prompt engineer. Your task is to improve the given prompt to achieve better performance on reasoning tasks.

Current Prompt Performance:
Accuracy: {accuracy}
Confidence: {confidence}
Consistency: {consistency}
Reasoning Quality: {reasoning_quality}

Current Prompt:
{current_prompt}

Performance Issues Identified:
{issues}

Failed Examples:
{failed_examples}

Your task is to create an improved version of this prompt that:
1. Addresses the specific failure modes shown in the failed examples
2. Maintains the core reasoning approach while improving clarity
3. Provides better guidance for the reasoning process
4. Helps the model generate more accurate and consistent responses
5. Improves the overall reasoning quality

Consider these improvement strategies:
- Make instructions more specific and clear
- Add better reasoning structure or framework
- Include relevant examples or guidance
- Improve the step-by-step process
- Address common error patterns

Improved Prompt:
{improved_prompt}

Explanation of Changes:
{explanation}
""",
            prompt_type=PromptType.OPTIMIZATION,
            description="OPRO-style prompt optimization template",
            variables=["accuracy", "confidence", "consistency", "reasoning_quality", "current_prompt", "issues", "failed_examples", "improved_prompt", "explanation"],
            domain="optimization"
        )
        
        templates["textgrad_optimization"] = PromptTemplate(
            name="textgrad_optimization",
            template="""
Analyze the performance feedback for this prompt and provide specific improvement suggestions:

Current Prompt:
{current_prompt}

Performance Feedback:
{performance_feedback}

Error Analysis:
{error_analysis}

Based on this feedback, provide specific, actionable modifications to improve the prompt. Focus on:

1. Clarity Issues: How can the instructions be made clearer?
2. Structure Problems: How can the reasoning structure be improved?
3. Missing Guidance: What additional guidance should be provided?
4. Error Prevention: How can common errors be prevented?
5. Consistency: How can response consistency be improved?

Specific Improvement Suggestions:
1. {suggestion_1}
2. {suggestion_2}
3. {suggestion_3}
4. {suggestion_4}
5. {suggestion_5}

Priority Areas for Improvement:
{priority_areas}

Recommended Changes:
{recommended_changes}
""",
            prompt_type=PromptType.OPTIMIZATION,
            description="TextGrad-style gradient-based prompt optimization template",
            variables=["current_prompt", "performance_feedback", "error_analysis", "suggestion_1", "suggestion_2", "suggestion_3", "suggestion_4", "suggestion_5", "priority_areas", "recommended_changes"],
            domain="optimization"
        )
        
        # System Templates
        templates["system_reasoning"] = PromptTemplate(
            name="system_reasoning",
            template="""
You are an expert reasoning assistant. Your role is to help solve complex problems through structured, logical thinking.

Key Principles:
1. Always think step by step
2. Be precise and accurate in your reasoning
3. Show your work clearly
4. Verify your answers when possible
5. Acknowledge uncertainty when appropriate

Approach:
- Break complex problems into smaller parts
- Use appropriate reasoning methods for each problem type
- Provide clear explanations for each step
- Double-check calculations and logic
- Give confident answers when certain, express uncertainty when not

Remember: Quality reasoning is more important than speed. Take your time to think through problems carefully.
""",
            prompt_type=PromptType.SYSTEM,
            description="System prompt for reasoning tasks",
            variables=[],
            domain="general"
        )
        
        templates["system_evaluation"] = PromptTemplate(
            name="system_evaluation",
            template="""
You are an expert evaluator tasked with assessing the quality of reasoning and answers.

Your evaluation should be:
1. Objective and fair
2. Based on clear criteria
3. Constructive and helpful
4. Specific with examples
5. Balanced in perspective

Evaluation Criteria:
- Correctness: Is the answer factually accurate?
- Completeness: Does it fully address the question?
- Clarity: Is it well-explained and understandable?
- Logic: Is the reasoning sound and valid?
- Evidence: Is it well-supported with appropriate evidence?

Provide scores (0-1) for each criterion and overall feedback for improvement.
""",
            prompt_type=PromptType.SYSTEM,
            description="System prompt for evaluation tasks",
            variables=[],
            domain="evaluation"
        )
        
        # Domain-Specific Templates
        templates["math_word_problems"] = PromptTemplate(
            name="math_word_problems",
            template="""
Solve this math word problem step by step:

Problem: {problem}

Let me work through this systematically:

1. Reading and Understanding:
   - What is the problem asking for?
   - What information am I given?

2. Setting up the problem:
   - What variables do I need?
   - What equations or relationships can I establish?

3. Solving:
   - What mathematical operations or methods should I use?
   - Let me work through the calculations step by step.

4. Checking:
   - Does my answer make sense in the context?
   - Can I verify this answer?

Solution:
Understanding: {understanding}
Setup: {setup}
Calculations: {calculations}
Verification: {verification}
Final Answer: {final_answer}
""",
            prompt_type=PromptType.REASONING,
            description="Template for math word problems",
            variables=["problem", "understanding", "setup", "calculations", "verification", "final_answer"],
            domain="mathematics"
        )
        
        templates["common_sense_qa"] = PromptTemplate(
            name="common_sense_qa",
            template="""
Answer this question using common sense reasoning:

Question: {question}

Let me think about this using everyday knowledge and experience:

1. What is this question really asking?
2. What relevant knowledge or experience applies here?
3. What would be the most reasonable answer?
4. Does this answer make sense in real-world context?

Reasoning:
Question Analysis: {analysis}
Relevant Knowledge: {knowledge}
Logical Reasoning: {reasoning}
Real-world Check: {reality_check}

Answer: {answer}
Explanation: {explanation}
""",
            prompt_type=PromptType.REASONING,
            description="Template for common sense reasoning questions",
            variables=["question", "analysis", "knowledge", "reasoning", "reality_check", "answer", "explanation"],
            domain="common_sense"
        )
        
        return templates
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get a specific template by name."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]
    
    def get_templates_by_type(self, prompt_type: PromptType) -> Dict[str, PromptTemplate]:
        """Get all templates of a specific type."""
        return {name: template for name, template in self.templates.items() 
                if template.prompt_type == prompt_type}
    
    def get_templates_by_domain(self, domain: str) -> Dict[str, PromptTemplate]:
        """Get all templates for a specific domain."""
        return {name: template for name, template in self.templates.items() 
                if template.domain == domain}
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())
    
    def format_template(self, name: str, **kwargs) -> str:
        """Format a template with provided variables."""
        template = self.get_template(name)
        
        # Check if all required variables are provided
        missing_vars = [var for var in template.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing variables for template '{name}': {missing_vars}")
        
        return template.template.format(**kwargs)
    
    def create_few_shot_examples(self, domain: str) -> List[Dict[str, str]]:
        """Create few-shot examples for a specific domain."""
        examples = {
            "mathematics": [
                {
                    "problem": "Sarah has 3 times as many apples as Tom. Together they have 24 apples. How many apples does each person have?",
                    "solution": "Let Tom have x apples. Then Sarah has 3x apples. Together: x + 3x = 24, so 4x = 24, therefore x = 6. Tom has 6 apples and Sarah has 18 apples."
                },
                {
                    "problem": "A rectangle is 3 meters longer than it is wide. If the perimeter is 26 meters, what are the dimensions?",
                    "solution": "Let width = w. Then length = w + 3. Perimeter = 2(length + width) = 2(w + 3 + w) = 2(2w + 3) = 4w + 6 = 26. So 4w = 20, w = 5. Width = 5 meters, Length = 8 meters."
                }
            ],
            "logic": [
                {
                    "problem": "All roses are flowers. Some flowers are red. Can we conclude that some roses are red?",
                    "solution": "No, we cannot conclude that some roses are red. While all roses are flowers, and some flowers are red, the red flowers could be entirely non-rose flowers. We need additional information to make this conclusion."
                }
            ],
            "programming": [
                {
                    "problem": "This function should return the factorial of n, but it has infinite recursion: def factorial(n): if n == 0: return 1; else: return n * factorial(n + 1)",
                    "solution": "The error is in the recursive call. It should be factorial(n - 1) instead of factorial(n + 1). The current code increases n, so it never reaches the base case of n == 0."
                }
            ],
            "common_sense": [
                {
                    "problem": "Why do people typically carry umbrellas on rainy days?",
                    "solution": "People carry umbrellas on rainy days to protect themselves from getting wet. Umbrellas are waterproof and block rain from above, keeping the person dry and comfortable."
                }
            ]
        }
        
        return examples.get(domain, [])
    
    def get_template_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a template."""
        template = self.get_template(name)
        return {
            "name": template.name,
            "description": template.description,
            "type": template.prompt_type.value,
            "domain": template.domain,
            "difficulty": template.difficulty,
            "variables": template.variables,
            "variable_count": len(template.variables),
            "template_length": len(template.template),
            "examples": template.examples or []
        }