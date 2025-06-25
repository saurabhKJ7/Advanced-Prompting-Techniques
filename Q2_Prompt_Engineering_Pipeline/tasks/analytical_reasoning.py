"""
Analytical Reasoning Task Definition

This module defines a collection of analytical reasoning problems that require
pattern recognition, sequence completion, and causal reasoning.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import json
import re


@dataclass
class AnalyticalProblem:
    """Represents a single analytical reasoning problem."""
    id: str
    problem: str
    expected_answer: str
    solution_steps: List[str]
    difficulty: str
    category: str
    patterns: List[str]
    reasoning_type: str


class AnalyticalReasoning:
    """Collection of analytical reasoning problems for testing structured thinking."""
    
    def __init__(self):
        self.problems = self._initialize_problems()
    
    def _initialize_problems(self) -> List[AnalyticalProblem]:
        """Initialize the collection of analytical reasoning problems."""
        return [
            AnalyticalProblem(
                id="analytical_001",
                problem="What comes next in this sequence: 2, 6, 12, 20, 30, ?",
                expected_answer="42",
                solution_steps=[
                    "Look at the differences between consecutive terms",
                    "6 - 2 = 4, 12 - 6 = 6, 20 - 12 = 8, 30 - 20 = 10",
                    "The differences form an arithmetic sequence: 4, 6, 8, 10",
                    "Next difference should be 12",
                    "Therefore, next term = 30 + 12 = 42"
                ],
                difficulty="easy",
                category="sequence_completion",
                patterns=["arithmetic_progression", "second_differences"],
                reasoning_type="pattern_recognition"
            ),
            AnalyticalProblem(
                id="analytical_002",
                problem="If all roses are flowers, and some flowers are red, can we conclude that some roses are red?",
                expected_answer="No, we cannot conclude that some roses are red from the given information.",
                solution_steps=[
                    "Premise 1: All roses are flowers (roses ⊆ flowers)",
                    "Premise 2: Some flowers are red (∃ flowers that are red)",
                    "Question: Are some roses red?",
                    "The red flowers could be non-rose flowers",
                    "We have no information connecting roses specifically to red color",
                    "Therefore, we cannot conclude that some roses are red"
                ],
                difficulty="medium",
                category="logical_reasoning",
                patterns=["syllogism", "set_theory"],
                reasoning_type="deductive_reasoning"
            ),
            AnalyticalProblem(
                id="analytical_003",
                problem="A car travels 60 km in the first hour, 50 km in the second hour, 40 km in the third hour. If this pattern continues, in which hour will the car travel exactly 10 km?",
                expected_answer="6th hour",
                solution_steps=[
                    "Identify the pattern: 60, 50, 40, ...",
                    "Each hour, distance decreases by 10 km",
                    "This is an arithmetic sequence with first term a₁ = 60 and common difference d = -10",
                    "General term: aₙ = 60 - 10(n-1) = 70 - 10n",
                    "Set aₙ = 10: 70 - 10n = 10",
                    "Solve: 10n = 60, n = 6",
                    "The car will travel 10 km in the 6th hour"
                ],
                difficulty="medium",
                category="sequence_completion",
                patterns=["arithmetic_sequence", "linear_decrease"],
                reasoning_type="mathematical_reasoning"
            ),
            AnalyticalProblem(
                id="analytical_004",
                problem="In a certain code, MONDAY is written as LNMCSX. How is FRIDAY written in the same code?",
                expected_answer="EQHCSX",
                solution_steps=[
                    "Compare MONDAY with LNMCSX letter by letter",
                    "M → L (shift back by 1), O → N (shift back by 1)",
                    "N → M (shift back by 1), D → C (shift back by 1)",
                    "A → S (A is 1st letter, S is 19th, shift forward by 18)",
                    "Y → X (Y is 25th letter, X is 24th, shift back by 1)",
                    "Pattern: Most letters shift back by 1, but A shifts to S",
                    "Apply to FRIDAY: F→E, R→Q, I→H, D→C, A→S, Y→X",
                    "FRIDAY becomes EQHCSX"
                ],
                difficulty="hard",
                category="coding_decoding",
                patterns=["letter_shift", "alphabet_cipher"],
                reasoning_type="pattern_recognition"
            ),
            AnalyticalProblem(
                id="analytical_005",
                problem="If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
                expected_answer="5 minutes",
                solution_steps=[
                    "Analyze the given information: 5 machines, 5 minutes, 5 widgets",
                    "Calculate rate per machine: 5 widgets ÷ 5 machines = 1 widget per machine",
                    "Time per widget per machine: 5 minutes ÷ 1 widget = 5 minutes",
                    "Each machine makes 1 widget in 5 minutes",
                    "For 100 machines to make 100 widgets: each machine makes 1 widget",
                    "Since each machine works independently, time remains 5 minutes"
                ],
                difficulty="medium",
                category="rate_problems",
                patterns=["proportional_reasoning", "rate_calculation"],
                reasoning_type="mathematical_reasoning"
            ),
            AnalyticalProblem(
                id="analytical_006",
                problem="A clock shows 3:00. What is the angle between the hour and minute hands?",
                expected_answer="90 degrees",
                solution_steps=[
                    "At 3:00, minute hand points to 12",
                    "Hour hand points to 3",
                    "Clock is divided into 12 equal parts",
                    "Each part represents 360° ÷ 12 = 30°",
                    "From 12 to 3, there are 3 parts",
                    "Angle = 3 × 30° = 90°"
                ],
                difficulty="easy",
                category="geometric_reasoning",
                patterns=["circular_measurement", "angle_calculation"],
                reasoning_type="spatial_reasoning"
            ),
            AnalyticalProblem(
                id="analytical_007",
                problem="If the day before yesterday was Thursday, what day will it be the day after tomorrow?",
                expected_answer="Monday",
                solution_steps=[
                    "Day before yesterday was Thursday",
                    "Yesterday was Friday",
                    "Today is Saturday",
                    "Tomorrow will be Sunday",
                    "Day after tomorrow will be Monday"
                ],
                difficulty="easy",
                category="temporal_reasoning",
                patterns=["day_sequence", "relative_time"],
                reasoning_type="temporal_reasoning"
            ),
            AnalyticalProblem(
                id="analytical_008",
                problem="In a group of 20 people, 12 like coffee, 8 like tea, and 3 like both. How many people like neither coffee nor tea?",
                expected_answer="3 people",
                solution_steps=[
                    "Use the principle of inclusion-exclusion",
                    "People who like coffee only: 12 - 3 = 9",
                    "People who like tea only: 8 - 3 = 5",
                    "People who like both: 3",
                    "Total who like at least one: 9 + 5 + 3 = 17",
                    "People who like neither: 20 - 17 = 3"
                ],
                difficulty="medium",
                category="set_operations",
                patterns=["venn_diagram", "inclusion_exclusion"],
                reasoning_type="set_theory_reasoning"
            ),
            AnalyticalProblem(
                id="analytical_009",
                problem="What is the next number in the sequence: 1, 1, 2, 3, 5, 8, 13, ?",
                expected_answer="21",
                solution_steps=[
                    "Identify the pattern: each number is the sum of the two preceding numbers",
                    "1 + 1 = 2",
                    "1 + 2 = 3", 
                    "2 + 3 = 5",
                    "3 + 5 = 8",
                    "5 + 8 = 13",
                    "8 + 13 = 21",
                    "This is the Fibonacci sequence"
                ],
                difficulty="easy",
                category="sequence_completion",
                patterns=["fibonacci_sequence", "recursive_addition"],
                reasoning_type="pattern_recognition"
            ),
            AnalyticalProblem(
                id="analytical_010",
                problem="A father is 4 times as old as his son. In 20 years, he will be twice as old as his son. How old are they now?",
                expected_answer="Father: 40 years old, Son: 10 years old",
                solution_steps=[
                    "Let son's current age = x years",
                    "Father's current age = 4x years",
                    "In 20 years: son's age = x + 20",
                    "In 20 years: father's age = 4x + 20",
                    "Condition: 4x + 20 = 2(x + 20)",
                    "Solve: 4x + 20 = 2x + 40",
                    "2x = 20, x = 10",
                    "Son is 10 years old, father is 40 years old"
                ],
                difficulty="medium",
                category="age_problems",
                patterns=["algebraic_equation", "age_relationship"],
                reasoning_type="algebraic_reasoning"
            )
        ]
    
    def get_problems(self) -> List[Dict[str, Any]]:
        """Return all problems in dictionary format."""
        return [
            {
                "id": problem.id,
                "problem": problem.problem,
                "expected": problem.expected_answer,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "category": problem.category,
                "patterns": problem.patterns,
                "reasoning_type": problem.reasoning_type
            }
            for problem in self.problems
        ]
    
    def get_problem_by_id(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific problem by ID."""
        for problem in self.problems:
            if problem.id == problem_id:
                return {
                    "id": problem.id,
                    "problem": problem.problem,
                    "expected": problem.expected_answer,
                    "solution_steps": problem.solution_steps,
                    "difficulty": problem.difficulty,
                    "category": problem.category,
                    "patterns": problem.patterns,
                    "reasoning_type": problem.reasoning_type
                }
        return None
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Get problems filtered by difficulty level."""
        return [
            {
                "id": problem.id,
                "problem": problem.problem,
                "expected": problem.expected_answer,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "category": problem.category,
                "patterns": problem.patterns,
                "reasoning_type": problem.reasoning_type
            }
            for problem in self.problems
            if problem.difficulty == difficulty
        ]
    
    def get_problems_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get problems filtered by category."""
        return [
            {
                "id": problem.id,
                "problem": problem.problem,
                "expected": problem.expected_answer,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "category": problem.category,
                "patterns": problem.patterns,
                "reasoning_type": problem.reasoning_type
            }
            for problem in self.problems
            if problem.category == category
        ]
    
    def get_problems_by_reasoning_type(self, reasoning_type: str) -> List[Dict[str, Any]]:
        """Get problems filtered by reasoning type."""
        return [
            {
                "id": problem.id,
                "problem": problem.problem,
                "expected": problem.expected_answer,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "category": problem.category,
                "patterns": problem.patterns,
                "reasoning_type": problem.reasoning_type
            }
            for problem in self.problems
            if problem.reasoning_type == reasoning_type
        ]
    
    def evaluate_solution(self, problem_id: str, solution: str) -> Dict[str, Any]:
        """Evaluate a solution against the expected answer."""
        problem = self.get_problem_by_id(problem_id)
        if not problem:
            return {"error": "Problem not found"}
        
        # Extract key terms and numbers
        expected_terms = self._extract_key_terms(problem["expected"])
        solution_terms = self._extract_key_terms(solution)
        
        # Check numerical accuracy
        numerical_accuracy = self._check_numerical_accuracy(problem, solution)
        
        # Check reasoning approach
        reasoning_approach = self._check_reasoning_approach(problem, solution)
        
        # Check pattern recognition
        pattern_recognition = self._check_pattern_recognition(problem, solution)
        
        # Overall score
        overall_score = (numerical_accuracy + reasoning_approach + pattern_recognition) / 3
        
        return {
            "problem_id": problem_id,
            "correct": overall_score >= 0.7,
            "expected": problem["expected"],
            "provided": solution,
            "scores": {
                "numerical_accuracy": numerical_accuracy,
                "reasoning_approach": reasoning_approach,
                "pattern_recognition": pattern_recognition,
                "overall": overall_score
            },
            "feedback": self._generate_feedback(problem, solution, overall_score)
        }
    
    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract key terms from text for comparison."""
        text_lower = text.lower()
        
        # Extract numbers
        numbers = set(re.findall(r'\d+', text))
        
        # Extract key reasoning terms
        reasoning_terms = {
            'sequence', 'pattern', 'difference', 'arithmetic', 'fibonacci',
            'hour', 'minute', 'angle', 'degree', 'yesterday', 'tomorrow',
            'coffee', 'tea', 'neither', 'both', 'father', 'son', 'age',
            'machine', 'widget', 'rate', 'time', 'roses', 'flowers', 'red'
        }
        
        found_terms = {term for term in reasoning_terms if term in text_lower}
        
        return numbers.union(found_terms)
    
    def _check_numerical_accuracy(self, problem: Dict[str, Any], solution: str) -> float:
        """Check if solution contains correct numerical answer."""
        expected_numbers = set(re.findall(r'\d+', problem["expected"]))
        solution_numbers = set(re.findall(r'\d+', solution))
        
        if not expected_numbers:
            return 1.0  # No numbers to check
        
        # Check if key numbers are present
        correct_numbers = expected_numbers.intersection(solution_numbers)
        return len(correct_numbers) / len(expected_numbers)
    
    def _check_reasoning_approach(self, problem: Dict[str, Any], solution: str) -> float:
        """Check if solution follows appropriate reasoning approach."""
        solution_lower = solution.lower()
        reasoning_type = problem["reasoning_type"]
        
        approach_indicators = {
            "pattern_recognition": ["pattern", "sequence", "next", "follows", "trend"],
            "deductive_reasoning": ["therefore", "conclude", "premise", "given", "if"],
            "mathematical_reasoning": ["equation", "solve", "calculate", "formula", "arithmetic"],
            "spatial_reasoning": ["angle", "degree", "position", "direction", "space"],
            "temporal_reasoning": ["yesterday", "tomorrow", "before", "after", "day"],
            "set_theory_reasoning": ["union", "intersection", "both", "neither", "only"],
            "algebraic_reasoning": ["equation", "variable", "solve", "let", "substitute"]
        }
        
        relevant_indicators = approach_indicators.get(reasoning_type, [])
        mentioned_indicators = sum(1 for indicator in relevant_indicators if indicator in solution_lower)
        
        return min(mentioned_indicators / len(relevant_indicators), 1.0) if relevant_indicators else 0.0
    
    def _check_pattern_recognition(self, problem: Dict[str, Any], solution: str) -> float:
        """Check if solution recognizes the underlying patterns."""
        solution_lower = solution.lower()
        patterns = problem["patterns"]
        
        pattern_keywords = {
            "arithmetic_progression": ["arithmetic", "difference", "constant", "add"],
            "second_differences": ["second", "difference", "pattern", "sequence"],
            "fibonacci_sequence": ["fibonacci", "sum", "preceding", "add"],
            "letter_shift": ["shift", "alphabet", "letter", "code"],
            "proportional_reasoning": ["proportion", "rate", "ratio", "per"],
            "inclusion_exclusion": ["both", "neither", "only", "overlap"],
            "venn_diagram": ["venn", "overlap", "intersection", "union"],
            "age_relationship": ["age", "times", "years", "older"],
            "circular_measurement": ["circle", "clock", "angle", "degree"],
            "day_sequence": ["day", "sequence", "order", "week"]
        }
        
        total_score = 0
        for pattern in patterns:
            keywords = pattern_keywords.get(pattern, [])
            if keywords:
                mentioned = sum(1 for keyword in keywords if keyword in solution_lower)
                total_score += min(mentioned / len(keywords), 1.0)
        
        return total_score / len(patterns) if patterns else 0.0
    
    def _generate_feedback(self, problem: Dict[str, Any], solution: str, score: float) -> str:
        """Generate feedback for the solution."""
        if score >= 0.8:
            return "Excellent! You demonstrated strong analytical reasoning and arrived at the correct answer."
        elif score >= 0.6:
            return "Good reasoning! You're on the right track but could strengthen some aspects of your analysis."
        elif score >= 0.4:
            return "Partial understanding shown. Focus on identifying the key patterns and relationships."
        else:
            category = problem["category"]
            if category == "sequence_completion":
                return "Look for patterns in differences between consecutive terms or relationships between terms."
            elif category == "logical_reasoning":
                return "Carefully analyze the logical relationships between the given statements."
            elif category == "coding_decoding":
                return "Compare the original and coded versions letter by letter to find the transformation pattern."
            elif category == "rate_problems":
                return "Consider the rate per unit and how it scales with different quantities."
            elif category == "geometric_reasoning":
                return "Think about the geometric relationships and apply relevant formulas."
            elif category == "temporal_reasoning":
                return "Map out the sequence of days step by step from the given reference point."
            elif category == "set_operations":
                return "Use set theory principles like inclusion-exclusion to solve the problem."
            elif category == "age_problems":
                return "Set up algebraic equations based on the current and future age relationships."
            else:
                return "Break down the problem systematically and look for underlying patterns or relationships."
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the problem collection."""
        categories = {}
        difficulties = {}
        reasoning_types = {}
        
        for problem in self.problems:
            categories[problem.category] = categories.get(problem.category, 0) + 1
            difficulties[problem.difficulty] = difficulties.get(problem.difficulty, 0) + 1
            reasoning_types[problem.reasoning_type] = reasoning_types.get(problem.reasoning_type, 0) + 1
        
        return {
            "total_problems": len(self.problems),
            "categories": categories,
            "difficulties": difficulties,
            "reasoning_types": reasoning_types,
            "average_steps": sum(len(p.solution_steps) for p in self.problems) / len(self.problems),
            "average_patterns": sum(len(p.patterns) for p in self.problems) / len(self.problems)
        }
    
    def export_to_json(self, filename: str) -> None:
        """Export problems to JSON file."""
        problems_data = {
            "metadata": {
                "total_problems": len(self.problems),
                "categories": list(set(p.category for p in self.problems)),
                "difficulties": list(set(p.difficulty for p in self.problems)),
                "reasoning_types": list(set(p.reasoning_type for p in self.problems))
            },
            "problems": self.get_problems()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(problems_data, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, filename: str) -> None:
        """Load problems from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.problems = []
        for prob_data in data.get("problems", []):
            problem = AnalyticalProblem(
                id=prob_data["id"],
                problem=prob_data["problem"],
                expected_answer=prob_data["expected"],
                solution_steps=prob_data["solution_steps"],
                difficulty=prob_data["difficulty"],
                category=prob_data["category"],
                patterns=prob_data["patterns"],
                reasoning_type=prob_data["reasoning_type"]
            )
            self.problems.append(problem)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the analytical reasoning collection
    analytical = AnalyticalReasoning()
    
    # Display statistics
    print("Analytical Reasoning Problems Collection")
    print("=" * 40)
    stats = analytical.get_statistics()
    print(f"Total Problems: {stats['total_problems']}")
    print(f"Categories: {', '.join(stats['categories'].keys())}")
    print(f"Difficulties: {', '.join(stats['difficulties'].keys())}")
    print(f"Reasoning Types: {', '.join(stats['reasoning_types'].keys())}")
    print(f"Average Solution Steps: {stats['average_steps']:.1f}")
    print(f"Average Patterns: {stats['average_patterns']:.1f}")
    
    # Test a few problems
    print("\nSample Problems:")
    print("-" * 20)
    
    for i, problem in enumerate(analytical.get_problems()[:3]):
        print(f"\nProblem {i+1}: {problem['problem']}")
        print(f"Expected: {problem['expected']}")
        print(f"Category: {problem['category']}")
        print(f"Difficulty: {problem['difficulty']}")
        print(f"Reasoning Type: {problem['reasoning_type']}")
        print(f"Patterns: {', '.join(problem['patterns'])}")
    
    # Test evaluation
    print("\nEvaluation Test:")
    print("-" * 20)
    
    test_solution = "The differences between consecutive terms are 4, 6, 8, 10, forming an arithmetic sequence. The next difference should be 12, so the next term is 30 + 12 = 42."
    result = analytical.evaluate_solution("analytical_001", test_solution)
    print(f"Test Solution: {test_solution}")
    print(f"Evaluation Result: {result}")