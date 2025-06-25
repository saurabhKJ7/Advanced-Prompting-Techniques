"""
Math Word Problems Task Definition

This module defines a collection of multi-step math word problems that require
structured reasoning and algebraic manipulation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re


@dataclass
class MathProblem:
    """Represents a single math word problem."""
    id: str
    problem: str
    expected_answer: str
    solution_steps: List[str]
    difficulty: str
    category: str
    keywords: List[str]


class MathWordProblems:
    """Collection of math word problems for testing reasoning capabilities."""
    
    def __init__(self):
        self.problems = self._initialize_problems()
    
    def _initialize_problems(self) -> List[MathProblem]:
        """Initialize the collection of math problems."""
        return [
            MathProblem(
                id="math_001",
                problem="Sarah has 3 times as many apples as Tom. Together they have 24 apples. How many apples does each person have?",
                expected_answer="Tom: 6 apples, Sarah: 18 apples",
                solution_steps=[
                    "Let Tom have x apples",
                    "Then Sarah has 3x apples",
                    "Together: x + 3x = 24",
                    "4x = 24",
                    "x = 6",
                    "Tom: 6 apples, Sarah: 18 apples"
                ],
                difficulty="easy",
                category="algebra",
                keywords=["ratio", "system", "variables"]
            ),
            MathProblem(
                id="math_002",
                problem="A rectangular garden is 3 meters longer than it is wide. If the perimeter is 26 meters, what are the dimensions of the garden?",
                expected_answer="Width: 5 meters, Length: 8 meters",
                solution_steps=[
                    "Let width = w meters",
                    "Then length = w + 3 meters",
                    "Perimeter = 2(length + width) = 26",
                    "2(w + 3 + w) = 26",
                    "2(2w + 3) = 26",
                    "4w + 6 = 26",
                    "4w = 20",
                    "w = 5",
                    "Width: 5 meters, Length: 8 meters"
                ],
                difficulty="medium",
                category="geometry",
                keywords=["rectangle", "perimeter", "dimensions"]
            ),
            MathProblem(
                id="math_003",
                problem="A store sells notebooks for $3 each and pens for $2 each. Maria bought 8 items total and spent $21. How many notebooks and pens did she buy?",
                expected_answer="5 notebooks and 3 pens",
                solution_steps=[
                    "Let n = number of notebooks, p = number of pens",
                    "n + p = 8 (total items)",
                    "3n + 2p = 21 (total cost)",
                    "From first equation: p = 8 - n",
                    "Substitute: 3n + 2(8 - n) = 21",
                    "3n + 16 - 2n = 21",
                    "n = 5",
                    "p = 8 - 5 = 3",
                    "5 notebooks and 3 pens"
                ],
                difficulty="medium",
                category="system_equations",
                keywords=["system", "substitution", "word_problem"]
            ),
            MathProblem(
                id="math_004",
                problem="A train travels 120 miles in 2 hours. Another train travels 180 miles in 3 hours. If both trains maintain their speeds, how long will it take for the faster train to travel 300 miles?",
                expected_answer="5 hours",
                solution_steps=[
                    "Train 1 speed: 120 miles ÷ 2 hours = 60 mph",
                    "Train 2 speed: 180 miles ÷ 3 hours = 60 mph",
                    "Both trains have the same speed: 60 mph",
                    "Time for 300 miles: 300 miles ÷ 60 mph = 5 hours"
                ],
                difficulty="easy",
                category="rate_time_distance",
                keywords=["speed", "distance", "time", "rate"]
            ),
            MathProblem(
                id="math_005",
                problem="In a class of 30 students, the ratio of boys to girls is 3:2. If 4 more girls join the class, what will be the new ratio of boys to girls?",
                expected_answer="3:2.4 or 15:12",
                solution_steps=[
                    "Total students = 30",
                    "Ratio boys:girls = 3:2",
                    "Total ratio parts = 3 + 2 = 5",
                    "Boys = (3/5) × 30 = 18",
                    "Girls = (2/5) × 30 = 12",
                    "After 4 girls join: Girls = 12 + 4 = 16",
                    "New ratio = 18:16 = 9:8"
                ],
                difficulty="medium",
                category="ratios",
                keywords=["ratio", "proportion", "change"]
            ),
            MathProblem(
                id="math_006",
                problem="A water tank can be filled by pipe A in 6 hours and by pipe B in 4 hours. If both pipes work together, how long will it take to fill the tank?",
                expected_answer="2.4 hours or 2 hours 24 minutes",
                solution_steps=[
                    "Pipe A rate: 1/6 tank per hour",
                    "Pipe B rate: 1/4 tank per hour",
                    "Combined rate: 1/6 + 1/4",
                    "Find common denominator: 2/12 + 3/12 = 5/12",
                    "Time = 1 ÷ (5/12) = 12/5 = 2.4 hours",
                    "2.4 hours = 2 hours 24 minutes"
                ],
                difficulty="medium",
                category="work_rate",
                keywords=["rate", "combined_work", "time"]
            ),
            MathProblem(
                id="math_007",
                problem="The sum of three consecutive integers is 72. What are the three integers?",
                expected_answer="23, 24, 25",
                solution_steps=[
                    "Let the three consecutive integers be n, n+1, n+2",
                    "Sum: n + (n+1) + (n+2) = 72",
                    "3n + 3 = 72",
                    "3n = 69",
                    "n = 23",
                    "The integers are 23, 24, 25"
                ],
                difficulty="easy",
                category="consecutive_numbers",
                keywords=["consecutive", "integers", "sum"]
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
                "keywords": problem.keywords
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
                    "keywords": problem.keywords
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
                "keywords": problem.keywords
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
                "keywords": problem.keywords
            }
            for problem in self.problems
            if problem.category == category
        ]
    
    def evaluate_solution(self, problem_id: str, solution: str) -> Dict[str, Any]:
        """Evaluate a solution against the expected answer."""
        problem = self.get_problem_by_id(problem_id)
        if not problem:
            return {"error": "Problem not found"}
        
        # Extract numerical values from both expected and provided solutions
        expected_numbers = self._extract_numbers(problem["expected"])
        solution_numbers = self._extract_numbers(solution)
        
        # Check if the key numbers match
        numbers_match = set(expected_numbers) == set(solution_numbers)
        
        # Additional semantic matching could be added here
        semantic_match = self._semantic_match(problem["expected"], solution)
        
        return {
            "problem_id": problem_id,
            "correct": numbers_match or semantic_match,
            "expected": problem["expected"],
            "provided": solution,
            "score": 1.0 if (numbers_match or semantic_match) else 0.0,
            "feedback": self._generate_feedback(problem, solution, numbers_match, semantic_match)
        }
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        # Find all numbers (including decimals)
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(num) for num in numbers]
    
    def _semantic_match(self, expected: str, solution: str) -> bool:
        """Check for semantic similarity between expected and provided solutions."""
        # Simple keyword matching - could be enhanced with NLP
        expected_lower = expected.lower()
        solution_lower = solution.lower()
        
        # Check if key terms are present
        key_terms = ["tom", "sarah", "width", "length", "notebooks", "pens", "boys", "girls", "hours", "minutes"]
        
        for term in key_terms:
            if term in expected_lower and term in solution_lower:
                return True
        
        return False
    
    def _generate_feedback(self, problem: Dict[str, Any], solution: str, 
                          numbers_match: bool, semantic_match: bool) -> str:
        """Generate feedback for the solution."""
        if numbers_match or semantic_match:
            return "Correct! The solution matches the expected answer."
        
        feedback = "Incorrect. "
        
        # Provide hints based on the problem category
        if problem["category"] == "algebra":
            feedback += "Check your variable definitions and algebraic manipulations."
        elif problem["category"] == "geometry":
            feedback += "Verify your formula usage and calculations for geometric properties."
        elif problem["category"] == "system_equations":
            feedback += "Review your system of equations setup and solution method."
        elif problem["category"] == "rate_time_distance":
            feedback += "Double-check your rate calculations and formula applications."
        elif problem["category"] == "ratios":
            feedback += "Ensure you're correctly handling ratio operations and proportions."
        elif problem["category"] == "work_rate":
            feedback += "Verify your rate calculations and combined work formulas."
        else:
            feedback += "Review your problem setup and solution steps."
        
        return feedback
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the problem collection."""
        categories = {}
        difficulties = {}
        
        for problem in self.problems:
            categories[problem.category] = categories.get(problem.category, 0) + 1
            difficulties[problem.difficulty] = difficulties.get(problem.difficulty, 0) + 1
        
        return {
            "total_problems": len(self.problems),
            "categories": categories,
            "difficulties": difficulties,
            "average_steps": sum(len(p.solution_steps) for p in self.problems) / len(self.problems)
        }
    
    def export_to_json(self, filename: str) -> None:
        """Export problems to JSON file."""
        problems_data = {
            "metadata": {
                "total_problems": len(self.problems),
                "categories": list(set(p.category for p in self.problems)),
                "difficulties": list(set(p.difficulty for p in self.problems))
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
            problem = MathProblem(
                id=prob_data["id"],
                problem=prob_data["problem"],
                expected_answer=prob_data["expected"],
                solution_steps=prob_data["solution_steps"],
                difficulty=prob_data["difficulty"],
                category=prob_data["category"],
                keywords=prob_data["keywords"]
            )
            self.problems.append(problem)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the math problems collection
    math_problems = MathWordProblems()
    
    # Display statistics
    print("Math Word Problems Collection")
    print("=" * 40)
    stats = math_problems.get_statistics()
    print(f"Total Problems: {stats['total_problems']}")
    print(f"Categories: {', '.join(stats['categories'].keys())}")
    print(f"Difficulties: {', '.join(stats['difficulties'].keys())}")
    print(f"Average Solution Steps: {stats['average_steps']:.1f}")
    
    # Test a few problems
    print("\nSample Problems:")
    print("-" * 20)
    
    for i, problem in enumerate(math_problems.get_problems()[:3]):
        print(f"\nProblem {i+1}: {problem['problem']}")
        print(f"Expected: {problem['expected']}")
        print(f"Category: {problem['category']}")
        print(f"Difficulty: {problem['difficulty']}")
    
    # Test evaluation
    print("\nEvaluation Test:")
    print("-" * 20)
    
    test_solution = "Tom has 6 apples and Sarah has 18 apples"
    result = math_problems.evaluate_solution("math_001", test_solution)
    print(f"Test Solution: {test_solution}")
    print(f"Evaluation Result: {result}")