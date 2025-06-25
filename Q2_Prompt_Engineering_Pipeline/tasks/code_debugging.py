"""
Code Debugging Task Definition

This module defines a collection of code debugging problems that require
systematic error identification, analysis, and correction.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import json
import re


@dataclass
class CodeDebugProblem:
    """Represents a single code debugging problem."""
    id: str
    problem: str
    buggy_code: str
    expected_output: str
    actual_output: str
    bug_type: str
    expected_fix: str
    solution_steps: List[str]
    difficulty: str
    language: str
    concepts: List[str]


class CodeDebugging:
    """Collection of code debugging problems for testing analytical reasoning."""
    
    def __init__(self):
        self.problems = self._initialize_problems()
    
    def _initialize_problems(self) -> List[CodeDebugProblem]:
        """Initialize the collection of code debugging problems."""
        return [
            CodeDebugProblem(
                id="debug_001",
                problem="This function should calculate the factorial of a number, but it's giving wrong results for some inputs.",
                buggy_code="""def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n + 1)""",
                expected_output="factorial(5) should return 120",
                actual_output="RecursionError: maximum recursion depth exceeded",
                bug_type="infinite_recursion",
                expected_fix="""def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)""",
                solution_steps=[
                    "Identify the recursive call: factorial(n + 1)",
                    "Recognize this creates infinite recursion (n keeps increasing)",
                    "The base case n == 0 will never be reached",
                    "Change recursive call to factorial(n - 1) to decrease n",
                    "This ensures we eventually reach the base case"
                ],
                difficulty="easy",
                language="python",
                concepts=["recursion", "base_case", "infinite_loop"]
            ),
            CodeDebugProblem(
                id="debug_002",
                problem="This function should find the maximum value in a list, but it returns incorrect results.",
                buggy_code="""def find_max(numbers):
    max_val = 0
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val""",
                expected_output="find_max([-5, -2, -10, -1]) should return -1",
                actual_output="find_max([-5, -2, -10, -1]) returns 0",
                bug_type="initialization_error",
                expected_fix="""def find_max(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val""",
                solution_steps=[
                    "Identify the issue: max_val initialized to 0",
                    "Problem occurs when all numbers are negative",
                    "0 is greater than all negative numbers",
                    "Initialize max_val to first element of list",
                    "Add check for empty list to avoid IndexError"
                ],
                difficulty="easy",
                language="python",
                concepts=["initialization", "edge_cases", "negative_numbers"]
            ),
            CodeDebugProblem(
                id="debug_003",
                problem="This function should reverse a string, but it's not working correctly.",
                buggy_code="""def reverse_string(s):
    reversed_str = ""
    for i in range(len(s)):
        reversed_str += s[i]
    return reversed_str""",
                expected_output="reverse_string('hello') should return 'olleh'",
                actual_output="reverse_string('hello') returns 'hello'",
                bug_type="logic_error",
                expected_fix="""def reverse_string(s):
    reversed_str = ""
    for i in range(len(s) - 1, -1, -1):
        reversed_str += s[i]
    return reversed_str""",
                solution_steps=[
                    "Current loop: for i in range(len(s)) iterates 0 to len(s)-1",
                    "This accesses characters in original order",
                    "To reverse, need to iterate backwards",
                    "Change to range(len(s)-1, -1, -1) to go from last to first",
                    "Or use simpler approach: reversed_str += s[len(s)-1-i]"
                ],
                difficulty="easy",
                language="python",
                concepts=["string_manipulation", "loop_direction", "indexing"]
            ),
            CodeDebugProblem(
                id="debug_004",
                problem="This function should check if a number is prime, but it has logical errors.",
                buggy_code="""def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return True
    return False""",
                expected_output="is_prime(7) should return True, is_prime(8) should return False",
                actual_output="is_prime(7) returns False, is_prime(8) returns True",
                bug_type="logic_error",
                expected_fix="""def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True""",
                solution_steps=[
                    "Logic is inverted: returns True when divisor found (should be False)",
                    "Returns False when no divisors found (should be True)",
                    "Fix: return False when n % i == 0 (found divisor)",
                    "Fix: return True at end (no divisors found)",
                    "Optimization: only check up to sqrt(n) for efficiency"
                ],
                difficulty="medium",
                language="python",
                concepts=["prime_numbers", "modulo", "boolean_logic", "optimization"]
            ),
            CodeDebugProblem(
                id="debug_005",
                problem="This binary search implementation has a bug that causes incorrect results.",
                buggy_code="""def binary_search(arr, target):
    left = 0
    right = len(arr)
    
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    
    return -1""",
                expected_output="binary_search([1, 2, 3, 4, 5], 3) should return 2",
                actual_output="Function enters infinite loop",
                bug_type="infinite_loop",
                expected_fix="""def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
                solution_steps=[
                    "Issue 1: right initialized to len(arr) should be len(arr) - 1",
                    "Issue 2: condition should be left <= right, not left < right",
                    "Issue 3: left = mid should be left = mid + 1",
                    "Issue 4: right = mid should be right = mid - 1",
                    "Without +1 and -1, left and right don't change, causing infinite loop"
                ],
                difficulty="hard",
                language="python",
                concepts=["binary_search", "loop_termination", "boundary_conditions"]
            ),
            CodeDebugProblem(
                id="debug_006",
                problem="This function should merge two sorted lists, but it's missing elements.",
                buggy_code="""def merge_sorted_lists(list1, list2):
    result = []
    i = j = 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    
    return result""",
                expected_output="merge_sorted_lists([1, 3, 5], [2, 4, 6, 7]) should return [1, 2, 3, 4, 5, 6, 7]",
                actual_output="merge_sorted_lists([1, 3, 5], [2, 4, 6, 7]) returns [1, 2, 3, 4, 5]",
                bug_type="incomplete_logic",
                expected_fix="""def merge_sorted_lists(list1, list2):
    result = []
    i = j = 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    
    # Add remaining elements from list1
    while i < len(list1):
        result.append(list1[i])
        i += 1
    
    # Add remaining elements from list2
    while j < len(list2):
        result.append(list2[j])
        j += 1
    
    return result""",
                solution_steps=[
                    "Main loop stops when one list is exhausted",
                    "Remaining elements in the other list are not added",
                    "Need to add remaining elements from list1 if any",
                    "Need to add remaining elements from list2 if any",
                    "Use while loops to append remaining elements"
                ],
                difficulty="medium",
                language="python",
                concepts=["merge_algorithm", "two_pointers", "remaining_elements"]
            ),
            CodeDebugProblem(
                id="debug_007",
                problem="This class method should calculate compound interest, but the calculations are wrong.",
                buggy_code="""class Investment:
    def __init__(self, principal, rate, time):
        self.principal = principal
        self.rate = rate
        self.time = time
    
    def compound_interest(self):
        amount = self.principal * (1 + self.rate) ** self.time
        interest = amount - self.principal
        return interest""",
                expected_output="Investment(1000, 0.05, 2).compound_interest() should return 102.5",
                actual_output="Investment(1000, 0.05, 2).compound_interest() returns 102.5 (actually correct)",
                bug_type="misunderstanding",
                expected_fix="The code is actually correct. The issue might be in understanding or test case.",
                solution_steps=[
                    "Check the compound interest formula: A = P(1 + r)^t",
                    "Principal = 1000, rate = 0.05, time = 2",
                    "Amount = 1000 * (1 + 0.05)^2 = 1000 * 1.1025 = 1102.5",
                    "Interest = 1102.5 - 1000 = 102.5",
                    "The calculation is correct, verify the expected output"
                ],
                difficulty="easy",
                language="python",
                concepts=["compound_interest", "formula_verification", "mathematical_calculation"]
            )
        ]
    
    def get_problems(self) -> List[Dict[str, Any]]:
        """Return all problems in dictionary format."""
        return [
            {
                "id": problem.id,
                "problem": problem.problem,
                "buggy_code": problem.buggy_code,
                "expected_output": problem.expected_output,
                "actual_output": problem.actual_output,
                "bug_type": problem.bug_type,
                "expected_fix": problem.expected_fix,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "language": problem.language,
                "concepts": problem.concepts
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
                    "buggy_code": problem.buggy_code,
                    "expected_output": problem.expected_output,
                    "actual_output": problem.actual_output,
                    "bug_type": problem.bug_type,
                    "expected_fix": problem.expected_fix,
                    "solution_steps": problem.solution_steps,
                    "difficulty": problem.difficulty,
                    "language": problem.language,
                    "concepts": problem.concepts
                }
        return None
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Get problems filtered by difficulty level."""
        return [
            {
                "id": problem.id,
                "problem": problem.problem,
                "buggy_code": problem.buggy_code,
                "expected_output": problem.expected_output,
                "actual_output": problem.actual_output,
                "bug_type": problem.bug_type,
                "expected_fix": problem.expected_fix,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "language": problem.language,
                "concepts": problem.concepts
            }
            for problem in self.problems
            if problem.difficulty == difficulty
        ]
    
    def get_problems_by_bug_type(self, bug_type: str) -> List[Dict[str, Any]]:
        """Get problems filtered by bug type."""
        return [
            {
                "id": problem.id,
                "problem": problem.problem,
                "buggy_code": problem.buggy_code,
                "expected_output": problem.expected_output,
                "actual_output": problem.actual_output,
                "bug_type": problem.bug_type,
                "expected_fix": problem.expected_fix,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "language": problem.language,
                "concepts": problem.concepts
            }
            for problem in self.problems
            if problem.bug_type == bug_type
        ]
    
    def evaluate_solution(self, problem_id: str, solution: str) -> Dict[str, Any]:
        """Evaluate a debugging solution against the expected fix."""
        problem = self.get_problem_by_id(problem_id)
        if not problem:
            return {"error": "Problem not found"}
        
        # Check if key bug identification terms are present
        bug_identification = self._check_bug_identification(problem, solution)
        
        # Check if solution addresses the core issue
        core_issue_addressed = self._check_core_issue(problem, solution)
        
        # Check if correct fix approach is mentioned
        fix_approach = self._check_fix_approach(problem, solution)
        
        # Calculate overall score
        overall_score = (bug_identification + core_issue_addressed + fix_approach) / 3
        
        return {
            "problem_id": problem_id,
            "correct": overall_score >= 0.7,
            "expected_fix": problem["expected_fix"],
            "provided": solution,
            "scores": {
                "bug_identification": bug_identification,
                "core_issue_addressed": core_issue_addressed,
                "fix_approach": fix_approach,
                "overall": overall_score
            },
            "feedback": self._generate_feedback(problem, solution, overall_score)
        }
    
    def _check_bug_identification(self, problem: Dict[str, Any], solution: str) -> float:
        """Check if the solution correctly identifies the bug type."""
        solution_lower = solution.lower()
        bug_type = problem["bug_type"]
        
        bug_keywords = {
            "infinite_recursion": ["recursion", "infinite", "base case", "stack overflow"],
            "initialization_error": ["initialization", "initial value", "zero", "first element"],
            "logic_error": ["logic", "condition", "boolean", "wrong return"],
            "infinite_loop": ["infinite loop", "loop", "termination", "condition"],
            "incomplete_logic": ["missing", "incomplete", "remaining", "leftover"],
            "misunderstanding": ["correct", "misunderstanding", "expected", "formula"]
        }
        
        relevant_keywords = bug_keywords.get(bug_type, [])
        mentioned_keywords = sum(1 for keyword in relevant_keywords if keyword in solution_lower)
        
        return min(mentioned_keywords / len(relevant_keywords), 1.0) if relevant_keywords else 0.0
    
    def _check_core_issue(self, problem: Dict[str, Any], solution: str) -> float:
        """Check if solution addresses the core issue."""
        solution_lower = solution.lower()
        
        # Extract key concepts from the problem
        concepts = problem["concepts"]
        mentioned_concepts = sum(1 for concept in concepts if concept in solution_lower)
        
        return min(mentioned_concepts / len(concepts), 1.0) if concepts else 0.0
    
    def _check_fix_approach(self, problem: Dict[str, Any], solution: str) -> float:
        """Check if solution mentions correct fix approach."""
        solution_lower = solution.lower()
        expected_fix = problem["expected_fix"].lower()
        
        # Extract key terms from expected fix
        fix_terms = set()
        
        # Common programming terms that indicate fixes
        if "n - 1" in expected_fix:
            fix_terms.add("n - 1")
        if "numbers[0]" in expected_fix:
            fix_terms.add("first element")
        if "range(" in expected_fix and "-1" in expected_fix:
            fix_terms.add("backwards")
        if "return false" in expected_fix:
            fix_terms.add("return false")
        if "mid + 1" in expected_fix or "mid - 1" in expected_fix:
            fix_terms.add("mid + 1")
            fix_terms.add("mid - 1")
        if "while" in expected_fix and "remaining" in problem["solution_steps"][-1]:
            fix_terms.add("remaining elements")
        
        if not fix_terms:
            # Fallback: check for general fix indicators
            fix_indicators = ["change", "fix", "correct", "modify", "update", "add"]
            mentioned_indicators = sum(1 for indicator in fix_indicators if indicator in solution_lower)
            return min(mentioned_indicators / 2, 1.0)
        
        mentioned_terms = sum(1 for term in fix_terms if term in solution_lower)
        return min(mentioned_terms / len(fix_terms), 1.0)
    
    def _generate_feedback(self, problem: Dict[str, Any], solution: str, score: float) -> str:
        """Generate feedback for the debugging solution."""
        if score >= 0.8:
            return "Excellent! You correctly identified the bug and provided a good solution approach."
        elif score >= 0.6:
            return "Good analysis! You identified most of the key issues. Consider elaborating on the fix."
        elif score >= 0.4:
            return "Partial understanding. You identified some issues but missed key aspects of the bug."
        else:
            feedback = "Consider focusing on: "
            
            if problem["bug_type"] == "infinite_recursion":
                feedback += "the recursive call direction and base case reachability."
            elif problem["bug_type"] == "initialization_error":
                feedback += "how the initial value affects edge cases like negative numbers."
            elif problem["bug_type"] == "logic_error":
                feedback += "the logical flow and return conditions."
            elif problem["bug_type"] == "infinite_loop":
                feedback += "the loop termination conditions and variable updates."
            elif problem["bug_type"] == "incomplete_logic":
                feedback += "what happens to remaining elements after the main loop."
            else:
                feedback += "the core logic and expected behavior."
            
            return feedback
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the problem collection."""
        bug_types = {}
        difficulties = {}
        languages = {}
        
        for problem in self.problems:
            bug_types[problem.bug_type] = bug_types.get(problem.bug_type, 0) + 1
            difficulties[problem.difficulty] = difficulties.get(problem.difficulty, 0) + 1
            languages[problem.language] = languages.get(problem.language, 0) + 1
        
        return {
            "total_problems": len(self.problems),
            "bug_types": bug_types,
            "difficulties": difficulties,
            "languages": languages,
            "average_steps": sum(len(p.solution_steps) for p in self.problems) / len(self.problems),
            "average_concepts": sum(len(p.concepts) for p in self.problems) / len(self.problems)
        }
    
    def export_to_json(self, filename: str) -> None:
        """Export problems to JSON file."""
        problems_data = {
            "metadata": {
                "total_problems": len(self.problems),
                "bug_types": list(set(p.bug_type for p in self.problems)),
                "difficulties": list(set(p.difficulty for p in self.problems)),
                "languages": list(set(p.language for p in self.problems))
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
            problem = CodeDebugProblem(
                id=prob_data["id"],
                problem=prob_data["problem"],
                buggy_code=prob_data["buggy_code"],
                expected_output=prob_data["expected_output"],
                actual_output=prob_data["actual_output"],
                bug_type=prob_data["bug_type"],
                expected_fix=prob_data["expected_fix"],
                solution_steps=prob_data["solution_steps"],
                difficulty=prob_data["difficulty"],
                language=prob_data["language"],
                concepts=prob_data["concepts"]
            )
            self.problems.append(problem)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the code debugging collection
    code_debug = CodeDebugging()
    
    # Display statistics
    print("Code Debugging Problems Collection")
    print("=" * 40)
    stats = code_debug.get_statistics()
    print(f"Total Problems: {stats['total_problems']}")
    print(f"Bug Types: {', '.join(stats['bug_types'].keys())}")
    print(f"Difficulties: {', '.join(stats['difficulties'].keys())}")
    print(f"Languages: {', '.join(stats['languages'].keys())}")
    print(f"Average Solution Steps: {stats['average_steps']:.1f}")
    print(f"Average Concepts: {stats['average_concepts']:.1f}")
    
    # Test a few problems
    print("\nSample Problems:")
    print("-" * 20)
    
    for i, problem in enumerate(code_debug.get_problems()[:2]):
        print(f"\nProblem {i+1}: {problem['problem']}")
        print(f"Bug Type: {problem['bug_type']}")
        print(f"Difficulty: {problem['difficulty']}")
        print("Buggy Code:")
        print(problem['buggy_code'])
        print(f"Expected Output: {problem['expected_output']}")
        print(f"Actual Output: {problem['actual_output']}")
    
    # Test evaluation
    print("\nEvaluation Test:")
    print("-" * 20)
    
    test_solution = "The recursive call is going in the wrong direction. It should be factorial(n-1) instead of factorial(n+1) to reach the base case."
    result = code_debug.evaluate_solution("debug_001", test_solution)
    print(f"Test Solution: {test_solution}")
    print(f"Evaluation Result: {result}")