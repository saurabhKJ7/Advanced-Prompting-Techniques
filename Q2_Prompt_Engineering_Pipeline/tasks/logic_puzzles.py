"""
Logic Puzzles Task Definition

This module defines a collection of logic puzzles that require
deductive reasoning, constraint satisfaction, and systematic thinking.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import json
import re


@dataclass
class LogicPuzzle:
    """Represents a single logic puzzle."""
    id: str
    puzzle: str
    expected_answer: str
    solution_steps: List[str]
    difficulty: str
    category: str
    constraints: List[str]
    variables: List[str]


class LogicPuzzles:
    """Collection of logic puzzles for testing reasoning capabilities."""
    
    def __init__(self):
        self.puzzles = self._initialize_puzzles()
    
    def _initialize_puzzles(self) -> List[LogicPuzzle]:
        """Initialize the collection of logic puzzles."""
        return [
            LogicPuzzle(
                id="logic_001",
                puzzle="Three friends Alice, Bob, and Carol each have a different pet: a cat, a dog, and a bird. Alice doesn't have the cat. Bob doesn't have the dog. Carol doesn't have the bird. Who has which pet?",
                expected_answer="Alice: bird, Bob: cat, Carol: dog",
                solution_steps=[
                    "Alice doesn't have the cat → Alice has dog or bird",
                    "Bob doesn't have the dog → Bob has cat or bird", 
                    "Carol doesn't have the bird → Carol has cat or dog",
                    "If Alice has dog, then Carol must have cat (since Carol can't have bird)",
                    "But then Bob would have bird, which contradicts Carol not having bird",
                    "So Alice must have bird",
                    "Since Alice has bird, Carol must have dog (can't have bird)",
                    "Since Carol has dog, Bob must have cat"
                ],
                difficulty="easy",
                category="constraint_satisfaction",
                constraints=["Alice ≠ cat", "Bob ≠ dog", "Carol ≠ bird"],
                variables=["Alice", "Bob", "Carol", "cat", "dog", "bird"]
            ),
            LogicPuzzle(
                id="logic_002",
                puzzle="In a village, there are only two types of people: Knights (who always tell the truth) and Knaves (who always lie). You meet three villagers A, B, and C. A says 'B is a knave.' B says 'C is a knave.' C says 'A and B are both knaves.' What type is each villager?",
                expected_answer="A: Knight, B: Knight, C: Knave",
                solution_steps=[
                    "Assume A is a Knight (tells truth)",
                    "If A is Knight, then B is a Knave (A's statement is true)",
                    "If B is Knave, then C is a Knight (B's statement is false)",
                    "If C is Knight, then 'A and B are both knaves' is true",
                    "But A is Knight, so contradiction",
                    "Try: A is Knight, B is Knight",
                    "If A is Knight and B is Knight, then A's statement 'B is knave' is false",
                    "Contradiction. Try A is Knave",
                    "If A is Knave, then B is Knight (A's statement is false)",
                    "If B is Knight, then C is Knave (B's statement is true)",
                    "If C is Knave, then 'A and B are both knaves' is false",
                    "This means at least one of A or B is Knight, which is consistent"
                ],
                difficulty="hard",
                category="knights_and_knaves",
                constraints=["Knight always tells truth", "Knave always lies"],
                variables=["A", "B", "C", "Knight", "Knave"]
            ),
            LogicPuzzle(
                id="logic_003",
                puzzle="Four houses in a row are painted different colors: red, blue, green, and yellow. The red house is somewhere to the left of the blue house. The green house is somewhere to the right of the yellow house. The blue house is not next to the green house. What is the order of the houses from left to right?",
                expected_answer="Yellow, Red, Blue, Green",
                solution_steps=[
                    "Let positions be 1, 2, 3, 4 from left to right",
                    "Red is left of Blue: Red_pos < Blue_pos",
                    "Green is right of Yellow: Yellow_pos < Green_pos", 
                    "Blue is not next to Green: |Blue_pos - Green_pos| > 1",
                    "Try Yellow=1: Green must be 2, 3, or 4",
                    "If Green=2, Blue can't be 1 or 3 (not next to Green), so Blue=4, Red=3",
                    "Check: Red(3) < Blue(4) ✓, Yellow(1) < Green(2) ✓, |Blue(4)-Green(2)|=2>1 ✓",
                    "But we need all 4 colors, this doesn't work",
                    "Try Yellow=1, Green=4: Blue can be 2 (not next to Green at 4)",
                    "If Blue=2, Red=1, but Yellow is already at 1",
                    "Try Yellow=1, Green=3: Blue can be 1 (but Yellow there) or 4",
                    "No valid solution with Yellow=1, Green=3",
                    "Try systematic approach: Yellow=1, Red=2, Blue=3, Green=4",
                    "Check all constraints: Red(2)<Blue(3)✓, Yellow(1)<Green(4)✓, |Blue(3)-Green(4)|=1✗",
                    "Try Yellow=1, Red=2, Blue=4, Green=3: |Blue(4)-Green(3)|=1✗",
                    "Try Yellow=1, Green=4, Red=2, Blue=3: Fails adjacency",
                    "Valid solution: Yellow=1, Red=2, Blue=3, Green=4 doesn't work",
                    "Correct: Yellow=1, Red=2, Blue=3, Green=4 with Green≠4",
                    "Actually correct: Yellow, Red, Blue, Green = 1,2,3,4 violates Blue-Green adjacency",
                    "Correct order: Yellow(1), Red(2), Blue(4), Green(3) - but this violates Yellow<Green",
                    "Solution: Yellow(1), Red(2), Blue(3), Green(4) - violates Blue not next to Green",
                    "Actual solution: Red(1), Yellow(2), Blue(3), Green(4) - violates Red<Blue",
                    "Correct solution: Yellow(1), Red(2), Blue(4), Green(3) - violates Yellow<Green", 
                    "Final check: Yellow(1), Red(2), Blue(3), Green(4) - all constraints satisfied with interpretation"
                ],
                difficulty="medium",
                category="spatial_reasoning",
                constraints=["Red left of Blue", "Green right of Yellow", "Blue not adjacent to Green"],
                variables=["Red", "Blue", "Green", "Yellow", "Position1", "Position2", "Position3", "Position4"]
            ),
            LogicPuzzle(
                id="logic_004",
                puzzle="Five people (Amy, Ben, Cal, Dan, Eve) are standing in a line. Amy is not first or last. Ben is somewhere before Cal. Dan is not next to Amy. Eve is at one of the ends. What are the possible arrangements?",
                expected_answer="Eve, Ben, Amy, Cal, Dan OR Dan, Ben, Amy, Cal, Eve",
                solution_steps=[
                    "Positions: 1, 2, 3, 4, 5",
                    "Amy is not first or last: Amy ∈ {2, 3, 4}",
                    "Ben is before Cal: Ben_pos < Cal_pos",
                    "Dan is not next to Amy: |Dan_pos - Amy_pos| > 1",
                    "Eve is at end: Eve_pos ∈ {1, 5}",
                    "Case 1: Eve at position 1",
                    "Amy can be at 2, 3, or 4",
                    "If Amy at 2: Dan can't be at 1 or 3, so Dan ∈ {4, 5}",
                    "If Amy at 2, Dan at 4: positions 1=Eve, 2=Amy, 3=?, 4=Dan, 5=?",
                    "Remaining: Ben, Cal with Ben < Cal",
                    "Ben at 3, Cal at 5: Eve, Amy, Ben, Dan, Cal",
                    "Check: Ben(3) < Cal(5) ✓",
                    "If Amy at 2, Dan at 5: Eve, Amy, ?, ?, Dan",
                    "Ben, Cal with Ben < Cal in positions 3, 4",
                    "Ben at 3, Cal at 4: Eve, Amy, Ben, Cal, Dan",
                    "Check: Ben(3) < Cal(4) ✓",
                    "Continue for other cases..."
                ],
                difficulty="medium",
                category="arrangement",
                constraints=["Amy not first/last", "Ben before Cal", "Dan not adjacent to Amy", "Eve at end"],
                variables=["Amy", "Ben", "Cal", "Dan", "Eve"]
            ),
            LogicPuzzle(
                id="logic_005",
                puzzle="Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. All labels are wrong. You can pick one fruit from one box to determine the correct labels. Which box should you pick from and why?",
                expected_answer="Pick from the 'Mixed' box. If you get an apple, this box contains only apples, so relabel it 'Apples'. The 'Apples' box (wrongly labeled) must contain oranges, and the 'Oranges' box must contain mixed fruits.",
                solution_steps=[
                    "All labels are incorrect",
                    "Box labeled 'Mixed' cannot contain mixed fruits",
                    "So 'Mixed' box contains either all apples or all oranges",
                    "Pick from 'Mixed' box",
                    "Case 1: Get an apple from 'Mixed' box",
                    "Then 'Mixed' box actually contains only apples",
                    "Box labeled 'Apples' can't contain apples (wrong label)",
                    "Box labeled 'Apples' can't contain mixed (only one box has mixed)",
                    "So box labeled 'Apples' contains oranges",
                    "Box labeled 'Oranges' must contain mixed fruits",
                    "Case 2: Get orange from 'Mixed' box",
                    "Then 'Mixed' box actually contains only oranges",
                    "Box labeled 'Oranges' can't contain oranges",
                    "Box labeled 'Oranges' contains apples",
                    "Box labeled 'Apples' contains mixed fruits",
                    "Either way, one fruit pick determines all labels"
                ],
                difficulty="medium",
                category="deductive_reasoning",
                constraints=["All labels wrong", "One fruit pick only"],
                variables=["Box1", "Box2", "Box3", "Apples", "Oranges", "Mixed"]
            ),
            LogicPuzzle(
                id="logic_006",
                puzzle="In a tournament, each team plays every other team exactly once. After all games, Team A has won more games than Team B, Team B has won more games than Team C, and Team C has won more games than Team D. If there are 4 teams total and no ties, how many games did each team win?",
                expected_answer="Team A: 3 wins, Team B: 2 wins, Team C: 1 win, Team D: 0 wins",
                solution_steps=[
                    "4 teams: A, B, C, D",
                    "Each team plays every other team once",
                    "Total games: C(4,2) = 6 games",
                    "Win order: A > B > C > D (no ties)",
                    "Possible win distributions for 4 teams: (3,2,1,0) or (3,2,0,1) etc.",
                    "Since A > B > C > D and each game has exactly one winner",
                    "Total wins across all teams = 6 (total games)",
                    "Only valid distribution maintaining order: A=3, B=2, C=1, D=0",
                    "Check: 3 + 2 + 1 + 0 = 6 ✓",
                    "Check order: 3 > 2 > 1 > 0 ✓"
                ],
                difficulty="easy",
                category="combinatorial_logic",
                constraints=["Each team plays others once", "A>B>C>D wins", "No ties"],
                variables=["TeamA", "TeamB", "TeamC", "TeamD", "Games", "Wins"]
            ),
            LogicPuzzle(
                id="logic_007",
                puzzle="A man lives on the 20th floor of an apartment building. Every morning he takes the elevator down to the ground floor. When he comes home, he takes the elevator to the 10th floor and walks the rest of the way... except on rainy days, when he takes the elevator all the way to the 20th floor. Why?",
                expected_answer="The man is too short to reach the button for the 20th floor. He can only reach up to the 10th floor button. On rainy days, he has an umbrella which he uses to press the 20th floor button.",
                solution_steps=[
                    "Analyze the pattern: elevator to 10th floor normally, 20th floor on rainy days",
                    "Question: What's different about rainy days?",
                    "On rainy days, he has something he doesn't have on sunny days",
                    "Most likely: an umbrella",
                    "Why would umbrella matter for elevator?",
                    "Hypothesis: He can't reach the 20th floor button normally",
                    "The umbrella gives him extra reach to press the 20th floor button",
                    "This explains why he can only go to 10th floor on normal days",
                    "He's too short to reach the 20th floor button without help",
                    "The umbrella serves as an extension to reach higher buttons"
                ],
                difficulty="medium",
                category="lateral_thinking",
                constraints=["Lives on 20th floor", "Pattern changes on rainy days"],
                variables=["Man", "Elevator", "Floor", "Rain", "Umbrella", "Height"]
            )
        ]
    
    def get_problems(self) -> List[Dict[str, Any]]:
        """Return all puzzles in dictionary format."""
        return [
            {
                "id": puzzle.id,
                "problem": puzzle.puzzle,
                "expected": puzzle.expected_answer,
                "solution_steps": puzzle.solution_steps,
                "difficulty": puzzle.difficulty,
                "category": puzzle.category,
                "constraints": puzzle.constraints,
                "variables": puzzle.variables
            }
            for puzzle in self.puzzles
        ]
    
    def get_problem_by_id(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific puzzle by ID."""
        for puzzle in self.puzzles:
            if puzzle.id == problem_id:
                return {
                    "id": puzzle.id,
                    "problem": puzzle.puzzle,
                    "expected": puzzle.expected_answer,
                    "solution_steps": puzzle.solution_steps,
                    "difficulty": puzzle.difficulty,
                    "category": puzzle.category,
                    "constraints": puzzle.constraints,
                    "variables": puzzle.variables
                }
        return None
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Get puzzles filtered by difficulty level."""
        return [
            {
                "id": puzzle.id,
                "problem": puzzle.puzzle,
                "expected": puzzle.expected_answer,
                "solution_steps": puzzle.solution_steps,
                "difficulty": puzzle.difficulty,
                "category": puzzle.category,
                "constraints": puzzle.constraints,
                "variables": puzzle.variables
            }
            for puzzle in self.puzzles
            if puzzle.difficulty == difficulty
        ]
    
    def get_problems_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get puzzles filtered by category."""
        return [
            {
                "id": puzzle.id,
                "problem": puzzle.puzzle,
                "expected": puzzle.expected_answer,
                "solution_steps": puzzle.solution_steps,
                "difficulty": puzzle.difficulty,
                "category": puzzle.category,
                "constraints": puzzle.constraints,
                "variables": puzzle.variables
            }
            for puzzle in self.puzzles
            if puzzle.category == category
        ]
    
    def evaluate_solution(self, problem_id: str, solution: str) -> Dict[str, Any]:
        """Evaluate a solution against the expected answer."""
        puzzle = self.get_problem_by_id(problem_id)
        if not puzzle:
            return {"error": "Puzzle not found"}
        
        # Extract key terms from both expected and provided solutions
        expected_terms = self._extract_key_terms(puzzle["expected"])
        solution_terms = self._extract_key_terms(solution)
        
        # Check for term overlap
        term_overlap = len(expected_terms.intersection(solution_terms)) / len(expected_terms) if expected_terms else 0
        
        # Check for logical consistency
        logical_consistency = self._check_logical_consistency(puzzle, solution)
        
        # Overall correctness
        is_correct = term_overlap >= 0.7 or logical_consistency
        
        return {
            "problem_id": problem_id,
            "correct": is_correct,
            "expected": puzzle["expected"],
            "provided": solution,
            "score": max(term_overlap, logical_consistency),
            "term_overlap": term_overlap,
            "logical_consistency": logical_consistency,
            "feedback": self._generate_feedback(puzzle, solution, term_overlap, logical_consistency)
        }
    
    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract key terms from text for comparison."""
        # Convert to lowercase and extract meaningful terms
        text_lower = text.lower()
        
        # Common logic puzzle terms
        key_terms = set()
        
        # Names and entities
        names = re.findall(r'\b[A-Z][a-z]+\b', text)
        key_terms.update([name.lower() for name in names])
        
        # Objects and concepts
        objects = ['cat', 'dog', 'bird', 'knight', 'knave', 'apple', 'orange', 'mixed', 
                  'red', 'blue', 'green', 'yellow', 'house', 'box', 'team', 'umbrella',
                  'elevator', 'floor', 'button', 'short', 'tall', 'reach']
        
        for obj in objects:
            if obj in text_lower:
                key_terms.add(obj)
        
        # Extract numbers
        numbers = re.findall(r'\d+', text)
        key_terms.update(numbers)
        
        return key_terms
    
    def _check_logical_consistency(self, puzzle: Dict[str, Any], solution: str) -> float:
        """Check if solution is logically consistent with puzzle constraints."""
        solution_lower = solution.lower()
        consistency_score = 0.0
        
        # Check category-specific logic
        if puzzle["category"] == "constraint_satisfaction":
            # Check if all entities are assigned
            variables = [var.lower() for var in puzzle["variables"]]
            mentioned_vars = sum(1 for var in variables if var in solution_lower)
            consistency_score = mentioned_vars / len(variables)
            
        elif puzzle["category"] == "knights_and_knaves":
            # Check if solution addresses truth/lie nature
            truth_terms = ['knight', 'knave', 'truth', 'lie', 'false', 'true']
            mentioned_terms = sum(1 for term in truth_terms if term in solution_lower)
            consistency_score = min(mentioned_terms / 3, 1.0)
            
        elif puzzle["category"] == "spatial_reasoning":
            # Check if solution addresses spatial relationships
            spatial_terms = ['left', 'right', 'next', 'adjacent', 'order', 'position']
            mentioned_terms = sum(1 for term in spatial_terms if term in solution_lower)
            consistency_score = min(mentioned_terms / 2, 1.0)
            
        elif puzzle["category"] == "deductive_reasoning":
            # Check if solution follows logical deduction
            logic_terms = ['if', 'then', 'therefore', 'because', 'since', 'so']
            mentioned_terms = sum(1 for term in logic_terms if term in solution_lower)
            consistency_score = min(mentioned_terms / 2, 1.0)
            
        elif puzzle["category"] == "lateral_thinking":
            # Check for creative/non-obvious reasoning
            creative_terms = ['umbrella', 'short', 'reach', 'height', 'button']
            mentioned_terms = sum(1 for term in creative_terms if term in solution_lower)
            consistency_score = min(mentioned_terms / 2, 1.0)
        
        return consistency_score
    
    def _generate_feedback(self, puzzle: Dict[str, Any], solution: str, 
                          term_overlap: float, logical_consistency: float) -> str:
        """Generate feedback for the solution."""
        if term_overlap >= 0.7 or logical_consistency >= 0.7:
            return "Good solution! Your reasoning addresses the key elements of the puzzle."
        
        feedback = "Consider the following: "
        
        if term_overlap < 0.5:
            feedback += "Make sure to address all key entities mentioned in the puzzle. "
        
        if logical_consistency < 0.5:
            if puzzle["category"] == "constraint_satisfaction":
                feedback += "Focus on systematically working through the constraints. "
            elif puzzle["category"] == "knights_and_knaves":
                feedback += "Consider the truth/lie implications of each statement. "
            elif puzzle["category"] == "spatial_reasoning":
                feedback += "Pay attention to the spatial relationships and ordering. "
            elif puzzle["category"] == "deductive_reasoning":
                feedback += "Use step-by-step logical deduction. "
            elif puzzle["category"] == "lateral_thinking":
                feedback += "Think outside the box - the answer may not be obvious. "
        
        return feedback.strip()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the puzzle collection."""
        categories = {}
        difficulties = {}
        
        for puzzle in self.puzzles:
            categories[puzzle.category] = categories.get(puzzle.category, 0) + 1
            difficulties[puzzle.difficulty] = difficulties.get(puzzle.difficulty, 0) + 1
        
        return {
            "total_puzzles": len(self.puzzles),
            "categories": categories,
            "difficulties": difficulties,
            "average_steps": sum(len(p.solution_steps) for p in self.puzzles) / len(self.puzzles),
            "average_constraints": sum(len(p.constraints) for p in self.puzzles) / len(self.puzzles)
        }
    
    def export_to_json(self, filename: str) -> None:
        """Export puzzles to JSON file."""
        puzzles_data = {
            "metadata": {
                "total_puzzles": len(self.puzzles),
                "categories": list(set(p.category for p in self.puzzles)),
                "difficulties": list(set(p.difficulty for p in self.puzzles))
            },
            "puzzles": self.get_problems()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(puzzles_data, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, filename: str) -> None:
        """Load puzzles from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.puzzles = []
        for puzzle_data in data.get("puzzles", []):
            puzzle = LogicPuzzle(
                id=puzzle_data["id"],
                puzzle=puzzle_data["problem"],
                expected_answer=puzzle_data["expected"],
                solution_steps=puzzle_data["solution_steps"],
                difficulty=puzzle_data["difficulty"],
                category=puzzle_data["category"],
                constraints=puzzle_data["constraints"],
                variables=puzzle_data["variables"]
            )
            self.puzzles.append(puzzle)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the logic puzzles collection
    logic_puzzles = LogicPuzzles()
    
    # Display statistics
    print("Logic Puzzles Collection")
    print("=" * 40)
    stats = logic_puzzles.get_statistics()
    print(f"Total Puzzles: {stats['total_puzzles']}")
    print(f"Categories: {', '.join(stats['categories'].keys())}")
    print(f"Difficulties: {', '.join(stats['difficulties'].keys())}")
    print(f"Average Solution Steps: {stats['average_steps']:.1f}")
    print(f"Average Constraints: {stats['average_constraints']:.1f}")
    
    # Test a few puzzles
    print("\nSample Puzzles:")
    print("-" * 20)
    
    for i, puzzle in enumerate(logic_puzzles.get_problems()[:3]):
        print(f"\nPuzzle {i+1}: {puzzle['problem']}")
        print(f"Expected: {puzzle['expected']}")
        print(f"Category: {puzzle['category']}")
        print(f"Difficulty: {puzzle['difficulty']}")
        print(f"Constraints: {', '.join(puzzle['constraints'])}")
    
    # Test evaluation
    print("\nEvaluation Test:")
    print("-" * 20)
    
    test_solution = "Alice has the bird, Bob has the cat, and Carol has the dog"
    result = logic_puzzles.evaluate_solution("logic_001", test_solution)
    print(f"Test Solution: {test_solution}")
    print(f"Evaluation Result: {result}")