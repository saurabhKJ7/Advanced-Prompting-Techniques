"""
Common Sense Q&A Task Definition

This module defines a collection of common sense reasoning problems that require
everyday knowledge, social understanding, and physical world reasoning.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import json
import re


@dataclass
class CommonSenseProblem:
    """Represents a single common sense reasoning problem."""
    id: str
    question: str
    expected_answer: str
    solution_steps: List[str]
    difficulty: str
    category: str
    knowledge_domains: List[str]
    reasoning_type: str


class CommonSenseQA:
    """Collection of common sense Q&A problems for testing everyday reasoning."""
    
    def __init__(self):
        self.problems = self._initialize_problems()
    
    def _initialize_problems(self) -> List[CommonSenseProblem]:
        """Initialize the collection of common sense problems."""
        return [
            CommonSenseProblem(
                id="cs_001",
                question="Why do people typically carry umbrellas on rainy days?",
                expected_answer="To protect themselves from getting wet in the rain.",
                solution_steps=[
                    "Rain consists of water droplets falling from the sky",
                    "Water makes people wet and uncomfortable",
                    "Umbrellas are waterproof and block rain from above",
                    "Carrying an umbrella keeps the person dry",
                    "This is why people use umbrellas on rainy days"
                ],
                difficulty="easy",
                category="physical_world",
                knowledge_domains=["weather", "objects", "protection"],
                reasoning_type="causal_reasoning"
            ),
            CommonSenseProblem(
                id="cs_002",
                question="If someone is wearing a winter coat in July, what might you reasonably assume?",
                expected_answer="They might be in the Southern Hemisphere where it's winter, in a very cold location, or have a medical condition that makes them feel cold.",
                solution_steps=[
                    "July is summer in the Northern Hemisphere",
                    "Winter coats are typically worn in cold weather",
                    "Wearing a winter coat in July suggests unusual circumstances",
                    "Possible explanations: different hemisphere, cold location, medical reasons",
                    "Southern Hemisphere has winter during July",
                    "High altitude or air-conditioned environments can be cold",
                    "Some medical conditions affect temperature regulation"
                ],
                difficulty="medium",
                category="contextual_reasoning",
                knowledge_domains=["seasons", "geography", "clothing", "health"],
                reasoning_type="abductive_reasoning"
            ),
            CommonSenseProblem(
                id="cs_003",
                question="Why don't people usually eat with their hands at fancy restaurants?",
                expected_answer="Because using utensils is considered proper etiquette and more hygienic in formal dining settings.",
                solution_steps=[
                    "Fancy restaurants have formal dining standards",
                    "Social etiquette dictates proper behavior in formal settings",
                    "Using utensils is considered more refined and polite",
                    "Eating with hands can be seen as unhygienic or crude",
                    "Formal dining emphasizes presentation and manners",
                    "Cultural norms expect utensil use in upscale establishments"
                ],
                difficulty="medium",
                category="social_norms",
                knowledge_domains=["etiquette", "culture", "hygiene", "social_behavior"],
                reasoning_type="social_reasoning"
            ),
            CommonSenseProblem(
                id="cs_004",
                question="What would happen if you tried to use a fork to eat soup?",
                expected_answer="It would be very difficult and ineffective because soup is liquid and would flow through the fork's prongs.",
                solution_steps=[
                    "Forks have prongs with gaps between them",
                    "Soup is a liquid food",
                    "Liquids flow through openings due to gravity",
                    "The soup would drip through the fork's prongs",
                    "Very little soup would reach your mouth",
                    "This makes forks impractical tools for eating soup"
                ],
                difficulty="easy",
                category="physical_world",
                knowledge_domains=["physics", "utensils", "food_properties"],
                reasoning_type="physical_reasoning"
            ),
            CommonSenseProblem(
                id="cs_005",
                question="Why do most people close their eyes when they sneeze?",
                expected_answer="It's an involuntary reflex. The body automatically closes the eyes during sneezing, possibly to protect them from particles or due to nerve connections.",
                solution_steps=[
                    "Sneezing is a reflex action controlled by the nervous system",
                    "The body has automatic responses to protect itself",
                    "Eye closing during sneezing is involuntary",
                    "May protect eyes from expelled particles",
                    "Nerve pathways might be connected between sneeze and blink reflexes",
                    "People cannot easily control this automatic response"
                ],
                difficulty="medium",
                category="biological_processes",
                knowledge_domains=["biology", "reflexes", "human_body"],
                reasoning_type="scientific_reasoning"
            ),
            CommonSenseProblem(
                id="cs_006",
                question="If you see someone running quickly down the street with a worried expression, what might be happening?",
                expected_answer="They might be late for something important, rushing to help someone, or fleeing from danger.",
                solution_steps=[
                    "Running quickly suggests urgency",
                    "Worried expression indicates concern or stress",
                    "Combination suggests a pressing situation",
                    "Common scenarios: being late, emergency, danger",
                    "Could be rushing to appointment, meeting, or transportation",
                    "Might be responding to emergency call or situation",
                    "Could be avoiding or escaping from something threatening"
                ],
                difficulty="easy",
                category="social_situations",
                knowledge_domains=["emotions", "body_language", "social_behavior"],
                reasoning_type="social_reasoning"
            ),
            CommonSenseProblem(
                id="cs_007",
                question="Why do people usually turn on lights when entering a dark room?",
                expected_answer="To see clearly and navigate safely, as humans need light to see in dark environments.",
                solution_steps=[
                    "Human eyes need light to see objects",
                    "Dark rooms lack sufficient light for vision",
                    "Poor visibility makes movement dangerous",
                    "People might trip, bump into things, or get injured",
                    "Electric lights provide artificial illumination",
                    "Turning on lights makes the room safe and functional"
                ],
                difficulty="easy",
                category="physical_world",
                knowledge_domains=["vision", "light", "safety", "human_physiology"],
                reasoning_type="practical_reasoning"
            ),
            CommonSenseProblem(
                id="cs_008",
                question="What's unusual about someone brushing their teeth with orange juice?",
                expected_answer="Orange juice is acidic and sugary, which is harmful to teeth. People typically brush teeth with toothpaste and water to clean them, not with substances that can cause damage.",
                solution_steps=[
                    "Toothbrushing is meant to clean and protect teeth",
                    "Toothpaste contains cleaning agents and fluoride",
                    "Orange juice is acidic and contains sugar",
                    "Acid can erode tooth enamel",
                    "Sugar feeds bacteria that cause tooth decay",
                    "Using orange juice defeats the purpose of dental hygiene",
                    "This would likely harm rather than help teeth"
                ],
                difficulty="medium",
                category="health_hygiene",
                knowledge_domains=["dental_health", "chemistry", "hygiene"],
                reasoning_type="health_reasoning"
            ),
            CommonSenseProblem(
                id="cs_009",
                question="Why do people typically say 'bless you' when someone sneezes?",
                expected_answer="It's a social custom rooted in historical beliefs about health and politeness, now continued as a courteous gesture.",
                solution_steps=[
                    "Saying 'bless you' is a cultural tradition",
                    "Historically, people believed sneezing was spiritually significant",
                    "Some thought the soul could escape during sneezing",
                    "Others believed it was a sign of illness",
                    "The blessing was meant to protect the person",
                    "Modern usage is simply polite social behavior",
                    "It shows care and acknowledgment of the person"
                ],
                difficulty="medium",
                category="social_customs",
                knowledge_domains=["culture", "history", "social_behavior", "traditions"],
                reasoning_type="cultural_reasoning"
            ),
            CommonSenseProblem(
                id="cs_010",
                question="What would likely happen if you planted a seed in a cup of water instead of soil?",
                expected_answer="The seed might initially germinate but would likely die because it lacks essential nutrients and proper support that soil provides.",
                solution_steps=[
                    "Seeds need water, oxygen, and nutrients to grow",
                    "Water alone provides moisture but lacks nutrients",
                    "Soil contains minerals and organic matter plants need",
                    "Soil also provides physical support for roots",
                    "Initial germination might occur with just water",
                    "Long-term growth requires nutrient-rich medium",
                    "The plant would eventually weaken and die without proper nutrition"
                ],
                difficulty="medium",
                category="biological_processes",
                knowledge_domains=["botany", "plant_biology", "nutrition", "growth"],
                reasoning_type="biological_reasoning"
            )
        ]
    
    def get_problems(self) -> List[Dict[str, Any]]:
        """Return all problems in dictionary format."""
        return [
            {
                "id": problem.id,
                "question": problem.question,
                "expected": problem.expected_answer,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "category": problem.category,
                "knowledge_domains": problem.knowledge_domains,
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
                    "question": problem.question,
                    "expected": problem.expected_answer,
                    "solution_steps": problem.solution_steps,
                    "difficulty": problem.difficulty,
                    "category": problem.category,
                    "knowledge_domains": problem.knowledge_domains,
                    "reasoning_type": problem.reasoning_type
                }
        return None
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Get problems filtered by difficulty level."""
        return [
            {
                "id": problem.id,
                "question": problem.question,
                "expected": problem.expected_answer,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "category": problem.category,
                "knowledge_domains": problem.knowledge_domains,
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
                "question": problem.question,
                "expected": problem.expected_answer,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "category": problem.category,
                "knowledge_domains": problem.knowledge_domains,
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
                "question": problem.question,
                "expected": problem.expected_answer,
                "solution_steps": problem.solution_steps,
                "difficulty": problem.difficulty,
                "category": problem.category,
                "knowledge_domains": problem.knowledge_domains,
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
        
        # Check conceptual understanding
        conceptual_understanding = self._check_conceptual_understanding(problem, solution)
        
        # Check knowledge domain coverage
        domain_coverage = self._check_domain_coverage(problem, solution)
        
        # Check reasoning quality
        reasoning_quality = self._check_reasoning_quality(problem, solution)
        
        # Check practical applicability
        practical_applicability = self._check_practical_applicability(problem, solution)
        
        # Overall score
        overall_score = (conceptual_understanding + domain_coverage + reasoning_quality + practical_applicability) / 4
        
        return {
            "problem_id": problem_id,
            "correct": overall_score >= 0.7,
            "expected": problem["expected"],
            "provided": solution,
            "scores": {
                "conceptual_understanding": conceptual_understanding,
                "domain_coverage": domain_coverage,
                "reasoning_quality": reasoning_quality,
                "practical_applicability": practical_applicability,
                "overall": overall_score
            },
            "feedback": self._generate_feedback(problem, solution, overall_score)
        }
    
    def _check_conceptual_understanding(self, problem: Dict[str, Any], solution: str) -> float:
        """Check if solution demonstrates understanding of core concepts."""
        solution_lower = solution.lower()
        
        # Create concept keywords based on the problem category
        concept_keywords = {
            "physical_world": ["physical", "properties", "gravity", "liquid", "solid"],
            "social_norms": ["social", "etiquette", "polite", "custom", "behavior"],
            "biological_processes": ["body", "reflex", "automatic", "natural", "biological"],
            "health_hygiene": ["health", "clean", "hygiene", "protect", "safe"],
            "social_customs": ["tradition", "custom", "culture", "historical", "belief"],
            "contextual_reasoning": ["context", "situation", "circumstances", "environment"],
            "social_situations": ["emergency", "urgent", "worried", "expression", "emotion"]
        }
        
        category = problem["category"]
        relevant_keywords = concept_keywords.get(category, [])
        
        if not relevant_keywords:
            return 0.5  # Default score if no specific keywords
        
        mentioned_keywords = sum(1 for keyword in relevant_keywords if keyword in solution_lower)
        return min(mentioned_keywords / len(relevant_keywords), 1.0)
    
    def _check_domain_coverage(self, problem: Dict[str, Any], solution: str) -> float:
        """Check if solution covers relevant knowledge domains."""
        solution_lower = solution.lower()
        domains = problem["knowledge_domains"]
        
        domain_indicators = {
            "weather": ["rain", "water", "wet", "weather", "precipitation"],
            "objects": ["umbrella", "tool", "utensil", "item", "object"],
            "protection": ["protect", "shield", "block", "guard", "safe"],
            "seasons": ["summer", "winter", "seasonal", "july", "month"],
            "geography": ["hemisphere", "location", "north", "south", "altitude"],
            "clothing": ["coat", "jacket", "wear", "clothing", "dress"],
            "health": ["medical", "condition", "temperature", "body", "health"],
            "etiquette": ["manners", "proper", "polite", "formal", "etiquette"],
            "culture": ["cultural", "society", "tradition", "custom", "norm"],
            "hygiene": ["clean", "hygiene", "sanitary", "dirty", "wash"],
            "physics": ["gravity", "flow", "liquid", "force", "physical"],
            "utensils": ["fork", "spoon", "knife", "utensil", "tool"],
            "food_properties": ["liquid", "solid", "texture", "consistency", "food"],
            "biology": ["biological", "body", "organ", "system", "natural"],
            "reflexes": ["reflex", "automatic", "involuntary", "response", "reaction"],
            "human_body": ["body", "eyes", "nerve", "muscle", "physical"],
            "emotions": ["worried", "concerned", "anxious", "expression", "feeling"],
            "body_language": ["expression", "face", "gesture", "posture", "signal"],
            "social_behavior": ["behavior", "social", "interaction", "response", "action"],
            "vision": ["see", "sight", "visual", "eyes", "look"],
            "light": ["light", "bright", "dark", "illuminate", "visible"],
            "safety": ["safe", "danger", "risk", "hazard", "protect"],
            "human_physiology": ["human", "body", "physiology", "biological", "natural"],
            "dental_health": ["teeth", "dental", "enamel", "cavity", "oral"],
            "chemistry": ["acid", "chemical", "reaction", "substance", "compound"],
            "history": ["historical", "past", "tradition", "ancient", "origin"],
            "traditions": ["tradition", "custom", "practice", "ritual", "habit"],
            "botany": ["plant", "seed", "grow", "botanical", "vegetation"],
            "plant_biology": ["plant", "root", "stem", "leaf", "photosynthesis"],
            "nutrition": ["nutrient", "food", "mineral", "vitamin", "nourishment"],
            "growth": ["grow", "development", "mature", "increase", "expand"]
        }
        
        total_coverage = 0
        for domain in domains:
            indicators = domain_indicators.get(domain, [])
            if indicators:
                mentioned = sum(1 for indicator in indicators if indicator in solution_lower)
                coverage = min(mentioned / len(indicators), 1.0)
                total_coverage += coverage
        
        return total_coverage / len(domains) if domains else 0.0
    
    def _check_reasoning_quality(self, problem: Dict[str, Any], solution: str) -> float:
        """Check the quality of reasoning demonstrated in the solution."""
        solution_lower = solution.lower()
        reasoning_type = problem["reasoning_type"]
        
        reasoning_indicators = {
            "causal_reasoning": ["because", "cause", "result", "leads to", "due to"],
            "abductive_reasoning": ["might", "could", "possible", "likely", "suggests"],
            "social_reasoning": ["social", "people", "society", "behavior", "norm"],
            "physical_reasoning": ["physical", "gravity", "flow", "properties", "force"],
            "scientific_reasoning": ["body", "system", "process", "biological", "mechanism"],
            "practical_reasoning": ["practical", "useful", "function", "purpose", "need"],
            "health_reasoning": ["health", "harmful", "beneficial", "safe", "dangerous"],
            "cultural_reasoning": ["culture", "tradition", "custom", "belief", "practice"],
            "biological_reasoning": ["biological", "natural", "organism", "life", "living"]
        }
        
        relevant_indicators = reasoning_indicators.get(reasoning_type, [])
        if not relevant_indicators:
            return 0.5
        
        mentioned_indicators = sum(1 for indicator in relevant_indicators if indicator in solution_lower)
        return min(mentioned_indicators / len(relevant_indicators), 1.0)
    
    def _check_practical_applicability(self, problem: Dict[str, Any], solution: str) -> float:
        """Check if solution demonstrates practical understanding."""
        solution_lower = solution.lower()
        
        # Look for practical application indicators
        practical_indicators = [
            "practical", "useful", "everyday", "common", "normal", "typical",
            "usually", "generally", "often", "regularly", "real", "actual"
        ]
        
        mentioned_practical = sum(1 for indicator in practical_indicators if indicator in solution_lower)
        
        # Also check for specific examples or concrete explanations
        example_indicators = [
            "example", "instance", "case", "situation", "scenario", "like", "such as"
        ]
        
        mentioned_examples = sum(1 for indicator in example_indicators if indicator in solution_lower)
        
        # Combine both scores
        practical_score = min(mentioned_practical / 3, 1.0)  # Normalize to max 1.0
        example_score = min(mentioned_examples / 2, 1.0)    # Normalize to max 1.0
        
        return (practical_score + example_score) / 2
    
    def _generate_feedback(self, problem: Dict[str, Any], solution: str, score: float) -> str:
        """Generate feedback for the solution."""
        if score >= 0.8:
            return "Excellent! You demonstrated strong common sense reasoning with good understanding of the underlying concepts."
        elif score >= 0.6:
            return "Good reasoning! You showed understanding of the main concepts but could elaborate on some aspects."
        elif score >= 0.4:
            return "Partial understanding. Try to consider more aspects of the situation and explain your reasoning more thoroughly."
        else:
            category = problem["category"]
            if category == "physical_world":
                return "Consider the physical properties and laws that govern the situation."
            elif category == "social_norms":
                return "Think about social expectations and cultural standards in different contexts."
            elif category == "biological_processes":
                return "Consider how the human body naturally functions and responds to stimuli."
            elif category == "health_hygiene":
                return "Think about what promotes health and what might be harmful to the body."
            elif category == "social_customs":
                return "Consider the historical and cultural origins of social practices."
            elif category == "contextual_reasoning":
                return "Think about the broader context and what circumstances might explain the situation."
            elif category == "social_situations":
                return "Consider human emotions, motivations, and typical responses to different situations."
            else:
                return "Try to think about the everyday knowledge and experience that applies to this situation."
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the problem collection."""
        categories = {}
        difficulties = {}
        reasoning_types = {}
        
        for problem in self.problems:
            categories[problem.category] = categories.get(problem.category, 0) + 1
            difficulties[problem.difficulty] = difficulties.get(problem.difficulty, 0) + 1
            reasoning_types[problem.reasoning_type] = reasoning_types.get(problem.reasoning_type, 0) + 1
        
        # Calculate knowledge domain distribution
        all_domains = []
        for problem in self.problems:
            all_domains.extend(problem.knowledge_domains)
        
        domain_counts = {}
        for domain in all_domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            "total_problems": len(self.problems),
            "categories": categories,
            "difficulties": difficulties,
            "reasoning_types": reasoning_types,
            "knowledge_domains": domain_counts,
            "average_steps": sum(len(p.solution_steps) for p in self.problems) / len(self.problems),
            "average_domains": sum(len(p.knowledge_domains) for p in self.problems) / len(self.problems)
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
            problem = CommonSenseProblem(
                id=prob_data["id"],
                question=prob_data["question"],
                expected_answer=prob_data["expected"],
                solution_steps=prob_data["solution_steps"],
                difficulty=prob_data["difficulty"],
                category=prob_data["category"],
                knowledge_domains=prob_data["knowledge_domains"],
                reasoning_type=prob_data["reasoning_type"]
            )
            self.problems.append(problem)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the common sense QA collection
    cs_qa = CommonSenseQA()
    
    # Display statistics
    print("Common Sense Q&A Problems Collection")
    print("=" * 40)
    stats = cs_qa.get_statistics()
    print(f"Total Problems: {stats['total_problems']}")
    print(f"Categories: {', '.join(stats['categories'].keys())}")
    print(f"Difficulties: {', '.join(stats['difficulties'].keys())}")
    print(f"Reasoning Types: {', '.join(stats['reasoning_types'].keys())}")
    print(f"Knowledge Domains: {len(stats['knowledge_domains'])} unique domains")
    print(f"Average Solution Steps: {stats['average_steps']:.1f}")
    print(f"Average Domains per Problem: {stats['average_domains']:.1f}")
    
    # Test a few problems
    print("\nSample Problems:")
    print("-" * 20)
    
    for i, problem in enumerate(cs_qa.get_problems()[:3]):
        print(f"\nProblem {i+1}: {problem['question']}")
        print(f"Expected: {problem['expected']}")
        print(f"Category: {problem['category']}")
        print(f"Difficulty: {problem['difficulty']}")
        print(f"Reasoning Type: {problem['reasoning_type']}")
        print(f"Knowledge Domains: {', '.join(problem['knowledge_domains'])}")
    
    # Test evaluation
    print("\nEvaluation Test:")
    print("-" * 20)
    
    test_solution = "People carry umbrellas to stay dry when it rains because umbrellas are waterproof and block the rain from above."
    result = cs_qa.evaluate_solution("cs_001", test_solution)
    print(f"Test Solution: {test_solution}")
    print(f"Evaluation Result: {result}")