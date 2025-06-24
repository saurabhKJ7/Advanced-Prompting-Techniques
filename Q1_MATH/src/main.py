import os
import json
import openai
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MathTutorAgent:
    def __init__(self, api_key: str = None):
        """Initialize the Math Tutor Agent with OpenAI API"""
        # Load environment variables from .env file
        load_dotenv(Path(__file__).parent.parent / '.env')
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or environment variable.")
        
        openai.api_key = self.api_key
        self.project_root = Path(__file__).parent.parent
        self.prompts_dir = self.project_root / "prompts"
        self.eval_dir = self.project_root / "evaluation"
        
        # Load configuration from environment
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.3'))
        
        # Load prompt templates
        self.load_prompt_templates()
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Load test queries for evaluation
        self.load_test_queries()

    def load_prompt_templates(self):
        """Load all prompt templates from files"""
        self.prompt_templates = {}
        template_files = {
            "zero_shot": "zero_shot.txt",
            "few_shot": "few_shot.txt", 
            "cot": "cot_prompt.txt",
            "meta": "meta_prompt.txt"
        }
        
        for prompt_type, filename in template_files.items():
            try:
                filepath = self.prompts_dir / filename
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.prompt_templates[prompt_type] = f.read().strip()
                    logger.info(f"Loaded {prompt_type} prompt template")
                else:
                    logger.warning(f"Prompt file not found: {filepath}")
            except Exception as e:
                logger.error(f"Error loading {prompt_type} template: {e}")

    def load_test_queries(self):
        """Load test queries for evaluation"""
        try:
            queries_file = self.eval_dir / "input_queries.json"
            if queries_file.exists():
                with open(queries_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.test_queries = data.get("test_queries", [])
                    self.eval_criteria = data.get("evaluation_criteria", {})
                logger.info(f"Loaded {len(self.test_queries)} test queries")
            else:
                logger.warning("Test queries file not found")
                self.test_queries = []
                self.eval_criteria = {}
        except Exception as e:
            logger.error(f"Error loading test queries: {e}")
            self.test_queries = []
            self.eval_criteria = {}

    def get_response(self, user_question: str, prompt_type: str = "zero_shot") -> str:
        """Get response from OpenAI using specified prompt type"""
        if prompt_type not in self.prompt_templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(self.prompt_templates.keys())}")
        
        # Get the system prompt
        system_prompt = self.prompt_templates[prompt_type]
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Log the interaction
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "prompt_type": prompt_type,
                "question": user_question,
                "response": answer,
                "tokens_used": response.usage.total_tokens
            })
            
            return answer
            
        except Exception as e:
            logger.error(f"Error getting OpenAI response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def interactive_session(self):
        """Run interactive session with the math tutor"""
        print("🧮 Math Tutor Agent - Interactive Session")
        print("=" * 50)
        print("Available prompt types:")
        for i, prompt_type in enumerate(self.prompt_templates.keys(), 1):
            print(f"{i}. {prompt_type}")
        print("Commands: 'quit' to exit, 'switch' to change prompt type, 'evaluate' to run tests")
        print("=" * 50)
        
        current_prompt_type = "zero_shot"
        print(f"Current prompt type: {current_prompt_type}")
        
        while True:
            try:
                user_input = input("\n📝 Ask your math question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Happy learning!")
                    break
                
                elif user_input.lower() == 'switch':
                    print("\nAvailable prompt types:")
                    prompt_types = list(self.prompt_templates.keys())
                    for i, pt in enumerate(prompt_types, 1):
                        print(f"{i}. {pt}")
                    
                    try:
                        choice = int(input("Select prompt type (number): ")) - 1
                        if 0 <= choice < len(prompt_types):
                            current_prompt_type = prompt_types[choice]
                            print(f"✅ Switched to: {current_prompt_type}")
                        else:
                            print("❌ Invalid choice")
                    except ValueError:
                        print("❌ Please enter a valid number")
                    continue
                
                elif user_input.lower() == 'evaluate':
                    self.run_evaluation()
                    continue
                
                elif user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue
                
                elif not user_input:
                    continue
                
                # Get response using current prompt type
                print(f"\n🤖 Using {current_prompt_type} prompt...")
                response = self.get_response(user_input, current_prompt_type)
                print(f"\n📚 Math Tutor Response:\n{response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Happy learning!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def run_evaluation(self):
        """Run evaluation on test queries"""
        if not self.test_queries:
            print("❌ No test queries available for evaluation")
            return
        
        print("\n🔍 Running Evaluation...")
        print("=" * 50)
        
        results = {}
        
        for prompt_type in self.prompt_templates.keys():
            print(f"\n📊 Testing {prompt_type} prompts...")
            prompt_results = []
            
            for i, query in enumerate(self.test_queries[:3], 1):  # Test first 3 queries
                print(f"  Question {i}: {query['query'][:60]}...")
                
                try:
                    response = self.get_response(query['query'], prompt_type)
                    prompt_results.append({
                        'query': query,
                        'response': response,
                        'success': True
                    })
                    print(f"    ✅ Response generated")
                except Exception as e:
                    prompt_results.append({
                        'query': query,
                        'response': f"Error: {e}",
                        'success': False
                    })
                    print(f"    ❌ Error: {e}")
            
            results[prompt_type] = prompt_results
        
        # Display results summary
        print("\n📊 Evaluation Summary:")
        print("=" * 50)
        for prompt_type, results_list in results.items():
            successful = sum(1 for r in results_list if r['success'])
            total = len(results_list)
            print(f"{prompt_type}: {successful}/{total} successful responses")
        
        # Save detailed results
        self.save_evaluation_results(results)

    def save_evaluation_results(self, results: Dict):
        """Save evaluation results to file"""
        try:
            output_file = self.eval_dir / "output_logs.json"
            
            # Create output data
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "evaluation_results": results,
                "conversation_history": self.conversation_history
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Results saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def show_conversation_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history available")
            return
        
        print("\n📚 Conversation History:")
        print("=" * 50)
        
        for i, entry in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            print(f"\n{i}. [{entry['prompt_type']}] {entry['timestamp'][:19]}")
            print(f"Q: {entry['question'][:100]}...")
            print(f"A: {entry['response'][:200]}...")
            print(f"Tokens: {entry['tokens_used']}")

    def compare_prompt_strategies(self, question: str):
        """Compare all prompt strategies for a single question"""
        print(f"\n🔍 Comparing prompt strategies for: {question}")
        print("=" * 70)
        
        for prompt_type in self.prompt_templates.keys():
            print(f"\n📝 {prompt_type.upper()} PROMPT:")
            print("-" * 30)
            try:
                response = self.get_response(question, prompt_type)
                print(response)
            except Exception as e:
                print(f"❌ Error: {e}")
            print("-" * 30)

def main():
    """Main function to run the Math Tutor Agent"""
    try:
        # Initialize the agent
        agent = MathTutorAgent()
        
        print("🚀 Math Tutor Agent initialized successfully!")
        print(f"📁 Project root: {agent.project_root}")
        print(f"🤖 Using model: {agent.model}")
        print(f"📋 Loaded {len(agent.prompt_templates)} prompt templates")
        print(f"🧪 Loaded {len(agent.test_queries)} test queries")
        
        # Check if we have a test question to demonstrate
        if agent.test_queries:
            demo_question = agent.test_queries[0]['query']
            print(f"\n🎯 Quick Demo with: {demo_question}")
            
            # Demo with zero-shot
            response = agent.get_response(demo_question, "zero_shot")
            print(f"\n📚 Zero-shot response:\n{response}")
        
        # Start interactive session
        agent.interactive_session()
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Please set your OpenAI API key in the .env file:")
        print("   1. Open the .env file in the project root")
        print("   2. Replace 'your-openai-api-key-here' with your actual API key")
        print("   3. Your key should start with 'sk-'")
        print("   4. Get your key from: https://platform.openai.com/api-keys")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()