#!/usr/bin/env python3
"""
Math Tutor Agent - Setup and Run Script
This script helps set up and run the EdTech Math Tutor with different prompting strategies.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_openai_key():
    """Check if OpenAI API key is set"""
    # Check for .env file first
    env_file = Path(__file__).parent / '.env'
    
    if env_file.exists():
        print("✅ .env file found")
        
        # Load and check the .env file
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            print("⚠️  python-dotenv not installed, trying to read .env manually")
            
        with open(env_file, 'r') as f:
            env_content = f.read()
            
        if 'OPENAI_API_KEY=your-openai-api-key-here' in env_content:
            print("❌ Please update your OpenAI API key in the .env file")
            print("  1. Open the .env file")
            print("  2. Replace 'your-openai-api-key-here' with your actual API key")
            print("  3. Your key should start with 'sk-'")
            print("  4. Get your key from: https://platform.openai.com/api-keys")
            return False
        elif 'OPENAI_API_KEY=' in env_content:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key.startswith('sk-'):
                print("✅ Valid OpenAI API key found in .env file")
                return True
            else:
                print("❌ Invalid OpenAI API key format in .env file")
                print("  Your key should start with 'sk-'")
                return False
    else:
        print("⚠️  .env file not found")
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print("✅ OpenAI API key found in environment variables")
            return True
        else:
            print("❌ OpenAI API key not found")
            print("Please set your OpenAI API key:")
            print("  1. Use the .env file (recommended):")
            print("     - A template .env file should be in the project root")
            print("     - Edit it and add your API key")
            print("  2. Or set environment variable:")
            print("     export OPENAI_API_KEY='your-key-here'")
            return False
    
    return False

def setup_project():
    """Set up the project environment"""
    print("🚀 Setting up Math Tutor Agent...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create .env file if it doesn't exist
    env_file = Path(__file__).parent / '.env'
    if not env_file.exists():
        print("📝 Creating .env file template...")
        env_template = """# OpenAI API Configuration
# Replace 'your-openai-api-key-here' with your actual OpenAI API key
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Model configuration
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.3

# Optional: Application settings
LOG_LEVEL=INFO
MAX_CONVERSATION_HISTORY=50

# Instructions:
# 1. Get your API key from: https://platform.openai.com/api-keys
# 2. Replace 'your-openai-api-key-here' with your actual key
# 3. Keep this file secure and never commit it to version control
# 4. The key should start with 'sk-' followed by characters
"""
        with open(env_file, 'w') as f:
            f.write(env_template)
        print("✅ .env file created")
    
    # Check OpenAI API key
    if not check_openai_key():
        return False
    
    print("✅ Setup completed successfully!")
    return True

def run_math_tutor():
    """Run the math tutor application"""
    try:
        # Add src directory to Python path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))
        
        # Import and run the main application
        from main import main
        main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required files are in place")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def show_help():
    """Show help information"""
    print("Math Tutor Agent - Help")
    print("=" * 30)
    print("Commands:")
    print("  python run.py setup    - Set up the environment")
    print("  python run.py start    - Start the math tutor")
    print("  python run.py help     - Show this help")
    print()
    print("Features:")
    print("  • Multiple prompt strategies (zero-shot, few-shot, CoT, meta)")
    print("  • Interactive Q&A session")
    print("  • Evaluation framework")
    print("  • Conversation history tracking")
    print()
    print("Usage Tips:")
    print("  • Set OPENAI_API_KEY environment variable")
    print("  • Ask math questions for grades 6-10")
    print("  • Use 'switch' command to change prompt types")
    print("  • Use 'evaluate' to run test queries")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        command = "start"
    else:
        command = sys.argv[1].lower()
    
    if command == "setup":
        setup_project()
    elif command == "start":
        print("🧮 Starting Math Tutor Agent...")
        run_math_tutor()
    elif command == "help":
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python run.py help' for available commands")

if __name__ == "__main__":
    main()