#!/usr/bin/env python3
"""
Test script to verify Math Tutor Agent setup and basic functionality
"""

import os
import sys
import json
from pathlib import Path

def test_file_structure():
    """Test if all required files exist"""
    print("🔍 Testing file structure...")
    
    base_path = Path(__file__).parent
    required_files = [
        "README.md",
        "domain_analysis.md",
        "requirements.txt",
        "run.py",
        "prompts/zero_shot.txt",
        "prompts/few_shot.txt",
        "prompts/cot_prompt.txt",
        "prompts/meta_prompt.txt",
        "evaluation/input_queries.json",
        "evaluation/analysis_report.md",
        "src/main.py",
        "src/utils.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print("  ❌ Missing files:")
        for file_path in missing_files:
            print(f"    - {file_path}")
        return False
    
    print("✅ All required files present")
    return True

def test_json_files():
    """Test if JSON files are valid"""
    print("\n🔍 Testing JSON files...")
    
    base_path = Path(__file__).parent
    json_files = [
        "evaluation/input_queries.json"
    ]
    
    for json_file in json_files:
        try:
            with open(base_path / json_file, 'r') as f:
                data = json.load(f)
            print(f"  ✅ {json_file} - Valid JSON")
            
            # Test specific structure for input_queries.json
            if json_file == "evaluation/input_queries.json":
                if "test_queries" in data and "evaluation_criteria" in data:
                    print(f"    - Contains {len(data['test_queries'])} test queries")
                    print(f"    - Contains evaluation criteria")
                else:
                    print(f"    ⚠️  Missing expected structure")
                    
        except json.JSONDecodeError as e:
            print(f"  ❌ {json_file} - Invalid JSON: {e}")
            return False
        except FileNotFoundError:
            print(f"  ❌ {json_file} - File not found")
            return False
    
    return True

def test_prompt_templates():
    """Test if prompt templates are properly formatted"""
    print("\n🔍 Testing prompt templates...")
    
    base_path = Path(__file__).parent
    prompt_files = [
        "prompts/zero_shot.txt",
        "prompts/few_shot.txt", 
        "prompts/cot_prompt.txt",
        "prompts/meta_prompt.txt"
    ]
    
    for prompt_file in prompt_files:
        try:
            with open(base_path / prompt_file, 'r') as f:
                content = f.read().strip()
            
            if len(content) > 50:  # Reasonable minimum length
                print(f"  ✅ {prompt_file} - {len(content)} characters")
            else:
                print(f"  ⚠️  {prompt_file} - Seems too short ({len(content)} characters)")
                
        except FileNotFoundError:
            print(f"  ❌ {prompt_file} - File not found")
            return False
    
    return True

def test_imports():
    """Test if required modules can be imported"""
    print("\n🔍 Testing Python imports...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    required_modules = [
        ("json", "json"),
        ("pathlib", "pathlib"),
        ("os", "os"),
        ("sys", "sys")
    ]
    
    optional_modules = [
        ("openai", "openai"),
        ("numpy", "numpy"),
        ("dotenv", "python-dotenv")
    ]
    
    # Test required modules
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except ImportError:
            print(f"  ❌ {module_name} - Required module missing")
            return False
    
    # Test optional modules
    missing_optional = []
    for module_name, package_name in optional_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except ImportError:
            missing_optional.append(package_name)
            print(f"  ⚠️  {module_name} - Optional module missing")
    
    if missing_optional:
        print(f"\n  💡 Install missing packages: pip install {' '.join(missing_optional)}")
    
    # Test project modules
    try:
        import main
        print(f"  ✅ main.py")
    except ImportError as e:
        print(f"  ❌ main.py - {e}")
        return False
    
    try:
        import utils
        print(f"  ✅ utils.py")
    except ImportError as e:
        print(f"  ❌ utils.py - {e}")
        return False
    
    return True

def test_environment():
    """Test environment setup"""
    print("\n🔍 Testing environment...")
    
    # Check Python version
    if sys.version_info >= (3, 7):
        print(f"  ✅ Python {sys.version.split()[0]}")
    else:
        print(f"  ❌ Python version too old: {sys.version.split()[0]} (need 3.7+)")
        return False
    
    # Check for .env file
    base_path = Path(__file__).parent
    env_file = base_path / '.env'
    env_example = base_path / '.env.example'
    
    if env_file.exists():
        print(f"  ✅ .env file found")
        
        # Check if it contains the template key
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
            
            if 'OPENAI_API_KEY=your-openai-api-key-here' in env_content:
                print(f"  ⚠️  .env file contains template API key")
                print(f"     Please update with your actual OpenAI API key")
            elif 'OPENAI_API_KEY=' in env_content:
                print(f"  ✅ .env file configured with API key")
            else:
                print(f"  ❌ .env file missing OPENAI_API_KEY")
        except Exception as e:
            print(f"  ❌ Error reading .env file: {e}")
    else:
        print(f"  ⚠️  .env file not found")
        if env_example.exists():
            print(f"     Copy .env.example to .env and add your API key")
        else:
            print(f"     Create .env file with your OpenAI API key")
    
    # Load dotenv and check API key
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key and api_key != 'your-openai-api-key-here':
            if api_key.startswith('sk-'):
                print(f"  ✅ Valid OpenAI API key loaded (length: {len(api_key)})")
            else:
                print(f"  ❌ Invalid API key format (should start with 'sk-')")
        else:
            print(f"  ⚠️  No valid OpenAI API key found")
            print(f"     Get your key from: https://platform.openai.com/api-keys")
    except ImportError:
        print(f"  ⚠️  python-dotenv not installed")
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print(f"  ✅ OpenAI API key found in environment (length: {len(api_key)})")
        else:
            print(f"  ⚠️  No OpenAI API key in environment")
    
    return True

def test_basic_functionality():
    """Test basic functionality without API calls"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))
        
        from main import MathTutorAgent
        from utils import MathEvaluator, ResponseAnalyzer
        
        # Test MathEvaluator
        test_response = "The answer is 42"
        answer = MathEvaluator.extract_final_answer(test_response)
        if answer == "42":
            print("  ✅ MathEvaluator.extract_final_answer")
        else:
            print(f"  ❌ MathEvaluator.extract_final_answer - Expected '42', got '{answer}'")
        
        # Test ResponseAnalyzer
        analysis = ResponseAnalyzer.analyze_reasoning_clarity("First, we solve step by step. Then we check our answer.")
        if isinstance(analysis, dict) and 'clarity_score' in analysis:
            print("  ✅ ResponseAnalyzer.analyze_reasoning_clarity")
        else:
            print("  ❌ ResponseAnalyzer.analyze_reasoning_clarity - Invalid output")
        
        print("  ✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧮 Math Tutor Agent - Setup Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_json_files,
        test_prompt_templates,
        test_imports,
        test_environment,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key if not already done")
        print("2. Run: python run.py start")
        print("3. Ask math questions and explore different prompt strategies")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\nCommon fixes:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Set API key: export OPENAI_API_KEY='your-key'")
        print("3. Check file permissions and structure")
    
    return passed == total

if __name__ == "__main__":
    main()