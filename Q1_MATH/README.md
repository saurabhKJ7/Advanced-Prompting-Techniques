# Advanced Prompting Techniques - EdTech Math Tutor

🧮 **An AI-powered mathematics tutor for students in grades 6-10, featuring multiple prompt engineering strategies**

## Overview

This project demonstrates advanced prompt engineering techniques by implementing a domain-specific LLM agent for mathematics education. The system compares different prompting strategies including zero-shot, few-shot, chain-of-thought (CoT), and meta-prompting approaches.

## Features

### 🎯 Core Capabilities
- **Interactive Math Tutoring**: Ask mathematical questions and get detailed explanations
- **Multiple Prompt Strategies**: Compare different AI reasoning approaches
- **Grade-Level Adaptation**: Content appropriate for grades 6-10
- **Step-by-Step Solutions**: Clear mathematical reasoning and problem-solving
- **Evaluation Framework**: Systematic testing of prompt effectiveness

### 📚 Supported Topics
- **Algebra**: Linear equations, quadratic expressions, systems of equations
- **Geometry**: Area, perimeter, volume, Pythagorean theorem
- **Arithmetic**: Basic operations, fractions, decimals, percentages
- **Word Problems**: Real-world mathematical applications
- **Probability**: Basic probability concepts and calculations

### 🧠 Prompt Strategies

1. **Zero-Shot**: Direct instruction-based responses
2. **Few-Shot**: Learning from provided examples
3. **Chain-of-Thought (CoT)**: Explicit step-by-step reasoning
4. **Meta-Prompting**: Self-reflective and adaptive responses

## Quick Start

### Prerequisites
- Python 3.7 or higher
- OpenAI API key
- Internet connection

### Installation

1. **Clone or download the project**
```bash
git clone <repository-url>
cd Advanced-Prompting-Techniques
```

2. **Set up your OpenAI API key**
```bash
# Method 1: Use the .env file (Recommended)
# Copy the example file and edit it
cp .env.example .env
# Then edit .env and replace 'your-openai-api-key-here' with your actual key

# Method 2: Environment variable
# Linux/Mac
export OPENAI_API_KEY='your-api-key-here'

# Windows
set OPENAI_API_KEY=your-api-key-here
```

**Get your API key from:** https://platform.openai.com/api-keys

3. **Run the setup**
```bash
python run.py setup
```

4. **Start the tutor**
```bash
python run.py start
```

## Usage Guide

### Interactive Mode

Once started, you can:

- **Ask math questions**: Type any mathematics question for grades 6-10
- **Switch prompt types**: Use `switch` command to change reasoning strategies
- **View history**: Use `history` command to see previous interactions
- **Run evaluation**: Use `evaluate` command to test all strategies
- **Exit**: Use `quit` or `exit` to close the application

### Example Questions

```
📝 Ask your math question: Solve the equation: 2x + 5 = 13
📝 Ask your math question: What is the area of a rectangle with length 8cm and width 5cm?
📝 Ask your math question: A train travels 240 km in 3 hours. What is its speed?
📝 Ask your math question: What is the Pythagorean theorem?
```

### Prompt Strategy Comparison

You can compare how different strategies handle the same question:

```
📝 Ask your math question: switch
Select prompt type:
1. zero_shot
2. few_shot  
3. cot
4. meta
```

## Project Structure

```
Advanced-Prompting-Techniques/
├── README.md                     # This file
├── domain_analysis.md           # Domain requirements analysis
├── requirements.txt             # Python dependencies
├── run.py                      # Setup and run script
├── prompts/                    # Prompt templates
│   ├── zero_shot.txt          # Direct instruction prompt
│   ├── few_shot.txt           # Example-based prompt
│   ├── cot_prompt.txt         # Chain-of-thought prompt
│   └── meta_prompt.txt        # Self-reflective prompt
├── evaluation/                 # Evaluation framework
│   ├── input_queries.json     # Test questions
│   ├── output_logs.json       # Results (generated)
│   └── analysis_report.md     # Analysis template
├── src/                       # Source code
│   ├── main.py               # Main application
│   └── utils.py              # Utility functions
└── hallucination_log.md      # Error tracking (generated)
```

## Evaluation Framework

### Metrics

The system evaluates responses across four dimensions:

1. **Accuracy (0-3)**: Mathematical correctness
2. **Reasoning Clarity (0-3)**: Step-by-step explanation quality
3. **Hallucination Score (0-3)**: Factual consistency
4. **Consistency (0-3)**: Uniform response structure

### Test Queries

The evaluation includes 10 representative questions covering:
- Different grade levels (6-10)
- Various mathematical topics
- Concept explanations and problem-solving
- Word problems and direct calculations

### Running Evaluations

```bash
# In interactive mode, type:
evaluate

# Or run specific comparisons
python -m src.main --evaluate
```

## Advanced Features

### Fallback Mechanisms

The system handles ambiguous inputs by:
- Requesting clarification for unclear questions
- Suggesting missing information
- Providing grade-appropriate explanations
- Offering alternative solution methods

### Conversation Context

- Tracks conversation history
- Adapts to user's mathematical level
- Maintains context across questions
- Logs interactions for analysis

### Error Handling

- Graceful handling of API errors
- Mathematical accuracy verification
- Input validation and sanitization
- Comprehensive logging

## Development

### Adding New Prompts

1. Create a new `.txt` file in `prompts/` directory
2. Add the prompt type to `src/main.py` template loading
3. Test with evaluation framework

### Customizing Evaluation

1. Modify `evaluation/input_queries.json` to add test cases
2. Adjust scoring criteria in the evaluation framework
3. Update analysis templates as needed

### Extending Functionality

- Add new mathematical topics in domain analysis
- Implement additional evaluation metrics
- Create specialized prompts for specific use cases

## Troubleshooting

### Common Issues

**API Key Not Found**
```
❌ Configuration Error: OpenAI API key is required
```
Solution: Set the OPENAI_API_KEY environment variable

**Import Errors**
```
❌ Import error: No module named 'openai'
```
Solution: Run `python run.py setup` to install dependencies

**Connection Issues**
```
❌ Error getting OpenAI response: Connection error
```
Solution: Check internet connection and API key validity

### Debug Mode

Enable detailed logging by setting:
```bash
export LOG_LEVEL=DEBUG
```

## Performance Optimization

### API Usage
- Responses cached during evaluation
- Token usage tracked and displayed
- Temperature settings optimized for mathematical accuracy

### Response Quality
- Mathematical notation properly formatted
- Grade-level language adaptation
- Verification steps included in solutions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## Educational Applications

### For Teachers
- Demonstrate different AI reasoning approaches
- Analyze student-AI interactions
- Customize prompts for specific curricula

### For Students
- Get personalized mathematical help
- Learn problem-solving strategies
- Practice with immediate feedback

### For Researchers
- Study prompt engineering effectiveness
- Analyze mathematical reasoning patterns
- Develop improved educational AI systems

## License

This project is provided for educational purposes. Please ensure compliance with OpenAI's usage policies when using their API.

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review existing documentation
3. Create detailed issue reports
4. Provide example inputs and expected outputs

---

**Built with advanced prompt engineering techniques for educational excellence** 🚀