"""
LLM Client for Advanced Math Question Generation

This module provides functions to interact with DeepSeek API for:
1. Generating integrated math problems (generator)
2. Verifying and scoring math problems (verifier)
"""

import os
import re
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

from prompt_templates_EN import prompt_templates
from question_node import QuestionNode


# Load environment variables
load_dotenv()

# Initialize OpenAI client with DeepSeek API
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Model name from environment variable, fallback to deepseek-reasoner
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-reasoner")


def generator(node: QuestionNode, new_skill: str, reference_examples: Optional[str] = None) -> str:
    """
    Generate a new math problem by integrating a new skill into the existing problem.
    
    Uses the question_generator prompt template and calls DeepSeek API.
    
    Args:
        node: The current QuestionNode containing the existing problem
        new_skill: The new knowledge point to integrate
        reference_examples: Reference examples for the new skill (optional, defaults to None)
        
    Returns:
        The raw LLM response string (key-value format, to be parsed by caller)
    """
    # Get existing problem and skills from node
    existing_problem = node.question if node.question else "(Empty - this is the root node)"
    existing_skills = ", ".join(sorted(node.integrated_knowledge)) if node.integrated_knowledge else "None"
    
    # Handle None reference_examples
    if reference_examples is None:
        reference_examples = "No reference examples provided. Please use your knowledge to generate an appropriate problem."
    
    # Format the prompt
    prompt = prompt_templates["question_generator"].format(
        existing_problem=existing_problem,
        existing_skills=existing_skills,
        new_skill=new_skill,
        reference_examples=reference_examples
    )
    
    # Call DeepSeek API
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a skilled mathematics problem designer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048
    )
    
    return response.choices[0].message.content


def verifier(node: QuestionNode, reference_examples: str) -> str:
    """
    Verify and score a math problem using the verifier prompt template.
    
    Uses the question_verifier prompt template and calls DeepSeek API.
    
    Args:
        node: The QuestionNode containing the problem to verify
        reference_examples: Reference examples for difficulty comparison
        
    Returns:
        The raw LLM response string (containing detailed evaluation and \boxed{score})
    """
    # Get problem and required skills from node
    problem_statement = node.question
    required_skills = ", ".join(sorted(node.integrated_knowledge)) if node.integrated_knowledge else "None"
    
    # Format the prompt
    prompt = prompt_templates["question_verifier"].format(
        problem_statement=problem_statement,
        required_skills=required_skills,
        reference_examples=reference_examples
    )
    
    # Call DeepSeek API
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a rigorous mathematics problem verifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for more consistent evaluation
        max_tokens=2048
    )
    
    return response.choices[0].message.content


def extract_score(verifier_output: str) -> Optional[float]:
    """
    Extract the numerical score from verifier output.
    
    Looks for \boxed{score} pattern in the output.
    
    Args:
        verifier_output: The raw output from verifier function
        
    Returns:
        The extracted score as float, or None if not found
    """
    match = re.search(r'\\boxed\{(\d+(?:\.\d+)?)\}', verifier_output)
    if match:
        return float(match.group(1))
    
    return None


def parse_generator_output(output: str) -> dict:
    """
    Parse the key-value output from generator function.
    Supports both English and Chinese format outputs.
    
    Args:
        output: The raw output from generator function
        
    Returns:
        Dictionary containing parsed fields (problem_statement, final_answer, etc.)
    """
    result = {}
    
    # Extract triple-quoted fields (English format)
    # problem_statement
    match = re.search(r'problem_statement:\s*"""(.*?)"""', output, re.DOTALL)
    if match:
        result['problem_statement'] = match.group(1).strip()
    
    # solution_path
    match = re.search(r'solution_path:\s*"""(.*?)"""', output, re.DOTALL)
    if match:
        result['solution_path'] = match.group(1).strip()
    
    # Extract single-line fields (English format)
    # integration_rationale
    match = re.search(r'integration_rationale:\s*(.+?)(?:\n|$)', output)
    if match:
        result['integration_rationale'] = match.group(1).strip()
    
    # final_answer
    match = re.search(r'final_answer:\s*(.+?)(?:\n|$)', output)
    if match:
        result['final_answer'] = match.group(1).strip()
    
    # difficulty_estimate
    match = re.search(r'difficulty_estimate:\s*(\d+)', output)
    if match:
        result['difficulty_estimate'] = int(match.group(1))
    
    # prerequisite_skills
    match = re.search(r'prerequisite_skills:\s*(.+?)(?:\n|$)', output)
    if match:
        skills = match.group(1).strip()
        result['prerequisite_skills'] = [s.strip() for s in skills.split(',')]
    
    # Extract Chinese format fields (if English format not found)
    # 新题目 (New Problem) -> problem_statement
    if 'problem_statement' not in result:
        match = re.search(r'###\s*新题目\s*\n(.*?)(?=###|$)', output, re.DOTALL)
        if match:
            result['problem_statement'] = match.group(1).strip()
    
    # 预期解题路径 (Expected Solution Path) -> solution_path
    if 'solution_path' not in result:
        match = re.search(r'###\s*预期解题路径\s*\n(.*?)(?=###|$)', output, re.DOTALL)
        if match:
            result['solution_path'] = match.group(1).strip()
    
    # 融合原理 (Integration Rationale) -> integration_rationale
    if 'integration_rationale' not in result:
        match = re.search(r'###\s*融合原理\s*\n(.*?)(?=###|$)', output, re.DOTALL)
        if match:
            result['integration_rationale'] = match.group(1).strip()
    
    # 最终答案 (Final Answer) -> final_answer
    if 'final_answer' not in result:
        match = re.search(r'###\s*最终答案\s*\n(.*?)(?=###|$)', output, re.DOTALL)
        if match:
            result['final_answer'] = match.group(1).strip()
    
    return result
