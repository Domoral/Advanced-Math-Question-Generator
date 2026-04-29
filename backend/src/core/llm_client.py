"""
LLM Client for Advanced Math Question Generation

This module provides functions to interact with DeepSeek API for:
1. Generating integrated math problems (generator)
2. Verifying and scoring math problems (verifier)
"""

import os
import re
import time
from typing import Optional, TYPE_CHECKING
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

from core.prompt_templates_CN import prompt_templates

if TYPE_CHECKING:
    from question_node import QuestionNode


# Load environment variables from .env file in the same directory as this module
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize OpenAI client with DeepSeek API
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Model name from environment variable, fallback to deepseek-reasoner
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-reasoner")


def generator(node: 'QuestionNode', new_skill: str, reference_examples: Optional[str] = None, 
              difficulty_range: Optional[tuple] = None, question_type: Optional[str] = None, max_retries: int = 3) -> str:
    """
    Generate a new math problem by integrating a new skill into the existing problem.
    
    Uses the question_generator prompt template and calls DeepSeek API.
    
    Args:
        node: The current QuestionNode containing the existing problem
        new_skill: The new knowledge point to integrate
        reference_examples: Reference examples for the new skill (optional, defaults to None)
        difficulty_range: Target difficulty range as (min, max) tuple (optional)
        question_type: Target question type (optional)
        max_retries: Maximum number of retry attempts when API returns empty content (default: 3)
        
    Returns:
        The raw LLM response string (key-value format, to be parsed by caller)
    """
    print(f"\n[DEBUG] ===== generator 函数开始 =====")
    
    # Get existing problem and skills from node
    existing_problem = node.question if node.question else "(Empty - this is the root node)"
    existing_skills = ", ".join(sorted(node.integrated_knowledge)) if node.integrated_knowledge else "None"
    
    print(f"[DEBUG] node.question 内容: {repr(existing_problem)}")
    print(f"[DEBUG] node.question 长度: {len(existing_problem)}")
    print(f"[DEBUG] existing_skills: {existing_skills}")
    print(f"[DEBUG] new_skill: {new_skill}")
    print(f"[DEBUG] is_terminal: {node.is_terminal()}")
    print(f"[DEBUG] waiting_knowledge: {node.waiting_knowledge}")
    print(f"[DEBUG] difficulty_range: {difficulty_range}")
    print(f"[DEBUG] question_type: {question_type}")
    
    # Handle None reference_examples
    if reference_examples is None:
        reference_examples = "No reference examples provided. Please use your knowledge to generate an appropriate problem."
    
    # Handle difficulty_range and question_type
    difficulty_str = f"{difficulty_range[0]:.1f}-{difficulty_range[1]:.1f}" if difficulty_range else "0.3-0.7"
    qtype_str = question_type if question_type else "计算题"
    
    # Format the prompt
    prompt = prompt_templates["question_generator"].format(
        existing_problem=existing_problem,
        existing_skills=existing_skills,
        new_skill=new_skill,
        reference_examples=reference_examples,
        difficulty_range=difficulty_str,
        question_type=qtype_str
    )
    
    print(f"[DEBUG] Prompt 长度: {len(prompt)}")
    print(f"[DEBUG] Prompt 前200字符: {prompt[:200]}")
    
    # Retry loop
    for retry_count in range(max_retries):
        # Call DeepSeek API
        try:
            print(f"[DEBUG] 开始调用 API... (尝试 {retry_count + 1}/{max_retries})")
            print(f"[DEBUG] 请求参数: model={MODEL_NAME}, temperature=0.7, max_tokens=20000")
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a skilled mathematics problem designer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=20000
            )
            
            print(f"[DEBUG] API 响应对象类型: {type(response)}")
            print(f"[DEBUG] response.choices 长度: {len(response.choices)}")
            
            if len(response.choices) == 0:
                print(f"[DEBUG] 错误: response.choices 为空！")
                print(f"[DEBUG] 完整 response 对象: {response}")
                if retry_count < max_retries - 1:
                    wait_time = (retry_count + 1) * 2
                    print(f"[DEBUG] {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                return ""
            
            choice = response.choices[0]
            print(f"[DEBUG] choice.finish_reason: {choice.finish_reason}")
            print(f"[DEBUG] choice.index: {choice.index}")
            
            if hasattr(response, 'usage'):
                print(f"[DEBUG] usage: {response.usage}")
            
            message = choice.message
            print(f"[DEBUG] message 对象存在: {message is not None}")
            print(f"[DEBUG] message.role: {message.role if message else 'N/A'}")
            print(f"[DEBUG] message.content 类型: {type(message.content)}")
            print(f"[DEBUG] message.content 值: {repr(message.content)}")
            
            result = message.content
            print(f"[DEBUG] API 返回结果长度: {len(result) if result else 0}")
            print(f"[DEBUG] API 返回结果前500字符: {result[:500] if result else 'None'}")
            
            if not result:
                print(f"[DEBUG] 警告: API 返回空内容！finish_reason={choice.finish_reason}")
                if retry_count < max_retries - 1:
                    wait_time = (retry_count + 1) * 2
                    print(f"[DEBUG] {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                print(f"[DEBUG] 已达到最大重试次数，返回空结果")
            
            print(f"[DEBUG] ===== generator 函数结束 =====\n")
            return result or ""
            
        except Exception as e:
            print(f"[DEBUG] API 调用异常: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[DEBUG] 异常堆栈:\n{traceback.format_exc()}")
            if retry_count < max_retries - 1:
                wait_time = (retry_count + 1) * 2
                print(f"[DEBUG] {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            print(f"[DEBUG] 已达到最大重试次数，返回空结果")
            print(f"[DEBUG] ===== generator 函数结束（异常） =====\n")
            return ""
    
    print(f"[DEBUG] ===== generator 函数结束 =====\n")
    return ""


def verifier(node: 'QuestionNode', reference_examples: Optional[str] = None, 
             difficulty_range: Optional[tuple] = None, max_retries: int = 3) -> str:
    """
    Verify and score a math problem using the verifier prompt template.
    
    Uses the question_verifier prompt template and calls DeepSeek API.
    
    Args:
        node: The QuestionNode containing the problem to verify
        reference_examples: Reference examples for difficulty comparison (optional)
        difficulty_range: Target difficulty range as (min, max) tuple (optional)
        max_retries: Maximum number of retry attempts when API returns empty content (default: 3)
        
    Returns:
        The raw LLM response string (containing detailed evaluation and \boxed{score})
    """
    print(f"\n[DEBUG] ===== verifier 函数开始 =====")
    
    # Get problem and required skills from node
    problem_statement = node.question
    required_skills = ", ".join(sorted(node.integrated_knowledge)) if node.integrated_knowledge else "None"
    
    print(f"[DEBUG] node.question 内容: {repr(problem_statement)}")
    print(f"[DEBUG] node.question 长度: {len(problem_statement)}")
    print(f"[DEBUG] integrated_knowledge: {required_skills}")
    print(f"[DEBUG] is_terminal: {node.is_terminal()}")
    print(f"[DEBUG] waiting_knowledge: {node.waiting_knowledge}")
    print(f"[DEBUG] difficulty_range: {difficulty_range}")
    
    # Handle None reference_examples
    if reference_examples is None:
        reference_examples = "No reference examples provided. Please evaluate based on your knowledge of standard difficulty levels."
    
    # Handle difficulty_range
    difficulty_str = f"{difficulty_range[0]:.1f}-{difficulty_range[1]:.1f}" if difficulty_range else "0.3-0.7"

    # Get solving_steps from node
    solving_steps = getattr(node, 'solving_steps', None) or "未提供解题步骤"
    print(f"[DEBUG] solving_steps: {solving_steps[:100]}..." if len(solving_steps) > 100 else f"[DEBUG] solving_steps: {solving_steps}")

    # Format the prompt
    prompt = prompt_templates["question_verifier"].format(
        problem_statement=problem_statement,
        required_skills=required_skills,
        reference_examples=reference_examples,
        difficulty_range=difficulty_str,
        solving_steps=solving_steps
    )
    
    print(f"[DEBUG] Prompt 长度: {len(prompt)}")
    print(f"[DEBUG] Prompt 前200字符: {prompt[:200]}")
    
    # Retry loop
    for retry_count in range(max_retries):
        # Call DeepSeek API
        try:
            print(f"[DEBUG] 开始调用 API... (尝试 {retry_count + 1}/{max_retries})")
            print(f"[DEBUG] 请求参数: model={MODEL_NAME}, temperature=0.3, max_tokens=20000")
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a rigorous mathematics problem verifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent evaluation
                max_tokens=20000
            )
            
            print(f"[DEBUG] API 响应对象类型: {type(response)}")
            print(f"[DEBUG] response.choices 长度: {len(response.choices)}")
            
            if len(response.choices) == 0:
                print(f"[DEBUG] 错误: response.choices 为空！")
                print(f"[DEBUG] 完整 response 对象: {response}")
                if retry_count < max_retries - 1:
                    wait_time = (retry_count + 1) * 2
                    print(f"[DEBUG] {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                return ""
            
            choice = response.choices[0]
            print(f"[DEBUG] choice.finish_reason: {choice.finish_reason}")
            print(f"[DEBUG] choice.index: {choice.index}")
            
            if hasattr(response, 'usage'):
                print(f"[DEBUG] usage: {response.usage}")
            
            message = choice.message
            print(f"[DEBUG] message 对象存在: {message is not None}")
            print(f"[DEBUG] message.role: {message.role if message else 'N/A'}")
            print(f"[DEBUG] message.content 类型: {type(message.content)}")
            print(f"[DEBUG] message.content 值: {repr(message.content)}")
            
            result = message.content
            print(f"[DEBUG] API 返回结果长度: {len(result) if result else 0}")
            print(f"[DEBUG] API 返回结果: {repr(result)}")
            
            if not result:
                print(f"[DEBUG] 警告: API 返回空内容！finish_reason={choice.finish_reason}")
                if retry_count < max_retries - 1:
                    wait_time = (retry_count + 1) * 2
                    print(f"[DEBUG] {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                print(f"[DEBUG] 已达到最大重试次数，返回空结果")
            
            print(f"[DEBUG] ===== verifier 函数结束 =====\n")
            return result or ""
            
        except Exception as e:
            print(f"[DEBUG] API 调用异常: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[DEBUG] 异常堆栈:\n{traceback.format_exc()}")
            if retry_count < max_retries - 1:
                wait_time = (retry_count + 1) * 2
                print(f"[DEBUG] {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            print(f"[DEBUG] 已达到最大重试次数，返回空结果")
            print(f"[DEBUG] ===== verifier 函数结束（异常） =====\n")
            return ""
    
    print(f"[DEBUG] ===== verifier 函数结束 =====\n")
    return ""


def extract_score(verifier_output: str) -> Optional[float]:
    """
    Extract the numerical score from verifier output.
    
    Looks for \boxed{score} pattern in the output.
    
    Args:
        verifier_output: The raw output from verifier function
        
    Returns:
        The extracted score as float, or None if not found
    """
    print(f"[DEBUG extract_score] 输入长度: {len(verifier_output)}")
    print(f"[DEBUG extract_score] 最后200字符: {repr(verifier_output[-200:])}")
    
    match = re.search(r'\\boxed\{(\d+(?:\.\d+)?)\}', verifier_output)
    if match:
        print(f"[DEBUG extract_score] 匹配成功: {match.group(1)}")
        return float(match.group(1))
    
    print(f"[DEBUG extract_score] 未找到 \\boxed{{}} 模式")
    # Try alternative patterns
    alt_match = re.search(r'boxed\{(\d+(?:\.\d+)?)\}', verifier_output)
    if alt_match:
        print(f"[DEBUG extract_score] 替代模式匹配成功: {alt_match.group(1)}")
        return float(alt_match.group(1))
    
    return None


def parse_generator_output(output: str) -> dict:
    """
    Parse the output from generator function.
    Matches the prompt template output format.
    
    Args:
        output: The raw output from generator function
        
    Returns:
        Dictionary containing parsed fields:
        - problem_statement: The generated problem
        - solving_steps: The solving steps
    """
    result = {}
    
    # Extract 新题目 (New Problem) -> problem_statement
    match = re.search(r'###\s*新题目\s*\n(.*?)(?=###|$)', output, re.DOTALL)
    if match:
        result['problem_statement'] = match.group(1).strip()
    
    # Extract 解题步骤 (Solving Steps) -> solving_steps
    match = re.search(r'###\s*解题步骤\s*\n(.*?)(?=###|$)', output, re.DOTALL)
    if match:
        result['solving_steps'] = match.group(1).strip()

    return result


def parse_verifier_output(output: str) -> dict:
    """
    Parse the verifier output to extract detailed evaluation and scores.
    
    Args:
        output: The raw output from verifier function
        
    Returns:
        Dictionary containing parsed evaluation details:
        - solution_attempt: The verifier's attempt to solve the problem
        - final_answer: The answer the verifier obtained
        - scores: Dict of individual dimension scores
        - total_score: The total score from \boxed{}
        - analyses: Dict of analysis text for each dimension
        - deduction_details: Details of deduction points
    """
    result = {
        'solution_attempt': '',
        'final_answer': '',
        'scores': {},
        'analyses': {},
        'total_score': None
    }
    
    # Extract solution attempt
    match = re.search(r'###\s*解题方法\s*\n(.*?)(?=###|$)', output, re.DOTALL)
    if match:
        result['solution_attempt'] = match.group(1).strip()
    
    # Extract final answer
    match = re.search(r'###\s*最终答案\s*\n(.*?)(?=###|$)', output, re.DOTALL)
    if match:
        result['final_answer'] = match.group(1).strip()
    
    # Extract detailed evaluation scores and analyses
    dimensions = [
        ('1', '单一答案要求'),
        ('2', '精确答案要求'),
        ('3', '技能融合'),
        ('4', '清晰性和完整性'),
        ('5', '计算可行性'),
        ('6', '语法和表达'),
        ('7', '小问数量限制'),
        ('8', '难度区间符合度'),
        ('9', '按解题步骤能否正确解答')
    ]

    for num, name in dimensions:
        # Extract analysis
        match = re.search(rf'\*\*{num}\.\s*{name}\*\*：\s*(.*?)\s*得分：', output, re.DOTALL)
        if match:
            result['analyses'][name] = match.group(1).strip()

        # Extract score
        match = re.search(rf'\*\*{num}\.\s*{name}\*\*：.*?得分：\s*(\d+(?:\.\d+)?)\s*分', output)
        if match:
            result['scores'][name] = float(match.group(1))

    # Extract deduction details
    match = re.search(r'###\s*扣分点详情\s*\n(.*?)(?=---|\*\*最终得分框|$)', output, re.DOTALL)
    if match:
        result['deduction_details'] = match.group(1).strip()

    # Extract total score from \boxed{}
    match = re.search(r'\\boxed\{(\d+(?:\.\d+)?)\}', output)
    if match:
        result['total_score'] = float(match.group(1))

    return result


def optimizer(node: 'QuestionNode', deduction_points: str, max_retries: int = 3) -> str:
    """
    Optimize an existing math problem based on verifier feedback.
    
    Uses the question_optimizer prompt template and calls DeepSeek API.
    
    Args:
        node: The QuestionNode containing the problem to optimize
        deduction_points: Natural language description of deduction points
        max_retries: Maximum number of retry attempts when API returns empty content (default: 3)
        
    Returns:
        The raw LLM response string containing the optimized problem
    """
    print(f"\n[DEBUG] ===== optimizer 函数开始 =====")
    
    # Get problem and required skills from node
    existing_problem = node.question
    existing_skills = ", ".join(sorted(node.integrated_knowledge)) if node.integrated_knowledge else "None"
    last_reward = node.last_reward if node.last_reward else 0.0
    
    print(f"[DEBUG] node.question 内容: {repr(existing_problem)}")
    print(f"[DEBUG] integrated_knowledge: {existing_skills}")
    print(f"[DEBUG] last_reward: {last_reward}")
    print(f"[DEBUG] deduction_points: {deduction_points}")
    
    # Use deduction_points directly as deduction_details
    deduction_details = deduction_points if deduction_points else "无明显扣分点"
    
    # Get difficulty range and question type from node
    difficulty_range = getattr(node, 'difficulty_range', (0.3, 0.7))
    question_type = getattr(node, 'question_type', '计算题')
    difficulty_str = f"{difficulty_range[0]:.1f}-{difficulty_range[1]:.1f}"

    solving_steps = getattr(node, 'solving_steps', None) or "无解题步骤信息"

    print(f"[DEBUG] difficulty_range: {difficulty_str}")
    print(f"[DEBUG] question_type: {question_type}")

    # Format the prompt
    prompt = prompt_templates["question_optimizer"].format(
        existing_problem=existing_problem,
        existing_skills=existing_skills,
        last_reward=f"{last_reward:.1f}",
        solving_steps=solving_steps,
        deduction_details=deduction_details,
        difficulty_range=difficulty_str,
        question_type=question_type
    )
    
    print(f"[DEBUG] Prompt 长度: {len(prompt)}")
    print(f"[DEBUG] Prompt 前200字符: {prompt[:200]}")
    
    # Retry loop
    for retry_count in range(max_retries):
        # Call DeepSeek API
        try:
            print(f"[DEBUG] 开始调用 API... (尝试 {retry_count + 1}/{max_retries})")
            print(f"[DEBUG] 请求参数: model={MODEL_NAME}, temperature=0.7, max_tokens=20000")
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a skilled mathematics problem optimization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=20000
            )
            
            print(f"[DEBUG] API 响应对象类型: {type(response)}")
            print(f"[DEBUG] response.choices 长度: {len(response.choices)}")
            
            if len(response.choices) == 0:
                print(f"[DEBUG] 错误: response.choices 为空！")
                if retry_count < max_retries - 1:
                    wait_time = (retry_count + 1) * 2
                    print(f"[DEBUG] {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                return ""
            
            choice = response.choices[0]
            print(f"[DEBUG] choice.finish_reason: {choice.finish_reason}")
            
            message = choice.message
            result = message.content
            
            print(f"[DEBUG] API 返回结果长度: {len(result) if result else 0}")
            print(f"[DEBUG] API 返回结果前500字符: {result[:500] if result else 'None'}")
            
            if not result:
                print(f"[DEBUG] 警告: API 返回空内容！")
                if retry_count < max_retries - 1:
                    wait_time = (retry_count + 1) * 2
                    print(f"[DEBUG] {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
            
            print(f"[DEBUG] ===== optimizer 函数结束 =====\n")
            return result or ""
            
        except Exception as e:
            print(f"[DEBUG] API 调用异常: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[DEBUG] 异常堆栈:\n{traceback.format_exc()}")
            if retry_count < max_retries - 1:
                wait_time = (retry_count + 1) * 2
                print(f"[DEBUG] {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            print(f"[DEBUG] ===== optimizer 函数结束（异常） =====\n")
            return ""
    
    print(f"[DEBUG] ===== optimizer 函数结束 =====\n")
    return ""


def parse_optimizer_output(output: str) -> dict:
    """
    Parse the output from optimizer function.
    
    Args:
        output: The raw output from optimizer function
        
    Returns:
        Dictionary containing parsed fields:
        - optimized_problem: The optimized problem statement
        - solving_steps: Solving steps for the optimized problem
    """
    result = {}
    
    # Extract optimized problem
    match = re.search(r'###\s*优化后的题目\s*\n(.*?)(?=###|$)', output, re.DOTALL)
    if match:
        result['optimized_problem'] = match.group(1).strip()
    
    # Extract solving steps
    match = re.search(r'###\s*解题步骤\s*\n(.*?)(?=###|$)', output, re.DOTALL)
    if match:
        result['solving_steps'] = match.group(1).strip()
    
    return result
