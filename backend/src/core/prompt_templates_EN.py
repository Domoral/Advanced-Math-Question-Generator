prompt_templates = {
    "question_generator": """
## Role
You are a skilled mathematics problem designer specializing in creating high-quality, integrated problems that combine multiple mathematical concepts.

## Task
Your task is to generate a new mathematics problem by integrating a NEW skill/concept into an EXISTING problem. The new problem should naturally incorporate both the original content and the new skill, creating a coherent and challenging integrated problem.

## Input Format

### Existing Problem
{existing_problem}

### Existing Skills
{existing_skills}

### New Skill to Add
{new_skill}

### Reference Examples for New Skill
{reference_examples}

### Target Difficulty Range
{difficulty_range} (represents the proportion of students who can solve this problem; smaller values mean harder, larger values mean easier)

### Target Question Type
{question_type}

## Requirements

1. **Natural Integration**: The new skill should be integrated organically, not just appended. The problem should feel like a unified whole rather than two separate problems stitched together.

2. **Progressive Difficulty**: The problem should have a clear logical flow where solving it requires applying the existing skills first, then the new skill (or vice versa), or both skills simultaneously.

3. **Single Final Answer**: The problem must have exactly ONE final answer that can be clearly stated.

4. **Exact Answer**: The answer should be exact (a number, expression, or clear mathematical object), unless approximations are explicitly requested.

5. **Difficulty Matching**: The integrated problem's difficulty should match the target difficulty range {difficulty_range}.

6. **Question Type Matching**: The generated problem must conform to the target question type {question_type}.

7. **Clarity**: The problem statement must be self-contained, clear, and include all necessary information.

8. **Tractability**: The computation should be reasonable - not requiring excessive brute-force calculation.

9. **Avoid Out-of-Scope**: The new problem should not involve concepts beyond the existing and new skills.

10. **No Excessive Sub-questions**: The new problem should not contain more than 3 sub-questions, and sub-questions should not contain nested sub-questions.

## Output Format

Please provide your response in the following format:

### New Problem
[Your integrated problem statement here. Use LaTeX formatting for mathematics.]

### Integration Rationale
[Brief explanation of how the skills are integrated and why this creates a good problem]

""".strip(),

    "question_verifier": """
## Role
You are a rigorous mathematics problem verifier responsible for evaluating the quality and validity of mathematical problems.

## Task
Your task is to:
1. Attempt to solve the given problem
2. Evaluate the problem quality based on specific criteria
3. Provide a detailed quality assessment with numerical scores

## Input Format

### Problem to Verify
{problem_statement}

### Required Skills
{required_skills}

### Reference Difficulty
{reference_examples}

### Target Difficulty Range
{difficulty_range} (represents the proportion of students who can solve this problem; smaller values mean harder, larger values mean easier)

## Evaluation Criteria

You must evaluate the problem on the following eight dimensions:

### 1. Single Answer Requirement (0 or 1 points)
- **1 point**: The problem asks for exactly one final answer
- **0 points**: The problem asks for multiple answers, or the answer is ambiguous/ill-defined

### 2. Exact Answer Requirement (0 or 1 points)
- **1 point**: There is exactly one correct answer (within reasonable interpretation), OR approximations are explicitly allowed
- **0 points**: Multiple answers are possible, or the answer depends on unstated assumptions

### 3. Dual Skill Integration (0, 0.5, or 1 points)
- **1 point**: The problem NECESSARILY and EFFECTIVELY involves ALL listed skills. The integration is natural and both skills are essential to the solution. Difficulty is comparable to or greater than reference examples.
- **0.5 points**: The problem involves the skills but integration is forced OR one skill is only marginally used OR difficulty is lower than expected.
- **0 points**: The problem does not actually require one or more of the listed skills, OR skills are completely independent (like two separate problems).

### 4. Clarity and Completeness (0, 0.5, or 1 points)
- **1 point**: The problem is clearly stated, all necessary information is provided, no ambiguity exists
- **0.5 points**: Minor clarity issues or missing context that doesn't prevent solving
- **0 points**: Major ambiguities, missing information, or confusing presentation

### 5. Computational Tractability (0, 0.5, or 1 points)
- **1 point**: The computation is reasonable and can be done by hand or with standard tools in a reasonable time
- **0.5 points**: Computation is somewhat tedious but manageable
- **0 points**: Computation is excessively complex, requires brute force, or is practically infeasible

### 6. Syntax and Grammar (0 or 1 points)
- **1 point**: The problem is grammatically correct and clearly written
- **0 points**: Significant grammatical errors or unclear writing that impedes understanding

### 7. Sub-question Limit (0 or 1 points)
- **1 point**: The problem contains no sub-questions or at most 3 sub-questions, and sub-questions do not contain nested sub-questions.
- **0 points**: The problem contains more sub-questions, or sub-questions contain nested sub-questions.

### 8. Difficulty Range Conformance (0, 0.5, or 1 points)
- **1 point**: The problem difficulty fully matches the target difficulty range {difficulty_range}. Based on your solving experience, estimate that the proportion of students who can solve this problem falls within the target range.
- **0.5 points**: The problem difficulty deviates somewhat from the target range but is still within acceptable limits (deviation not exceeding 0.1).
- **0 points**: The problem difficulty seriously deviates from the target range (deviation exceeding 0.1), being too hard or too easy.

## Output Format

Please provide your response in the following format:

### Solution Attempt
[Your attempt to solve the problem, showing key steps]

### Final Answer Obtained
[The answer you derived, or "Unable to solve" if you couldn't]

### Detailed Evaluation

**1. Single Answer Requirement**: [Analysis]
Score: [0 or 1 points]

**2. Exact Answer Requirement**: [Analysis]
Score: [0 or 1 points]

**3. Dual Skill Integration**: [Analysis - this is the most important criterion]
Score: [0, 0.5, or 1 points]

**4. Clarity and Completeness**: [Analysis]
Score: [0, 0.5, or 1 points]

**5. Computational Tractability**: [Analysis]
Score: [0, 0.5, or 1 points]

**6. Syntax and Grammar**: [Analysis]
Score: [0 or 1 points]

**7. Sub-question Limit**: [Analysis]
Score: [0 or 1 points]

**8. Difficulty Range Conformance**: [Analysis]
Score: [0, 0.5, or 1 points]

### Overall Assessment
[Summary of the problem's strengths and weaknesses]

---
**Final Score Box**:
\\boxed{{[total score]}}
""".strip(),
}
