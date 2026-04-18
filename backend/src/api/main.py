"""
FastAPI backend for Advanced Math Question Generator

Provides REST API endpoints for frontend to:
1. Generate questions via MCTS
2. Get generation status/progress
3. Retrieve generated questions
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for importing core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.question_node import QuestionNode, QuestionMCTS


app = FastAPI(
    title="高等数学综合题生成系统",
    description="Advanced Math Question Generator API",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (configure for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class GenerateRequest(BaseModel):
    """Request model for question generation"""
    knowledge_points: List[str]
    difficulty_range: Tuple[float, float] = (0.3, 0.7)
    question_type: str = "计算题"
    use_rag: bool = False
    max_iterations: int = 100
    target_leaf_nodes: int = 4
    save_threshold: float = 9


class GenerateResponse(BaseModel):
    """Response model for generation status"""
    status: str
    message: str
    output_dir: Optional[str] = None
    generated_count: int = 0


class QuestionData(BaseModel):
    """Model for a generated question"""
    id: str
    question: str
    knowledge_points: List[str]
    difficulty: float
    question_type: str
    metadata: dict


class StatusResponse(BaseModel):
    """Response model for detailed status with logs"""
    task_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    output_dir: Optional[str] = None
    generated_count: int = 0
    logs: List[str] = []
    progress: dict = {}


# ============== Global State ==============

generation_tasks = {}  # Store ongoing generation tasks


def get_task_log_path(task_id: str) -> str:
    """Get the log file path for a task."""
    backend_dir = Path(__file__).parent.parent
    project_root = backend_dir.parent
    log_dir = project_root / "backend" / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / f"{task_id}.log")


class TaskLogger:
    """Logger that writes to both console and file."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.log_file = get_task_log_path(task_id)
        self.logs = []
    
    def log(self, message: str):
        """Log a message to both console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        
        # Also print to console
        print(log_entry)
    
    def get_logs(self, last_n: int = 50) -> List[str]:
        """Get last N log entries."""
        return self.logs[-last_n:]


# Store loggers for active tasks
task_loggers = {}


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "高等数学综合题生成系统 API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "Start question generation",
            "GET /status/{task_id}": "Get generation status",
            "GET /questions": "List generated questions",
            "GET /questions/{question_id}": "Get specific question"
        }
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_questions(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start question generation task.
    
    Args:
        request: Generation parameters including knowledge points, difficulty, etc.
        
    Returns:
        Generation status and task ID
    """
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Validate inputs
    if not request.knowledge_points:
        raise HTTPException(status_code=400, detail="请至少选择一个知识点")
    
    if len(request.difficulty_range) != 2:
        raise HTTPException(status_code=400, detail="难度范围格式错误")
    
    # Store task info
    generation_tasks[task_id] = {
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "request": request.dict(),
        "output_dir": None,
        "generated_count": 0
    }
    
    # Start generation in background
    background_tasks.add_task(run_generation, task_id, request)
    
    return GenerateResponse(
        status="started",
        message=f"生成任务已启动 (Task ID: {task_id})",
        output_dir=None,
        generated_count=0
    )


@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str, lines: int = 50):
    """
    Get generation task status with logs.
    
    Args:
        task_id: The task ID returned by /generate
        lines: Number of recent log lines to return (default 50)
        
    Returns:
        Current status of the generation task with logs
    """
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = generation_tasks[task_id]
    
    # Get logs from memory or file
    logs = []
    if task_id in task_loggers:
        logs = task_loggers[task_id].get_logs(last_n=lines)
    else:
        # Try to read from log file
        log_file = get_task_log_path(task_id)
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                all_logs = f.readlines()
                logs = [line.strip() for line in all_logs[-lines:]]
    
    # Calculate progress
    progress = {}
    if task.get("max_iterations"):
        current_iter = task.get("current_iteration", 0)
        progress["iteration"] = f"{current_iter}/{task['max_iterations']}"
        progress["percentage"] = round((current_iter / task["max_iterations"]) * 100, 1)
    
    return StatusResponse(
        task_id=task_id,
        status=task["status"],
        start_time=task["start_time"],
        end_time=task.get("end_time"),
        output_dir=task.get("output_dir"),
        generated_count=task.get("generated_count", 0),
        logs=logs,
        progress=progress
    )


@app.get("/questions")
async def list_questions(output_dir: Optional[str] = None):
    """
    List all generated questions.
    
    Args:
        output_dir: Optional specific output directory to list
        
    Returns:
        List of generated question files
    """
    if output_dir and os.path.exists(output_dir):
        target_dir = output_dir
    else:
        # Default to latest generated_question directory
        base_dir = Path("./generated_question")
        if not base_dir.exists():
            return {"questions": [], "count": 0}
        
        # Find latest subdirectory
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return {"questions": [], "count": 0}
        
        target_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    
    # List JSON files
    questions = []
    for f in Path(target_dir).glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                questions.append({
                    "id": f.stem,
                    "filename": f.name,
                    "knowledge_points": data.get("knowledge_points", []),
                    "difficulty": data.get("difficulty", 0),
                    "question_type": data.get("question_type", "未知"),
                    "created_at": data.get("metadata", {}).get("timestamp", "")
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    return {
        "questions": questions,
        "count": len(questions),
        "output_dir": str(target_dir)
    }


@app.get("/questions/{question_id}")
async def get_question(question_id: str, output_dir: Optional[str] = None):
    """
    Get a specific generated question.
    
    Args:
        question_id: The question ID (filename without extension)
        output_dir: Optional specific output directory
        
    Returns:
        Full question data
    """
    if output_dir and os.path.exists(output_dir):
        target_dir = Path(output_dir)
    else:
        base_dir = Path("./generated_question")
        if not base_dir.exists():
            raise HTTPException(status_code=404, detail="未找到生成目录")
        
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not subdirs:
            raise HTTPException(status_code=404, detail="未找到生成目录")
        
        target_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    
    # Find question file
    question_file = target_dir / f"{question_id}.json"
    if not question_file.exists():
        raise HTTPException(status_code=404, detail="题目不存在")
    
    try:
        with open(question_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取题目失败: {str(e)}")


# ============== Background Task ==============

def run_generation(task_id: str, request: GenerateRequest):
    """
    Run MCTS generation in background with logging.
    
    Args:
        task_id: Task identifier
        request: Generation request parameters
    """
    # Create logger for this task
    logger = TaskLogger(task_id)
    task_loggers[task_id] = logger
    
    try:
        logger.log(f"Starting generation task {task_id}")
        logger.log(f"Knowledge points: {request.knowledge_points}")
        logger.log(f"Difficulty range: {request.difficulty_range}")
        logger.log(f"Question type: {request.question_type}")
        logger.log(f"Use RAG: {request.use_rag}")
        
        # Update task with max_iterations for progress tracking
        generation_tasks[task_id]["max_iterations"] = request.max_iterations
        generation_tasks[task_id]["current_iteration"] = 0
        
        # Create root node
        root = QuestionNode(
            question="",
            integrated_knowledge=set(),
            waiting_knowledge=request.knowledge_points,
            parent=None,
            difficulty_range=request.difficulty_range,
            question_type=request.question_type
        )
        
        # Initialize MCTS
        mcts = QuestionMCTS(
            exploration_weight=1.414,
            alpha=0.5,
            save_threshold=request.save_threshold,
            output_dir=None,  # Use default path
            difficulty_range=request.difficulty_range,
            question_type=request.question_type,
            need_optimize_threshold=7.5,
            use_rag=request.use_rag
        )
        
        logger.log(f"MCTS initialized. Output dir: {mcts.output_dir}")
        logger.log(f"Starting search: max_iterations={request.max_iterations}, target_leaf_nodes={request.target_leaf_nodes}")
        
        # Run search
        mcts.search(
            root=root,
            max_iterations=request.max_iterations,
            target_leaf_nodes=request.target_leaf_nodes
        )
        
        # Count generated questions
        output_files = []
        if os.path.exists(mcts.output_dir):
            output_files = [f for f in os.listdir(mcts.output_dir) if f.endswith('.json')]
        
        # Update task status
        generation_tasks[task_id].update({
            "status": "completed",
            "output_dir": mcts.output_dir,
            "generated_count": len(output_files),
            "end_time": datetime.now().isoformat()
        })
        
        logger.log(f"Generation completed. {len(output_files)} questions generated.")
        logger.log(f"Output directory: {mcts.output_dir}")
        
    except Exception as e:
        logger.log(f"Generation failed: {str(e)}")
        generation_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })
    finally:
        # Clean up logger reference
        if task_id in task_loggers:
            del task_loggers[task_id]


# ============== Main Entry ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
