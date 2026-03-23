"""
Knowledge Point Selector GUI

A simple GUI tool for selecting knowledge points with prerequisite dependency handling.
Uses tkinter for cross-platform compatibility.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Set, Tuple


# Default knowledge topology (prerequisite relationships)
DEFAULT_KNOWLEDGE_TOPOLOGY = {
    # 极限与连续
    "数列极限": [],
    "函数极限": ["数列极限"],
    "无穷小比较": ["函数极限"],
    "连续性": ["函数极限"],
    
    # 微分学
    "导数定义": ["函数极限"],
    "求导法则": ["导数定义"],
    "微分中值定理": ["导数定义", "连续性"],
    "泰勒展开": ["微分中值定理"],
    "函数性态分析": ["导数定义"],
    "洛必达法则": ["导数定义"],
    
    # 积分学
    "不定积分": ["求导法则"],
    "定积分": ["不定积分", "连续性"],
    "变限积分": ["定积分"],
    "反常积分": ["定积分"],
    "积分应用": ["定积分"],
    
    # 级数
    "数项级数": ["数列极限"],
    "幂级数": ["数项级数", "泰勒展开"],
    "傅里叶级数": ["定积分"],
    
    # 微分方程
    "一阶方程": ["不定积分"],
    "高阶线性方程": ["一阶方程", "导数定义"],
    
    # 多元函数
    "偏导数": ["导数定义"],
    "重积分": ["定积分", "偏导数"],
    "曲线积分": ["定积分", "偏导数"],
    "曲面积分": ["重积分", "曲线积分"],
}


class KnowledgeSelector:
    """
    GUI for selecting knowledge points with automatic prerequisite inclusion.
    """
    
    def __init__(self, topology: Dict[str, List[str]] = None):
        """
        Initialize the selector.
        
        Args:
            topology: Dictionary mapping knowledge points to their prerequisites
        """
        self.topology = topology or DEFAULT_KNOWLEDGE_TOPOLOGY
        self.selected: Set[str] = set()
        
        # Build reverse mapping (who depends on me)
        self.dependents: Dict[str, Set[str]] = {k: set() for k in self.topology}
        for k, prereqs in self.topology.items():
            for p in prereqs:
                if p in self.dependents:
                    self.dependents[p].add(k)
        
        self.root = tk.Tk()
        self.root.title("高数知识点选择器")
        self.root.geometry("700x800")
        
        # 难度区间设置 (默认 0.3-0.7，表示30%-70%的学生能做出)
        self.difficulty_min: float = 0.3
        self.difficulty_max: float = 0.7
        
        # 题型设置 (默认为计算题)
        self.question_type: str = "计算题"
        
        self._create_ui()
    
    def _create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="选择需要综合的知识点（前置依赖会自动添加）",
            font=("Arial", 12, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Organize knowledge points by category
        categories = {
            "极限与连续": ["数列极限", "函数极限", "无穷小比较", "连续性"],
            "微分学": ["导数定义", "求导法则", "微分中值定理", "泰勒展开", "函数性态分析", "洛必达法则"],
            "积分学": ["不定积分", "定积分", "变限积分", "反常积分", "积分应用"],
            "级数": ["数项级数", "幂级数", "傅里叶级数"],
            "微分方程": ["一阶方程", "高阶线性方程"],
            "多元函数": ["偏导数", "重积分", "曲线积分", "曲面积分"],
        }
        
        self.checkboxes: Dict[str, tk.BooleanVar] = {}
        
        for category, items in categories.items():
            frame = ttk.Frame(notebook, padding="10")
            notebook.add(frame, text=category)
            
            # Create checkboxes in a grid
            for i, item in enumerate(items):
                var = tk.BooleanVar(value=False)
                self.checkboxes[item] = var
                
                cb = ttk.Checkbutton(
                    frame, 
                    text=item,
                    variable=var,
                    command=lambda k=item: self._on_checkbox_changed(k)
                )
                cb.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=10, pady=5)
        
        # Selected display area
        selected_frame = ttk.LabelFrame(main_frame, text="已选择的知识点（含自动添加的前置）", padding="5")
        selected_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))
        selected_frame.columnconfigure(0, weight=1)
        
        self.selected_text = tk.Text(selected_frame, height=4, wrap=tk.WORD)
        self.selected_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        scrollbar = ttk.Scrollbar(selected_frame, orient=tk.HORIZONTAL, command=self.selected_text.xview)
        scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.selected_text.configure(xscrollcommand=scrollbar.set)
        
        # Difficulty setting area
        difficulty_frame = ttk.LabelFrame(main_frame, text="难度区间设置（能做出该题的学生比例）", padding="10")
        difficulty_frame.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))
        difficulty_frame.columnconfigure(0, weight=1)
        
        # Difficulty description
        difficulty_desc = ttk.Label(
            difficulty_frame, 
            text="难度值越小表示题目越难，越大表示题目越简单\n建议：难题 0.1-0.3 | 中档题 0.3-0.7 | 简单题 0.7-0.9",
            font=("Arial", 9),
            foreground="gray"
        )
        difficulty_desc.grid(row=0, column=0, columnspan=4, pady=(0, 10), sticky=tk.W)
        
        # Min difficulty
        ttk.Label(difficulty_frame, text="最小值（最难）:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.min_difficulty_var = tk.DoubleVar(value=0.3)
        self.min_difficulty_spin = ttk.Spinbox(
            difficulty_frame, 
            from_=0.0, 
            to=1.0, 
            increment=0.1,
            textvariable=self.min_difficulty_var,
            width=8,
            command=self._on_difficulty_changed
        )
        self.min_difficulty_spin.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Max difficulty
        ttk.Label(difficulty_frame, text="最大值（最简单）:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.max_difficulty_var = tk.DoubleVar(value=0.7)
        self.max_difficulty_spin = ttk.Spinbox(
            difficulty_frame, 
            from_=0.0, 
            to=1.0, 
            increment=0.1,
            textvariable=self.max_difficulty_var,
            width=8,
            command=self._on_difficulty_changed
        )
        self.max_difficulty_spin.grid(row=1, column=3, sticky=tk.W, padx=5)
        
        # Difficulty display
        self.difficulty_display = ttk.Label(
            difficulty_frame, 
            text="当前设置: 0.3 - 0.7（中档题）",
            font=("Arial", 10, "bold")
        )
        self.difficulty_display.grid(row=2, column=0, columnspan=4, pady=(10, 0), sticky=tk.W)
        
        # Question type setting area
        type_frame = ttk.LabelFrame(main_frame, text="题型选择", padding="10")
        type_frame.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))
        type_frame.columnconfigure(0, weight=1)
        
        # Question type description
        type_desc = ttk.Label(
            type_frame, 
            text="请选择生成的题目类型",
            font=("Arial", 9),
            foreground="gray"
        )
        type_desc.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # Question type options
        self.question_type_var = tk.StringVar(value="计算题")
        question_types = ["单选题", "多选题", "填空题", "计算题", "证明题", "应用题"]
        
        # Create a sub-frame for radio buttons to center them
        radio_frame = ttk.Frame(type_frame)
        radio_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        for i, qtype in enumerate(question_types):
            rb = ttk.Radiobutton(
                radio_frame,
                text=qtype,
                variable=self.question_type_var,
                value=qtype,
                command=self._on_question_type_changed
            )
            rb.grid(row=0, column=i, sticky=tk.W, padx=20, pady=5)
        
        # Question type display
        self.type_display = ttk.Label(
            type_frame, 
            text="当前题型: 计算题",
            font=("Arial", 10, "bold")
        )
        self.type_display.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, pady=10)
        
        ttk.Button(button_frame, text="全选", command=self._select_all).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="清空", command=self._clear_all).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="确认", command=self._confirm).grid(row=0, column=2, padx=5)
    
    def _get_all_prerequisites(self, knowledge: str, visited: Set[str] = None) -> Set[str]:
        """
        Get all prerequisites for a knowledge point (recursive).
        
        Args:
            knowledge: The knowledge point
            visited: Set of already visited nodes (for cycle detection)
            
        Returns:
            Set of all prerequisite knowledge points
        """
        if visited is None:
            visited = set()
        
        if knowledge in visited:
            return set()
        
        visited.add(knowledge)
        prereqs = set(self.topology.get(knowledge, []))
        
        # Recursively get prerequisites of prerequisites
        for prereq in list(prereqs):
            prereqs.update(self._get_all_prerequisites(prereq, visited))
        
        return prereqs
    
    def _update_selected(self):
        """Update the selected set and display."""
        self.selected.clear()
        
        # Get all checked items
        for k, var in self.checkboxes.items():
            if var.get():
                self.selected.add(k)
                # Add all prerequisites
                self.selected.update(self._get_all_prerequisites(k))
        
        # Update display
        self.selected_text.delete(1.0, tk.END)
        if self.selected:
            # Sort by topology order (roughly)
            sorted_selected = sorted(self.selected, 
                                   key=lambda x: list(self.topology.keys()).index(x) if x in self.topology else 999)
            self.selected_text.insert(1.0, " → ".join(sorted_selected))
    
    def _on_checkbox_changed(self, knowledge: str):
        """Handle checkbox state change."""
        self._update_selected()
    
    def _on_difficulty_changed(self):
        """Handle difficulty range change."""
        try:
            min_val = float(self.min_difficulty_var.get())
            max_val = float(self.max_difficulty_var.get())
            
            # Ensure min <= max
            if min_val > max_val:
                min_val, max_val = max_val, min_val
                self.min_difficulty_var.set(min_val)
                self.max_difficulty_var.set(max_val)
            
            self.difficulty_min = min_val
            self.difficulty_max = max_val
            
            # Update display text
            if max_val <= 0.3:
                level = "难题"
            elif min_val >= 0.7:
                level = "简单题"
            elif min_val >= 0.3 and max_val <= 0.7:
                level = "中档题"
            else:
                level = "混合难度"
            
            self.difficulty_display.config(text=f"当前设置: {min_val:.1f} - {max_val:.1f}（{level}）")
        except ValueError:
            pass
    
    def _on_question_type_changed(self):
        """Handle question type change."""
        self.question_type = self.question_type_var.get()
        self.type_display.config(text=f"当前题型: {self.question_type}")
    
    def _select_all(self):
        """Select all knowledge points."""
        for var in self.checkboxes.values():
            var.set(True)
        self._update_selected()
    
    def _clear_all(self):
        """Clear all selections."""
        for var in self.checkboxes.values():
            var.set(False)
        self._update_selected()
    
    def _confirm(self):
        """Confirm selection and close."""
        if not self.selected:
            messagebox.showwarning("警告", "请至少选择一个知识点！")
            return
        
        # Sort by topology order
        sorted_selected = sorted(self.selected, 
                               key=lambda x: list(self.topology.keys()).index(x) if x in self.topology else 999)
        
        # Update difficulty values
        self._on_difficulty_changed()
        
        # Update question type
        self._on_question_type_changed()
        
        # Return knowledge points, difficulty range and question type
        self.result = {
            'knowledge_points': sorted_selected,
            'difficulty_range': (self.difficulty_min, self.difficulty_max),
            'question_type': self.question_type
        }
        self.root.quit()
        self.root.destroy()
    
    def run(self) -> Dict:
        """
        Run the GUI and return selected knowledge points, difficulty range and question type.
        
        Returns:
            Dictionary containing:
            - 'knowledge_points': List of selected knowledge points in topological order
            - 'difficulty_range': Tuple of (min_difficulty, max_difficulty)
            - 'question_type': Selected question type
        """
        self.root.mainloop()
        return getattr(self, 'result', {'knowledge_points': [], 'difficulty_range': (0.3, 0.7), 'question_type': '计算题'})


def select_knowledge(topology: Dict[str, List[str]] = None) -> Dict:
    """
    Convenience function to run the knowledge selector.
    
    Args:
        topology: Optional custom knowledge topology
        
    Returns:
        Dictionary containing:
        - 'knowledge_points': List of selected knowledge points in topological order
        - 'difficulty_range': Tuple of (min_difficulty, max_difficulty)
        - 'question_type': Selected question type
        
    Example:
        >>> result = select_knowledge()
        >>> print(result['knowledge_points'])
        ['数列极限', '函数极限', '导数定义']
        >>> print(result['difficulty_range'])
        (0.3, 0.7)
        >>> print(result['question_type'])
        '计算题'
    """
    selector = KnowledgeSelector(topology)
    return selector.run()


if __name__ == "__main__":
    # Test run
    selected = select_knowledge()
    print(f"Selected: {selected}")
