"""
Knowledge Point Selector GUI

A simple GUI tool for selecting knowledge points with prerequisite dependency handling.
Uses tkinter for cross-platform compatibility.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Set


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
        self.root.geometry("600x500")
        
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
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
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
        
        self.result = sorted_selected
        self.root.quit()
        self.root.destroy()
    
    def run(self) -> List[str]:
        """
        Run the GUI and return selected knowledge points.
        
        Returns:
            List of selected knowledge points in topological order
        """
        self.root.mainloop()
        return getattr(self, 'result', [])


def select_knowledge(topology: Dict[str, List[str]] = None) -> List[str]:
    """
    Convenience function to run the knowledge selector.
    
    Args:
        topology: Optional custom knowledge topology
        
    Returns:
        List of selected knowledge points in topological order
        
    Example:
        >>> selected = select_knowledge()
        >>> print(selected)
        ['不定积分', '定积分', '重积分']
    """
    selector = KnowledgeSelector(topology)
    return selector.run()


if __name__ == "__main__":
    # Test run
    selected = select_knowledge()
    print(f"Selected knowledge points: {selected}")
