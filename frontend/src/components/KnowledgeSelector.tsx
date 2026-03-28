import React, { useState, useMemo, useCallback } from 'react';
import './KnowledgeSelector.css';

// 知识点拓扑结构（前置依赖关系）
const DEFAULT_KNOWLEDGE_TOPOLOGY: Record<string, string[]> = {
  // 极限与连续
  '数列极限': [],
  '函数极限': ['数列极限'],
  '无穷小比较': ['函数极限'],
  '连续性': ['函数极限'],

  // 微分学
  '导数定义': ['函数极限'],
  '求导法则': ['导数定义'],
  '微分中值定理': ['导数定义', '连续性'],
  '泰勒展开': ['微分中值定理'],
  '函数性态分析': ['导数定义'],
  '洛必达法则': ['导数定义'],

  // 积分学
  '不定积分': ['求导法则'],
  '定积分': ['不定积分', '连续性'],
  '变限积分': ['定积分'],
  '反常积分': ['定积分'],
  '积分应用': ['定积分'],

  // 级数
  '数项级数': ['数列极限'],
  '幂级数': ['数项级数', '泰勒展开'],
  '傅里叶级数': ['定积分'],

  // 微分方程
  '一阶方程': ['不定积分'],
  '高阶线性方程': ['一阶方程', '导数定义'],

  // 多元函数
  '偏导数': ['导数定义'],
  '重积分': ['定积分', '偏导数'],
  '曲线积分': ['定积分', '偏导数'],
  '曲面积分': ['重积分', '曲线积分'],
};

// 知识点分类
const CATEGORIES: Record<string, string[]> = {
  '极限与连续': ['数列极限', '函数极限', '无穷小比较', '连续性'],
  '微分学': ['导数定义', '求导法则', '微分中值定理', '泰勒展开', '函数性态分析', '洛必达法则'],
  '积分学': ['不定积分', '定积分', '变限积分', '反常积分', '积分应用'],
  '级数': ['数项级数', '幂级数', '傅里叶级数'],
  '微分方程': ['一阶方程', '高阶线性方程'],
  '多元函数': ['偏导数', '重积分', '曲线积分', '曲面积分'],
};

const QUESTION_TYPES = ['单选题', '多选题', '填空题', '计算题', '证明题', '应用题'];

export interface SelectionResult {
  knowledge_points: string[];
  difficulty_range: [number, number];
  question_type: string;
}

interface KnowledgeSelectorProps {
  onConfirm?: (result: SelectionResult) => void;
  onCancel?: () => void;
}

const KnowledgeSelector: React.FC<KnowledgeSelectorProps> = ({ onConfirm, onCancel }) => {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState<string>(Object.keys(CATEGORIES)[0]);
  const [minDifficulty, setMinDifficulty] = useState<number>(0.3);
  const [maxDifficulty, setMaxDifficulty] = useState<number>(0.7);
  const [questionType, setQuestionType] = useState<string>('计算题');

  // 构建反向依赖映射（谁依赖我）
  const dependents = useMemo(() => {
    const deps: Record<string, Set<string>> = {};
    Object.keys(DEFAULT_KNOWLEDGE_TOPOLOGY).forEach((k) => {
      deps[k] = new Set();
    });
    Object.entries(DEFAULT_KNOWLEDGE_TOPOLOGY).forEach(([k, prereqs]) => {
      prereqs.forEach((p) => {
        if (deps[p]) deps[p].add(k);
      });
    });
    return deps;
  }, []);

  // 递归获取所有前置依赖
  const getAllPrerequisites = useCallback((knowledge: string, visited = new Set<string>()): Set<string> => {
    if (visited.has(knowledge)) return new Set();
    visited.add(knowledge);
    const prereqs = new Set(DEFAULT_KNOWLEDGE_TOPOLOGY[knowledge] || []);
    prereqs.forEach((prereq) => {
      getAllPrerequisites(prereq, visited).forEach((p) => prereqs.add(p));
    });
    return prereqs;
  }, []);

  // 处理复选框变化
  const handleCheckboxChange = (knowledge: string, checked: boolean) => {
    const newSelected = new Set(selected);
    if (checked) {
      newSelected.add(knowledge);
      // 自动添加所有前置依赖
      getAllPrerequisites(knowledge).forEach((p) => newSelected.add(p));
    } else {
      newSelected.delete(knowledge);
      // 检查是否需要移除其他知识点（因为它们可能依赖当前取消的知识点）
      const toCheck = [knowledge];
      while (toCheck.length > 0) {
        const current = toCheck.pop()!;
        dependents[current]?.forEach((dep) => {
          if (newSelected.has(dep)) {
            const prereqs = getAllPrerequisites(dep);
            prereqs.delete(current);
            // 如果依赖项的其他前置依赖都不在选中列表中，则移除
            const hasOtherPrereq = Array.from(prereqs).some((p) => newSelected.has(p));
            if (!hasOtherPrereq && !newSelected.has(dep)) {
              // 保留，因为它可能直接被选中
            }
          }
        });
      }
    }
    setSelected(newSelected);
  };

  // 全选
  const selectAll = () => {
    const allKnowledge = Object.keys(DEFAULT_KNOWLEDGE_TOPOLOGY);
    setSelected(new Set(allKnowledge));
  };

  // 清空
  const clearAll = () => {
    setSelected(new Set());
  };

  // 确认选择
  const handleConfirm = () => {
    if (selected.size === 0) {
      alert('请至少选择一个知识点！');
      return;
    }
    // 按拓扑顺序排序
    const sortedSelected = Array.from(selected).sort(
      (a, b) =>
        Object.keys(DEFAULT_KNOWLEDGE_TOPOLOGY).indexOf(a) -
        Object.keys(DEFAULT_KNOWLEDGE_TOPOLOGY).indexOf(b)
    );
    onConfirm?.({
      knowledge_points: sortedSelected,
      difficulty_range: [minDifficulty, maxDifficulty],
      question_type: questionType,
    });
  };

  // 获取难度等级描述
  const getDifficultyLevel = (): string => {
    if (maxDifficulty <= 0.3) return '难题';
    if (minDifficulty >= 0.7) return '简单题';
    if (minDifficulty >= 0.3 && maxDifficulty <= 0.7) return '中档题';
    return '混合难度';
  };

  // 处理难度变化
  const handleMinDifficultyChange = (value: number) => {
    const newMin = Math.min(value, maxDifficulty);
    setMinDifficulty(newMin);
  };

  const handleMaxDifficultyChange = (value: number) => {
    const newMax = Math.max(value, minDifficulty);
    setMaxDifficulty(newMax);
  };

  return (
    <div className="knowledge-selector">
      <div className="selector-header">
        <h2>高数知识点选择器</h2>
        <p className="subtitle">选择需要综合的知识点（前置依赖会自动添加）</p>
      </div>

      {/* 分类标签页 */}
      <div className="tabs">
        {Object.keys(CATEGORIES).map((category) => (
          <button
            key={category}
            className={`tab ${activeTab === category ? 'active' : ''}`}
            onClick={() => setActiveTab(category)}
          >
            {category}
          </button>
        ))}
      </div>

      {/* 知识点选择区域 */}
      <div className="knowledge-grid">
        {CATEGORIES[activeTab].map((knowledge) => (
          <label key={knowledge} className="checkbox-label">
            <input
              type="checkbox"
              checked={selected.has(knowledge)}
              onChange={(e) => handleCheckboxChange(knowledge, e.target.checked)}
            />
            <span>{knowledge}</span>
          </label>
        ))}
      </div>

      {/* 已选择显示区域 */}
      <div className="selected-section">
        <h3>已选择的知识点（含自动添加的前置）</h3>
        <div className="selected-display">
          {selected.size > 0 ? (
            Array.from(selected)
              .sort(
                (a, b) =>
                  Object.keys(DEFAULT_KNOWLEDGE_TOPOLOGY).indexOf(a) -
                  Object.keys(DEFAULT_KNOWLEDGE_TOPOLOGY).indexOf(b)
              )
              .join(' → ')
          ) : (
            <span className="placeholder">尚未选择任何知识点</span>
          )}
        </div>
      </div>

      {/* 难度设置区域 */}
      <div className="difficulty-section">
        <h3>难度区间设置（能做出该题的学生比例）</h3>
        <p className="hint">难度值越小表示题目越难，越大表示题目越简单</p>
        <p className="hint">建议：难题 0.1-0.3 | 中档题 0.3-0.7 | 简单题 0.7-0.9</p>
        <div className="difficulty-inputs">
          <div className="input-group">
            <label>最小值（最难）:</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.1}
              value={minDifficulty}
              onChange={(e) => handleMinDifficultyChange(parseFloat(e.target.value))}
            />
          </div>
          <div className="input-group">
            <label>最大值（最简单）:</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.1}
              value={maxDifficulty}
              onChange={(e) => handleMaxDifficultyChange(parseFloat(e.target.value))}
            />
          </div>
        </div>
        <div className="difficulty-display">
          当前设置: {minDifficulty.toFixed(1)} - {maxDifficulty.toFixed(1)}（{getDifficultyLevel()}）
        </div>
      </div>

      {/* 题型选择区域 */}
      <div className="question-type-section">
        <h3>题型选择</h3>
        <div className="radio-group">
          {QUESTION_TYPES.map((type) => (
            <label key={type} className="radio-label">
              <input
                type="radio"
                name="questionType"
                value={type}
                checked={questionType === type}
                onChange={(e) => setQuestionType(e.target.value)}
              />
              <span>{type}</span>
            </label>
          ))}
        </div>
        <div className="type-display">当前题型: {questionType}</div>
      </div>

      {/* 按钮区域 */}
      <div className="button-group">
        <button className="btn btn-secondary" onClick={selectAll}>
          全选
        </button>
        <button className="btn btn-secondary" onClick={clearAll}>
          清空
        </button>
        {onCancel && (
          <button className="btn btn-secondary" onClick={onCancel}>
            取消
          </button>
        )}
        <button className="btn btn-primary" onClick={handleConfirm}>
          确认
        </button>
      </div>
    </div>
  );
};

export default KnowledgeSelector;
