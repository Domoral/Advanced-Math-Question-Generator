import React, { useState, useMemo } from 'react';
import './KnowledgeSelector.css';

type SubCategory = Record<string, string[]>;

const CATEGORIES: Record<string, SubCategory> = {
  '高等数学': {
    '极限': ['极限定义', '洛必达法则', '泰勒公式', '麦克劳林公式', '等价无穷小替换', '夹逼准则', '单调有界准则', '海涅定理'],
    '微分中值定理': ['费马引理', '罗尔定理', '拉格朗日中值定理', '柯西中值定理', '泰勒中值定理'],
    '导数与微分': ['导数定义求导', '隐函数求导', '参数方程求导', '曲率公式'],
    '不定积分': ['牛顿-莱布尼茨公式', '换元积分法', '分部积分法', '有理函数积分', '点火公式', '区间再现公式', '积分中值定理'],
    '定积分应用': ['定积分几何应用', '旋转体体积公式', '弧长公式'],
    '常微分方程': ['一阶线性微分方程', '可分离变量微分方程', '齐次微分方程', '二阶常系数齐次微分方程', '二阶常系数非齐次微分方程', '欧拉方程'],
    '级数': ['比较判别法', '比值判别法', '根值判别法', '莱布尼茨判别法', '绝对收敛与条件收敛', '幂级数收敛半径', '函数展开为幂级数', '傅里叶级数'],
    '多元积分': ['二重积分极坐标变换', '三重积分柱坐标变换', '三重积分球坐标变换', '格林公式', '高斯公式', '斯托克斯公式', '曲线积分与路径无关条件'],
    '广义积分与特殊函数': ['柯西-施瓦茨不等式', '高斯积分', '伽马函数', '贝塔函数'],
  },
  '线性代数': {
    '行列式与矩阵': ['行列式展开定理', '克拉默法则', '矩阵求逆公式', '矩阵的秩的性质', '分块矩阵运算'],
    '向量与空间': ['施密特正交化', '向量组线性相关性判定', '向量组秩的判定'],
    '线性方程组': ['齐次方程组基础解系', '非齐次方程组解的结构'],
    '特征值与特征向量': ['特征值特征向量定义', '相似矩阵性质', '实对称矩阵性质'],
    '二次型': ['合同变换', '正定矩阵判定'],
    '矩阵分解与坐标变换': ['正交矩阵性质', '基变换与坐标变换'],
  },
  '概率论与数理统计': {
    '概率基础': ['古典概型', '几何概型', '条件概率公式', '乘法公式', '全概率公式', '贝叶斯公式'],
    '随机变量': ['分布函数法求密度', '卷积公式', '常见分布性质'],
    '数字特征': ['方差计算公式', '协方差计算公式', '相关系数公式'],
    '极限定理': ['切比雪夫不等式', '大数定律', '独立同分布中心极限定理'],
    '参数估计': ['矩估计法', '最大似然估计法', '无偏性有效性一致性', '区间估计'],
    '假设检验': ['假设检验步骤', '单正态总体均值方差检验', '双正态总体均值差方差比检验'],
  },
};

const QUESTION_TYPES = ['单选题', '多选题', '填空题', '计算题', '证明题', '应用题'];

export interface SelectionResult {
  knowledge_points: string[];
  difficulty_range: [number, number];
  question_type: string;
  use_rag: boolean;
}

interface KnowledgeSelectorProps {
  onConfirm?: (result: SelectionResult) => void;
  onCancel?: () => void;
}

const KnowledgeSelector: React.FC<KnowledgeSelectorProps> = ({ onConfirm, onCancel }) => {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState<string>(Object.keys(CATEGORIES)[0]);
  const [activeSubCategory, setActiveSubCategory] = useState<string>('');
  const [minDifficulty, setMinDifficulty] = useState<number>(0.3);
  const [maxDifficulty, setMaxDifficulty] = useState<number>(0.7);
  const [questionType, setQuestionType] = useState<string>('计算题');
  const [useRag, setUseRag] = useState<boolean>(true);

  // 初始化 activeSubCategory
  useMemo(() => {
    const subCategories = Object.keys(CATEGORIES[activeTab]);
    if (subCategories.length > 0 && !activeSubCategory) {
      setActiveSubCategory(subCategories[0]);
    }
  }, [activeTab]);

  // 获取所有知识点
  const allKnowledge = useMemo(() => {
    return Object.values(CATEGORIES).flatMap(subCats => 
      Object.values(subCats).flat()
    );
  }, []);

  // 获取当前大类的所有二级分类
  const currentSubCategories = useMemo(() => {
    return Object.keys(CATEGORIES[activeTab] || {});
  }, [activeTab]);

  // 处理复选框变化
  const handleCheckboxChange = (knowledge: string, checked: boolean) => {
    const newSelected = new Set(selected);
    if (checked) {
      newSelected.add(knowledge);
    } else {
      newSelected.delete(knowledge);
    }
    setSelected(newSelected);
  };

  // 全选当前子分类
  const selectCurrentSubCategory = () => {
    const newSelected = new Set(selected);
    CATEGORIES[activeTab][activeSubCategory].forEach(k => newSelected.add(k));
    setSelected(newSelected);
  };

  // 清空当前子分类
  const clearCurrentSubCategory = () => {
    const newSelected = new Set(selected);
    CATEGORIES[activeTab][activeSubCategory].forEach(k => newSelected.delete(k));
    setSelected(newSelected);
  };

  // 全选
  const selectAll = () => {
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
    const sortedSelected = Array.from(selected).sort((a, b) => {
      const aMainCat = Object.keys(CATEGORIES).find(mainCat => 
        Object.values(CATEGORIES[mainCat]).some(list => list.includes(a))
      ) || '';
      const bMainCat = Object.keys(CATEGORIES).find(mainCat => 
        Object.values(CATEGORIES[mainCat]).some(list => list.includes(b))
      ) || '';
      if (aMainCat !== bMainCat) return aMainCat.localeCompare(bMainCat);
      return a.localeCompare(b);
    });
    onConfirm?.({
      knowledge_points: sortedSelected,
      difficulty_range: [minDifficulty, maxDifficulty],
      question_type: questionType,
      use_rag: useRag,
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
        <h2>知识点选择器</h2>
        <p className="subtitle">选择需要综合的知识点</p>
      </div>

      {/* 一级分类标签页 */}
      <div className="tabs">
        {Object.keys(CATEGORIES).map((category) => (
          <button
            key={category}
            className={`tab ${activeTab === category ? 'active' : ''}`}
            onClick={() => {
              setActiveTab(category);
              setActiveSubCategory(Object.keys(CATEGORIES[category])[0]);
            }}
          >
            {category}
          </button>
        ))}
      </div>

      {/* 二级分类标签页 */}
      <div className="sub-tabs">
        {currentSubCategories.map((subCat) => (
          <button
            key={subCat}
            className={`sub-tab ${activeSubCategory === subCat ? 'active' : ''}`}
            onClick={() => setActiveSubCategory(subCat)}
          >
            {subCat}
          </button>
        ))}
      </div>

      {/* 全选/清空当前子分类按钮 */}
      <div className="sub-category-buttons">
        <button className="btn btn-small" onClick={selectCurrentSubCategory}>
          全选当前分类
        </button>
        <button className="btn btn-small" onClick={clearCurrentSubCategory}>
          清空当前分类
        </button>
      </div>

      {/* 知识点选择区域 */}
      <div className="knowledge-grid">
        {(CATEGORIES[activeTab][activeSubCategory] || []).map((knowledge: string) => (
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
        <h3>已选择的知识点</h3>
        <div className="selected-display">
          {selected.size > 0 ? (
            Array.from(selected).sort((a, b) => {
              const aMainCat = Object.keys(CATEGORIES).find(mainCat => 
                Object.values(CATEGORIES[mainCat]).some(list => list.includes(a))
              ) || '';
              const bMainCat = Object.keys(CATEGORIES).find(mainCat => 
                Object.values(CATEGORIES[mainCat]).some(list => list.includes(b))
              ) || '';
              if (aMainCat !== bMainCat) return aMainCat.localeCompare(bMainCat);
              return a.localeCompare(b);
            }).join(' → ')
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

      {/* 生成模式选择区域 */}
      <div className="generation-mode-section">
        <h3>生成模式</h3>
        <div className="radio-group">
          <label className="radio-label">
            <input
              type="radio"
              name="generationMode"
              checked={useRag}
              onChange={() => setUseRag(true)}
            />
            <span>RAG 增强（基于知识库检索）</span>
          </label>
          <label className="radio-label">
            <input
              type="radio"
              name="generationMode"
              checked={!useRag}
              onChange={() => setUseRag(false)}
            />
            <span>Pure LLM（纯大模型生成）</span>
          </label>
        </div>
        <div className="mode-display">当前模式: {useRag ? 'RAG 增强' : 'Pure LLM'}</div>
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
