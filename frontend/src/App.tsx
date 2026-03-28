import React, { useState } from 'react';
import KnowledgeSelector, { SelectionResult } from './components/KnowledgeSelector';
import './App.css';

function App() {
  const [selectionResult, setSelectionResult] = useState<SelectionResult | null>(null);
  const [showSelector, setShowSelector] = useState<boolean>(true);

  const handleConfirm = (result: SelectionResult) => {
    setSelectionResult(result);
    setShowSelector(false);
    console.log('选择结果:', result);
  };

  const handleCancel = () => {
    setShowSelector(false);
  };

  const handleReset = () => {
    setSelectionResult(null);
    setShowSelector(true);
  };

  return (
    <div className="App">
      <div className="app-container">
        {showSelector ? (
          <KnowledgeSelector onConfirm={handleConfirm} onCancel={handleCancel} />
        ) : (
          <div className="result-container">
            <h1>选择结果</h1>
            {selectionResult ? (
              <div className="result-content">
                <div className="result-section">
                  <h3>已选择的知识点 ({selectionResult.knowledge_points.length}个)</h3>
                  <div className="knowledge-flow">
                    {selectionResult.knowledge_points.map((kp, index) => (
                      <span key={kp} className="knowledge-tag">
                        {kp}
                        {index < selectionResult.knowledge_points.length - 1 && (
                          <span className="arrow"> → </span>
                        )}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="result-section">
                  <h3>难度设置</h3>
                  <p>
                    区间: {selectionResult.difficulty_range[0].toFixed(1)} - {selectionResult.difficulty_range[1].toFixed(1)}
                    <span className="difficulty-badge">
                      {selectionResult.difficulty_range[1] <= 0.3
                        ? '难题'
                        : selectionResult.difficulty_range[0] >= 0.7
                        ? '简单题'
                        : selectionResult.difficulty_range[0] >= 0.3 && selectionResult.difficulty_range[1] <= 0.7
                        ? '中档题'
                        : '混合难度'}
                    </span>
                  </p>
                </div>

                <div className="result-section">
                  <h3>题型</h3>
                  <p className="question-type">{selectionResult.question_type}</p>
                </div>

                <button className="btn btn-primary" onClick={handleReset}>
                  重新选择
                </button>
              </div>
            ) : (
              <div className="no-selection">
                <p>未选择任何知识点</p>
                <button className="btn btn-primary" onClick={handleReset}>
                  返回选择
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
