import React, { useState } from 'react';
import KnowledgeSelector, { SelectionResult } from './components/KnowledgeSelector';
import './App.css';

// API 基础 URL
// 使用相对路径，通过 Nginx 代理到后端
const API_BASE_URL = '/api';

function App() {
  const [selectionResult, setSelectionResult] = useState<SelectionResult | null>(null);
  const [showSelector, setShowSelector] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [apiResponse, setApiResponse] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleConfirm = async (result: SelectionResult) => {
    setSelectionResult(result);
    setShowSelector(false);
    setIsLoading(true);
    setError(null);
    console.log('选择结果:', result);

    try {
      // 调用后端 API
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          knowledge_points: result.knowledge_points,
          difficulty_range: result.difficulty_range,
          question_type: result.question_type,
          use_rag: result.use_rag,
          max_iterations: 100,
          target_leaf_nodes: 4,
          save_threshold: 9.4,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setApiResponse(data);
      console.log('API 响应:', data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '请求失败');
      console.error('API 调用失败:', err);
    } finally {
      setIsLoading(false);
    }
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

                <div className="result-section">
                  <h3>生成模式</h3>
                  <p className="generation-mode">
                    {selectionResult.use_rag ? 'RAG 增强' : 'Pure LLM'}
                  </p>
                </div>

                {/* API 调用状态 */}
                {isLoading && (
                  <div className="result-section loading">
                    <h3>生成状态</h3>
                    <p>正在向后端发送请求...</p>
                  </div>
                )}

                {error && (
                  <div className="result-section error">
                    <h3>请求失败</h3>
                    <p>{error}</p>
                  </div>
                )}

                {apiResponse && (
                  <div className="result-section success">
                    <h3>后端响应</h3>
                    <p>状态: {apiResponse.status}</p>
                    <p>消息: {apiResponse.message}</p>
                  </div>
                )}

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
