import React, { useState } from 'react';
import { insightService } from '../services';
import '../main.css';

const InsightExplorer = ({ data = [] }) => {
  const [activeIndex, setActiveIndex] = useState(null);
  const [details, setDetails] = useState({});
  const [loading, setLoading] = useState(false);

  const handleToggle = async (index, name) => {
    if (activeIndex === index) {
      setActiveIndex(null);
      return;
    }

    setActiveIndex(index);

    if (!details[name]) {
      setLoading(true);
      try {
        const result = await insightService.getInsightDetails(name);
        setDetails((prev) => ({ ...prev, [name]: result }));
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="insight-analysis-container">
      <div className="stat-summary-strip">
        <div className="stat-badge-item">
          <span className="stat-badge-label">Insights</span>
          <span className="stat-badge-value">{data.length}</span>
        </div>
        <div className="stat-badge-item">
          <span className="stat-badge-label">Mode</span>
          <span className="stat-badge-value highlight">AGENTIC RAG</span>
        </div>
      </div>

      {data.map((name, index) => {
        const item = details[name];
        const isActive = activeIndex === index;

        return (
          <div key={index} className={`row-card ${isActive ? 'active-card' : ''}`}>
            <div className="row-header" onClick={() => handleToggle(index, name)}>
              <div className="header-main">
                <div className="icon-box">{String(index + 1).padStart(2, '0')}</div>
                <div className="col-title">
                  <span className="type-badge">Statistical Model</span>
                  <h3>{name}</h3>
                </div>
              </div>
              <div>
                {isActive && loading ? (
                  <div className="spinner"></div>
                ) : (
                  <span className={`status-badge ${item ? 'status-sig' : 'status-insig'}`}>
                    {isActive ? 'VIEWING' : 'SELECT'}
                  </span>
                )}
              </div>
            </div>

            {isActive && item && (
              <div className="expanded-content">
                <section className="section-group">
                  <h4 className="popping-title">01. Context Retrieval</h4>
                  <div className="doc-grid">
                    {item.documents_retrieved?.map((doc, i) => (
                      <div key={i} className="source-row">
                        <span className="source-tag">SOURCE {i + 1}</span>
                        <span className="pre-text source-value">{doc}</span>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="section-group">
                  <h4 className="popping-title">02. Logic & Execution</h4>
                  <div className="code-layout">
                    <div className="visual-container">
                      <div className="visual-header">Python Logic</div>
                      <pre className="code-block">{item.code_generated}</pre>
                    </div>
                    <div className="visual-container">
                      <div className="visual-header">Console Output</div>
                      <pre className="output-block">{item.code_output}</pre>
                    </div>
                  </div>
                </section>

                {item.plots?.length > 0 && (
                  <section className="section-group">
                    <h4 className="popping-title">03. Visual Analysis</h4>
                    <div className="plots-gallery">
                      {item.plots.map((url, i) => (
                        <div key={i} className="plot-card">
                          <img src={url} alt={`Analysis ${i}`} className="analysis-plot" />
                          <div className="plot-caption">Figure {i + 1}: Generated Visualization</div>
                        </div>
                      ))}
                    </div>
                  </section>
                )}

                <section className="section-group">
                  <h4 className="popping-title">04. Final Inference</h4>
                  <div className="inference-text pre-text">{item.final_inference}</div>
                </section>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default InsightExplorer;