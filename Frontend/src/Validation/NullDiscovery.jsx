import React from 'react';
import '../main.css';

const NullDashboard = ({ data }) => {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="null-dashboard-loading">
        <div className="spinner"></div>
        <p>Please Choose the nulls to search...</p>
      </div>
    );
  }

  const { search_config, discovery_results } = data;

  return (
    <div className="null-dashboard-container">
      <header className="null-dashboard-header">
        <h2>Null Value Analysis</h2>
      </header>

      <div className="null-dashboard-body">
        <div className="config-section">
          <div className="config-group">
            <label>Default Nulls Searched</label>
            <div className="tag-container">
              {search_config?.default_nulls?.length > 0 ? (
                search_config.default_nulls.map((val, i) => (
                  <span key={i} className="tag tag-default">{val}</span>
                ))
              ) : (
                <span className="fallback-text">No default nulls defined.</span>
              )}
            </div>
          </div>

          <div className="config-group">
            <label>User Custom Nulls</label>
            <div className="tag-container">
              {search_config?.custom_nulls?.length > 0 ? (
                search_config.custom_nulls.map((val, i) => (
                  <span key={i} className="tag tag-custom">{val}</span>
                ))
              ) : (
                <span className="fallback-text">No custom nulls defined.</span>
              )}
            </div>
          </div>
        </div>

        <div className="results-section">
          <label className="results-label">Discovered Anomalies</label>
          <div className="results-list">
            {discovery_results?.length > 0 ? (
              discovery_results.map((item, idx) => (
                <div key={idx} className="result-card">
                  <div className="result-main">
                    <span className="highlight">"{item.value}"</span> 
                    found <strong>{item.count}</strong> times
                  </div>
                  <div className="result-subtext">
                    Affected: {item.affected_columns?.join(', ') || 'N/A'}
                  </div>
                </div>
              ))
            ) : (
              <div className="empty-results">No null values discovered in the dataset.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NullDashboard;