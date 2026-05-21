import React, { useState, useEffect, useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label } from 'recharts';
import { multivariateService } from '../services';
import '../main.css';

const MultivariateCard = ({ pairId, data, isExpanded, onToggle }) => {
  const [plotData, setPlotData] = useState([]);
  const [loading, setLoading] = useState(false);

  const vars = useMemo(() => multivariateService.parsePairId(pairId), [pairId]);

  useEffect(() => {
    let isMounted = true;

    if (isExpanded && data.pair_type === 'Num-Num' && plotData.length === 0) {
      setLoading(true);
      multivariateService.fetchPlotData(vars[0], vars[1]).then((apiData) => {
        if (isMounted) {
          setPlotData(apiData);
          setLoading(false);
        }
      });
    }

    return () => { isMounted = false; };
  }, [isExpanded, data.pair_type, vars, plotData.length]);

  const renderPlot = () => {
    if (loading) {
      return (
        <div className="loading-container" style={{ height: '200px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
          <div className="spinner"></div>
          <span style={{ fontSize: '0.75rem', color: 'var(--chat-green-accent)', letterSpacing: '1px' }}>FETCHING DATA POINTS...</span>
        </div>
      );
    }

    if (data.pair_type === 'Num-Num') {
      return (
        <div style={{ width: '100%', height: 240, paddingBottom: '20px' }}>
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 30, left: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis type="number" dataKey="x" stroke="rgba(255,255,255,0.3)" fontSize={10}>
                <Label value={vars[0]} offset={-20} position="insideBottom" fill="var(--chat-slate-400)" style={{ fontSize: '0.7rem' }} />
              </XAxis>
              <YAxis type="number" dataKey="y" stroke="rgba(255,255,255,0.3)" fontSize={10}>
                <Label value={vars[1]} angle={-90} position="insideLeft" style={{ textAnchor: 'middle', fill: 'var(--chat-slate-400)', fontSize: '0.7rem' }} />
              </YAxis>
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }} 
                contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '4px' }} 
              />
              <Scatter name="Data" data={plotData} fill="var(--chat-green-accent)" fillOpacity={0.6} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      );
    }

    const effect = data.effect_val ?? 0;
    return (
      <div className="bar-visual">
        <h4 className="section-label" style={{ marginBottom: '1rem' }}>Effect Direction & Magnitude</h4>
        <div className="strength-track" style={{ display: 'flex', justifyContent: effect >= 0 ? 'flex-start' : 'flex-end', background: 'rgba(255,255,255,0.03)', height: '12px', borderRadius: '6px', overflow: 'hidden' }}>
          <div 
            className={`strength-bar ${data.rejected ? 'sig-bar' : 'insig-bar'}`} 
            style={{ width: `${Math.min(Math.abs(effect) * 100, 100)}%`, height: '100%', transition: 'width 0.5s ease-out', borderRadius: '6px' }}
          ></div>
        </div>
        <p className="visual-caption" style={{ textAlign: effect >= 0 ? 'left' : 'right', marginTop: '8px' }}>
            {effect < 0 ? 'Negative' : 'Positive'} Influence ({effect.toFixed(4)})
        </p>
      </div>
    );
  };

  if (!data) return null;

  return (
    <div className={`row-card ${isExpanded ? 'active-card' : ''}`}>
      <div className="row-header" onClick={onToggle}>
        <div className="header-main">
          <div className="icon-box">
            {data.pair_type === 'Num-Num' ? 'N²' : data.pair_type === 'Cat-Cat' ? 'C²' : 'NC'}
          </div>
          <div className="col-title">
            <div style={{ marginBottom: '4px' }}>
              <span className="type-badge">{data.pair_type ?? 'Unknown'}</span>
              <span className={`status-badge ${data.rejected ? 'status-sig' : 'status-insig'}`}>
                {data.rejected ? 'SIGNIFICANT' : 'INSIGNIFICANT'}
              </span>
            </div>
            <h3>{vars[0]} <span style={{color: 'var(--chat-green-base)'}}>×</span> {vars[1]}</h3>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '2rem', fontSize: '0.85rem', alignItems: 'center' }}>
          <div>
            <span style={{color: 'var(--chat-slate-500)'}}>Effect: </span>
            <span style={{color: 'var(--chat-white)', fontWeight: 600}}>{(data.effect_val ?? 0).toFixed(3)}</span>
          </div>
          <div style={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0)', transition: '0.3s ease', color: 'var(--chat-green-accent)' }}>▼</div>
        </div>
      </div>

      {isExpanded && (
        <div className="expanded-content">
          <div className="expanded-grid">
            <div className="stats-column">
              <h4 className="section-label">Statistical Metrics</h4>
              <div className="stat-line"><span className="stat-label">Method</span><span className="stat-value">{data.test_used ?? 'N/A'}</span></div>
              <div className="stat-line">
                <span className="stat-label">p-value</span>
                <span className="stat-value">
                  {data.p_adj != null ? (data.p_adj < 0.001 ? data.p_adj.toExponential(2) : data.p_adj.toFixed(4)) : "N/A"}
                </span>
              </div>
              <div className="stat-line"><span className="stat-label">Strength</span><span className="stat-value mag-highlight">{data.magnitude ?? 'N/A'}</span></div>
            </div>
            <div className="insight-column">
              <h4 className="section-label" style={{color: 'var(--chat-white)', opacity: 0.8}}>Automated Insight</h4>
              <p className="insight-text">
                Analysis reveals a <strong>{(data.magnitude ?? 'N/A').toLowerCase()}</strong> association. 
                The evidence <strong>{data.rejected ? 'strongly suggests' : 'does not provide sufficient evidence'}</strong> a relationship exists between these features.
              </p>
            </div>
          </div>
          <div className="plot-section">{renderPlot()}</div>
        </div>
      )}
    </div>
  );
};

const MultivariateDashboard = ({ data = {} }) => {
  const [activeTab, setActiveTab] = useState("All");
  const [showOnlySig, setShowOnlySig] = useState(false);
  const [search, setSearch] = useState("");
  const [expandedId, setExpandedId] = useState(null);

  const safeData = useMemo(() => data || {}, [data]);

  const stats = useMemo(() => {
    const all = Object.values(safeData);
    if (all.length === 0) return { totalPairs: 0, sigPairs: 0 };
    return {
      totalPairs: all.length,
      sigPairs: all.filter(v => v?.rejected).length
    };
  }, [safeData]);

  const filteredData = useMemo(() => {
    return Object.entries(safeData).filter(([key, val]) => {
      if (!val) return false;
      const matchesTab = activeTab === "All" || val.pair_type === activeTab;
      const matchesSig = showOnlySig ? val.rejected : true;
      const matchesSearch = key.toLowerCase().includes(search.toLowerCase());
      return matchesTab && matchesSig && matchesSearch;
    });
  }, [safeData, activeTab, showOnlySig, search]);
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="analysis-container">
        <header className="dashboard-header">
          <h1>Multivariate Analysis</h1>
          <p>Statistical interaction summary and feature correlation</p>
        </header>
        
        {/* Render a styled processing spinner matches your App.js design patterns */}
        <div 
          className="processing-state-container" 
          style={{ 
            padding: '6rem 2rem', 
            textAlign: 'center', 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center', 
            gap: '1.5rem' 
          }}
        >
          {/* Reuses your app's standard spinner styling rules */}
          <div className="spinner" style={{ width: '40px', height: '40px' }}></div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--chat-green-accent)', fontWeight: 600, letterSpacing: '1.5px', uppercase: 'true' }}>
              COMPUTING MULTIVARIATE INTERACTIONS
            </span>
            <p style={{ fontSize: '0.75rem', color: 'var(--chat-slate-500)', margin: 0 }}>
              Running matrix statistical tests. Data cards will appear dynamically as they compile...
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="analysis-container">
      <header className="dashboard-header">
        <h1>Multivariate Analysis</h1>
        <p>Statistical interaction summary and feature correlation</p>
      </header>

      <div className="stat-summary-strip">
        <div className="stat-badge-item">
          <span className="stat-badge-label">Total Combinations</span>
          <span className="stat-badge-value">{stats.totalPairs}</span>
        </div>
        <div className="stat-badge-item">
          <span className="stat-badge-label">Significant Hits</span>
          <span className="stat-badge-value highlight">{stats.sigPairs}</span>
        </div>
        <div className="stat-badge-item">
          <span className="stat-badge-label">Filtered</span>
          <span className="stat-badge-value">{filteredData.length}</span>
        </div>
      </div>

      <div className="controls-bar">
        <div className="filter-group">
          {["All", "Cat-Cat", "Num-Num", "Num-Cat"].map(tab => (
            <button 
                key={tab} 
                className={`filter-btn ${activeTab === tab ? 'active' : ''}`} 
                onClick={() => setActiveTab(tab)}
            >
                {tab}
            </button>
          ))}
        </div>
        <input className="search-bar" placeholder="Search features..." value={search} onChange={(e) => setSearch(e.target.value)} />
        <label className="sig-toggle">
          <input type="checkbox" checked={showOnlySig} className="custom-checkbox" onChange={() => setShowOnlySig(!showOnlySig)} />
          Significant Only
        </label>
      </div>

      <div className="cards-list">
        {filteredData.length > 0 ? (
          filteredData.map(([id, stats]) => (
            <MultivariateCard key={id} pairId={id} data={stats} isExpanded={expandedId === id} onToggle={() => setExpandedId(expandedId === id ? null : id)} />
          ))
        ) : (
          <div className="empty-state">No pairs match your current filter criteria.</div>
        )}
      </div>
    </div>
  );
};

export default MultivariateDashboard;