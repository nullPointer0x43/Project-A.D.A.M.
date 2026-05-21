import React, { useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, Tooltip, Cell, PieChart, Pie } from 'recharts';
import { ChevronDown, ChevronUp, BarChart3, Binary, Zap, Info, Loader2 } from 'lucide-react';
import '../main.css';

const COLORS = ['#10b981', '#059669', '#047857', '#065f46', '#064e3b'];

const UnivariateAnalysisDashboard = ({ data }) => {
  const [expandedKey, setExpandedKey] = useState(null);

  const handleToggle = (key) => {
    setExpandedKey(prevKey => (prevKey === key ? null : key));
  };

  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="analysis-loading-container" style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '300px',
        gap: '16px',
        color: '#64748b'
      }}>
        <Loader2 className="spinner" size={40} style={{
          color: '#10b981',
          animation: 'spin 1s linear infinite'
        }} />
        <p style={{ fontSize: '14px', fontWeight: '500' }}>
          Running univariate profiling and calculating metrics...
        </p>
      </div>
    );
  }

  return (
    <div className="analysis-container">
      <header className="dashboard-header">
        <h1>Data Univariate Analysis</h1>
        <p>Deep dive into feature distributions and ML readiness.</p>
      </header>

      {Object.entries(data).map(([colName, colData]) => (
        <AnalysisRow 
          key={colName} 
          name={colName} 
          stats={colData} 
          isOpen={expandedKey === colName} 
          onToggle={() => handleToggle(colName)}
        />
      ))}
    </div>
  );
};

const AnalysisRow = ({ name, stats, isOpen, onToggle }) => {
  const isNumeric = stats.type === 'numeric';

  return (
    <div className={`row-card ${isOpen ? 'is-open' : ''}`}>
      <div className="row-header" onClick={onToggle}>
        <div className="header-main">
          <div className="icon-wrapper">
            {isNumeric ? <BarChart3 size={20} className="numeric-icon" /> : <Binary size={20} className="cat-icon" />}
          </div>
          <div className="col-title">
            <h3>{name}</h3>
            <span className="type-badge">{stats.type}</span>
            {stats["is-id"] && <span className="id-badge">ID COLUMN</span>}
          </div>
        </div>
        
        <div className="header-preview">
          {!isOpen && (
            <>
              <StatMini label="Skew" value={stats.skew?.toFixed(2)} hide={!isNumeric} />
              <StatMini label="Imbalance" value={stats.imbalance_tier} hide={isNumeric} />
            </>
          )}
          {isOpen ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </div>
      </div>

      {isOpen && (
        <div className="expanded-grid">
          <div className="stats-section">
            <div className="stats-group">
              <h4><Info size={14} /> Statistical Summary</h4>
              <div className="internal-grid">
                {isNumeric ? (
                  <>
                    <StatRow label="Mean / Median" value={`${stats.mean} / ${stats.median}`} />
                    <StatRow label="Std Dev" value={stats.std?.toFixed(2)} />
                    <StatRow label="Skew / Kurtosis" value={`${stats.skew} / ${stats.kurtosis || 0}`} />
                    <StatRow label="IQR" value={stats.iqr} />
                    <StatRow label="Outliers (%)" value={`${((stats.Z_pc || 0) * 100).toFixed(1)}%`} />
                  </>
                ) : (
                  <>
                    <StatRow label="Cardinality" value={`${stats.cardinality} (${stats.cardinality_tier})`} />
                    <StatRow label="Entropy" value={stats.entropy} />
                    <StatRow label="Gini Index" value={stats.gini} />
                    <StatRow label="Imbalance" value={stats.imbalance_ratio} />
                  </>
                )}
              </div>
            </div>

            <div className={`ml-strategy-box ${stats.outlier_flag || stats.high_card_flag ? 'warning' : ''}`}>
              <div className="strategy-title"><Zap size={14} /> ML PIPELINE ACTION</div>
              <p>
                {isNumeric 
                  ? `Apply ${stats.transform} transformation. ${stats.outlier_flag ? 'Outlier robust scaling recommended.' : 'Standard scaling sufficient.'}` 
                  : `Use ${stats.encoding} encoding. Strategy: ${stats.handling}.`
                }
              </p>
              {stats.rare?.length > 0 && (
                <div className="sub-note">Rare categories: {stats.rare.join(', ')}</div>
              )}
            </div>
          </div>

          <div className="chart-container">
            <h4 className="chart-title">{isNumeric ? 'Distribution' : 'Category Split'}</h4>
            <ResponsiveContainer width="100%" height={200}>
              {isNumeric ? (
                <BarChart data={stats.chart_data}>
                  <XAxis dataKey="bin" fontSize={10} tick={{fill: '#64748b'}} axisLine={false} tickLine={false} />
                  <Tooltip 
                    cursor={{ fill: 'transparent' }}
                    contentStyle={{ backgroundColor: '#064e3b', border: '1px solid #10b981', color: '#fff' }}
                    itemStyle={{ color: '#10b981' }}
                  />
                  <Bar 
                    dataKey="count" 
                    fill="#10b981" 
                    radius={[4, 4, 0, 0]} 
                    stroke="#10b981"
                    strokeWidth={1}
                    activeBar={{
                      stroke: "#10b981",
                      strokeWidth: 2,
                      filter: 'drop-shadow(0 0 5px #10b981)',
                    }}
                  />
                </BarChart>
              ) : (
                <PieChart>
                  <Pie
                    data={stats.chart_data}
                    innerRadius={50}
                    outerRadius={70}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {stats.chart_data.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#064e3b', border: '1px solid #10b981', color: '#fff' }}
                    itemStyle={{ color: '#10b981' }}
                  />
                </PieChart>
              )}
            </ResponsiveContainer>
            
            {isNumeric && stats.percentiles && (
              <div className="percentile-strip">
                {Object.entries(stats.percentiles).map(([p, val]) => (
                  <div key={p} className="p-item">
                    <span className="p-label">{p}</span>
                    <span className="p-val">{val}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const StatMini = ({ label, value, hide }) => {
  if (hide || !value) return null;
  return (
    <div className="stat-mini" style={{ display: 'flex', gap: '4px' }}>
      <span style={{ color: 'var(--chat-slate-500)' }}>{label}:</span>
      <span style={{ color: 'var(--chat-green-accent)', fontWeight: 'bold' }}>{value}</span>
    </div>
  );
};

const StatRow = ({ label, value }) => (
  <div className="stat-line">
    <span className="stat-label">{label}</span>
    <span className="stat-value">{value}</span>
  </div>
);

export default UnivariateAnalysisDashboard;