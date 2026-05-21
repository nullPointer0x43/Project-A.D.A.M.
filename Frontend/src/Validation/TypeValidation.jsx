import React from 'react';
import { Loader2, CheckCircle2 } from 'lucide-react';
import "../main.css";

const DataValidationDashboard = ({ columnData = [] }) => {
  if (!Array.isArray(columnData)) return null;

  return (
    <div className="dv-dashboard-container">
      <div className="dv-table-wrapper">
        <table className="dv-table">
          <thead>
            <tr>
              <th className="dv-th">Column : Type</th>
              <th className="dv-th">Formatting</th>
              <th className="dv-th">Missingness (Sev/Type)</th>
              <th className="dv-th">Imputation Suggestion</th>
              <th className="dv-th" style={{ textAlign: 'center' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {columnData.map((col, index) => (
              <DataRow key={col.column_id || index} col={col} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const DataRow = ({ col }) => {
  const isAnalyzed = col.analysis_status === 'completed' || col.analysis_status === 'complete';
  const isImputed = col.imputation_status === 'completed' || col.imputation_status === 'complete';

  const formatSeverity = (sev) => {
    if (sev === null || sev === undefined || isNaN(sev)) return '0.00%';
    return `${parseFloat(sev).toFixed(2)}%`;
  };

  const getSeverityStyle = (sev) => {
    const s = parseFloat(sev);
    if (isNaN(s) || s <= 0) return { color: 'var(--chat-slate-500)' };
    if (s > 20) return { color: '#ef4444', fontWeight: 'bold' };
    if (s > 5) return { color: '#f59e0b', fontWeight: 'bold' };
    return { color: 'var(--chat-green-accent)', fontWeight: 'bold' };
  };

  const safeRender = (value, fallback = <span style={{ opacity: 0.3 }}>—</span>) => {
    if (value === null || value === undefined) return fallback;
    if (typeof value === 'object') {
      return value['datetime-format'] || value['unit'] || JSON.stringify(value);
    }
    return value;
  };

  return (
    <tr className="dv-row">
      <td className="dv-td">
        <span className="dv-col-name">{safeRender(col.display_name, 'Unknown')}</span>
        <code className="dv-data-type">{safeRender(col.data_type, 'any')}</code>
      </td>

      <td className="dv-td">
        <span style={{ fontSize: '0.875rem', color: 'var(--chat-slate-300)', whiteSpace: 'nowrap' }}>
          {safeRender(col.formatting)}
        </span>
      </td>

      <td className="dv-td">
        {!isAnalyzed ? (
          <div className="dv-shimmer" />
        ) : (
          <div style={{ fontSize: '0.875rem', whiteSpace: 'nowrap' }}>
            <span style={getSeverityStyle(col.missingness_severity)}>
              {formatSeverity(col.missingness_severity)}
            </span>
            <span style={{ color: 'var(--chat-green-mid)', margin: '0 0.5rem' }}>/</span>
            <span style={{ fontStyle: 'italic', color: 'var(--chat-slate-300)' }}>
              {safeRender(col.missingness_type, 'N/A')}
            </span>
          </div>
        )}
      </td>

      <td className="dv-td">
        {!isAnalyzed ? (
          <div className="dv-shimmer" style={{ width: '60px' }} />
        ) : (
          <span className="dv-badge-impute">
            {safeRender(col.imputation_suggested, 'None')}
          </span>
        )}
      </td>

      <td className="dv-td" style={{ textAlign: 'center' }}>
        {isImputed ? (
          <CheckCircle2 size={20} color="var(--chat-green-accent)" />
        ) : (
          <Loader2 size={18} className="dv-spin" />
        )}
      </td>
    </tr>
  );
};

export default DataValidationDashboard;