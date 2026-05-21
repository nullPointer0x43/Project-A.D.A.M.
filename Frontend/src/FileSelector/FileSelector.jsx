import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, CheckCircle, Loader2, AlertCircle, Table, Columns, Target } from 'lucide-react';
import { fileService } from '../services';
import '../main.css';

const FileSelector = ({ isReady, data, threadId, socket }) => {
  const [localFile, setLocalFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const isLocked = data?.islocked || false;
  const backendFile = data?.file || null;
  const displayFileName = isLocked ? backendFile : localFile?.name;
  const hasDisplayData = data && (data.rows || data.cols || data.targets?.length > 0);

  const handleFileSelection = async (file) => {
    if (!file || !isReady || isLocked) return;

    setIsUploading(true);
    setError(null);

    try {
      await fileService.uploadDataset(file, threadId);
      setLocalFile(file);
    } catch (err) {
      setError(err.message);
      setLocalFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemoveFile = async (e) => {
    e.stopPropagation();
    try {
      if (!isLocked) {
        await fileService.resetDataset(threadId);
      }
      setLocalFile(null);
      setError(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setError("Failed to remove file.");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (!isReady || isUploading || isLocked || localFile) return;
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) handleFileSelection(e.dataTransfer.files[0]);
  };

  const getSubtitle = () => {
    if (error) return <span className="error-text"><AlertCircle size={14} /> {error}</span>;
    if (!isReady) return "AI initializing...";
    if (isLocked) return "Dataset verified and locked";
    if (localFile) return `${(localFile.size / (1024 * 1024)).toFixed(2)} MB • Ready`;
    return "CSV/XLSX only";
  };

  return (
    <div className={`total-fileselector-container ${!isReady ? 'pipeline-loading' : ''}`}>
      <div className="uploader-container">
        <motion.div
          className={`dropzone 
            ${isDragging ? 'dragging' : ''} 
            ${(localFile || isLocked) ? 'has-file' : ''} 
            ${isUploading ? 'uploading' : ''} 
            ${(!isReady || isLocked) ? 'disabled' : ''}`
          }
          onDragOver={(e) => { e.preventDefault(); !isLocked && !localFile && setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => !isUploading && !isLocked && !localFile && fileInputRef.current.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => handleFileSelection(e.target.files[0])}
            className="hidden-input"
            accept=".csv,.xlsx"
            disabled={!isReady || isUploading || isLocked}
          />

          {(localFile || isLocked) && !isUploading && (
            <motion.button 
              className="remove-btn"
              onClick={handleRemoveFile}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              whileHover={{ scale: 1.1, backgroundColor: '#fee2e2', color: '#ef4444' }}
              title="Remove file"
            >
              <X size={18} />
            </motion.button>
          )}

          <div className="icon-wrapper">
            <AnimatePresence mode="wait">
              {!isReady ? (
                <motion.div key="ready-load" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}>
                  <Loader2 className="icon muted animate-spin" />
                </motion.div>
              ) : isUploading ? (
                <motion.div key="loading" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}>
                  <Loader2 className="icon primary animate-spin" />
                </motion.div>
              ) : (localFile || isLocked) ? (
                <motion.div key="success" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}>
                  <CheckCircle className="icon success" />
                </motion.div>
              ) : (
                <motion.div key="upload" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}>
                  <Upload className="icon primary" />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="text-content">
            <h3 className="title">
              {!isReady ? "Connecting..." : isUploading ? "Processing..." : displayFileName || "Analyze Dataset"}
            </h3>
            <p className="subtitle">{getSubtitle()}</p>
          </div>
        </motion.div>

        <AnimatePresence>
          {hasDisplayData && (
            <motion.div 
              className="dashboard-grid"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              {data.rows && (
                <StatPanel icon={<Table size={16} />} label="Rows" value={data.rows.toLocaleString()} />
              )}
              {data.cols && (
                <StatPanel icon={<Columns size={16} />} label="Columns" value={data.cols.toLocaleString()} />
              )}
              {data.targets?.length > 0 && (
                <StatPanel 
                  icon={<Target size={16} />} 
                  label="Analysis Targets" 
                  value={data.targets.join(', ')} 
                  fullWidth 
                />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

const StatPanel = ({ icon, label, value, fullWidth = false }) => (
  <div className={`stats-panel ${fullWidth ? 'full-width' : ''}`}>
    <div className="panel-icon">{icon}</div>
    <div className="panel-info">
      <span className="label">{label}</span>
      <span className="value">{value}</span>
    </div>
  </div>
);

export default FileSelector;