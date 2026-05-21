import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Sparkles, User, Loader2, Database } from 'lucide-react';
import '../main.css';

const AgentChat = ({ messages, sendMessage, isReady, progress_message }) => {
  const [input, setInput] = useState('');
  const scrollRef = useRef(null);
  const isProcessing = progress_message !== "";

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isProcessing]);

  const handleSend = () => {
    if (!input.trim() || !isReady || isProcessing) return;
    sendMessage(input);
    setInput('');
  };

  return (
    <div className="agent-chat-container">
      <header className="agent-chat-header">
        <div className="header-logo">
          <Database size={22} />
          <span>Data Analyst AI</span>
        </div>
      </header>

      <div className="chat-messages" ref={scrollRef}>
        {!isReady && (
          <div className="loading-overlay">
            <Loader2 className="animate-spin" />
            <p>Initializing Pipeline...</p>
          </div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`message-row ${msg.role}`}
            >
              <div className="avatar">
                {msg.role === 'ai' ? <Sparkles size={14} /> : <User size={14} />}
              </div>
              <div className="message-content">{msg.text}</div>
            </motion.div>
          ))}

          {isProcessing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="message-row ai thinking-row"
            >
              <div className="avatar">
                <Sparkles size={14} />
              </div>
              <div className="message-content thinking-content">
                <Loader2 className="animate-spin" size={14} />
                <span>{progress_message}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="agent-chat-input-area">
        <div className={`input-box ${(!isReady || isProcessing) ? 'disabled-input' : ''}`}>
          <input
            type="text"
            placeholder={!isReady ? "Waiting for connection..." : "Ask about a hypothesis..."}
            value={input}
            disabled={!isReady || isProcessing}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          />
          <button
            className="send-trigger"
            onClick={handleSend}
            disabled={!input.trim() || !isReady || isProcessing}
          >
            {isProcessing || !isReady ? (
              <Loader2 className="animate-spin" size={18} />
            ) : (
              <Send size={18} />
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AgentChat;