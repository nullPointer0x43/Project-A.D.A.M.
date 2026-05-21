import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Sparkles, User, Loader2 } from 'lucide-react';
import '../main.css';

const SidebarChat = ({ messages, sendMessage, isProcessing, lockChat }) => {
  const [input, setInput] = useState('');
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isProcessing]);

  const handleSend = () => {
    if (!input.trim() || isProcessing) return;
    sendMessage(input);
    setInput('');
  };

  return (
    <div className={`sidebar-chat-container`}>
      <div className="chat-header">
        <div className="header-logo">
          <Sparkles size={18} />
          <span>AutoEDA Insights</span>
        </div>
      </div>

      <div className="chat-messages" ref={scrollRef}>
        
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
              <div className="avatar"><Sparkles size={14} /></div>
              <div className="message-content thinking-content">
                <Loader2 className="animate-spin" size={14} />
                <span>AI is thinking...</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="chat-input-area">
        <div className={`input-box ${(isProcessing || lockChat) ? 'disabled-input' : ''}`}>
          <input
            type="text"
            placeholder="Ask about a hypothesis..."
            value={input}
            disabled={isProcessing || lockChat}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          />
          <button 
            className="send-trigger" 
            onClick={handleSend} 
            disabled={!input.trim() || isProcessing || lockChat}
          >
            {(isProcessing) ? (
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

export default SidebarChat;