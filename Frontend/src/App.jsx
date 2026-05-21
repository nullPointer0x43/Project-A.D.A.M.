import React, { useState, useEffect, useRef, useCallback } from 'react';
import { apiService } from './services';
import FileSelector from './FileSelector/FileSelector';
import Validation from './Validation/Validation';
import SideBar from './SideBar/SideBar';
import SidebarChat from './SideBarChat/SideBarChat';
import UnivariateAnalysisDashboard from './Univariate/Univariate';
import MultivariateDashboard from './Multivariate/Multivariate';
import Hypothesis from "./Hypothesis/Hypothesis"
import { IconLoader2 } from '@tabler/icons-react';
import "./main.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [isReady, setIsReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(true);
  const [dataState, setData] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [insights, setInsights] = useState([]);
  const [isDownloading, setIsDownloading] = useState(false);
  const [chatLock, setChatLock] = useState(true);

  const socket = useRef(null);

  const handlePageChange = useCallback(async (newPage) => {
    const page = Number(newPage);
    if (page === 5) {
      if (isDownloading) return;
      
      setIsDownloading(true);
      try {
        await apiService.downloadReport();
      } catch (err) {
        console.error("Report Download Failed:", err);
      } finally {
        setIsDownloading(false);
      }
    } else {
      setCurrentPage(page);
    }
  }, [isDownloading]);

  const sendMessage = useCallback((text) => {
    if (socket.current?.readyState === WebSocket.OPEN) {
      setMessages((prev) => [...prev, { role: 'user', text: text }]);
      socket.current.send(JSON.stringify({ content: {user_input: text}}));
      setIsProcessing(true);
    }
  }, []);

  useEffect(() => {
    socket.current = apiService.createWebSocket();

    socket.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case "llm_message":
          if (data.message.includes("SYSTEM_READY")) {
            setIsReady(true);
          } else {
            setMessages((prev) => [...prev, { role: 'ai', text: data.message }]);
            setIsProcessing(false);
          }
          break;
        case "dashboard_data":
          setData(data.message);
          break;
        case "dashboard_append":
          setData((prev) => {
            const currentState = prev && typeof prev === 'object' ? prev : {};
            return {
              ...currentState,
              ...data.message
            };
          });
          break;
        case "status_update":
          setProgress((prev) => prev + data.message);
          break;
        case "page_change":
          handlePageChange(data.message);
          break;
        case "process_update":
          setProgressMessage(data.message);
          break;
        case "insight_info":
          setInsights(data.message);
          break;
        case "relay":
          if (socket.current?.readyState === WebSocket.OPEN) {
            socket.current.send(JSON.stringify({ content: data.message}));
          }
          break;
        case "lock":
          setChatLock(true);
          break;
        case "unlock":
          setChatLock(false);
          setIsProcessing(false);
          break;
        default:
          break;
      }
    };

    return () => {
      if (socket.current) socket.current.close();
    };
  }, [handlePageChange]);

  useEffect(() => {
    if (currentPage === 5) return; 

    setData({});
    apiService.getPageData(currentPage)
      .then(response => {
        if (response) {
          setChatLock(!response.ready);
          setIsReady(prev => prev || response.ready);
          setData(response.page_data);
          setProgress(response.progress);
        }
      })
      .catch(err => console.error("Error updating page state:", err));
  }, [currentPage]);

  const renderContent = () => {
    switch (currentPage) {
      case 4:
        return (
          <Hypothesis 
            messages={messages} 
            sendMessage={sendMessage} 
            progressMessage={progressMessage}
            insights={insights} 
          />
        );
      case 3: return <MultivariateDashboard data={dataState}/>;
      case 2: return <UnivariateAnalysisDashboard data={dataState}/>;
      case 1: return <Validation data={dataState}/>;
      default: return <FileSelector isReady={isReady} data={dataState} threadId={"test_run_1"}/>;
    }
  };

  return (
    <div className='MainContainer'>
      {isDownloading && (
        <div className="download-overlay">
          <div className="download-modal">
            <IconLoader2 className="spinner-icon" size={48} />
            <h2>Generating Final Report</h2>
            <p>Please wait while we compile your statistical insights...</p>
          </div>
        </div>
      )}

      <div className='MainSideBarContainer' style={{ pointerEvents: isDownloading ? 'none' : 'auto' }}>
        <SideBar progress={progress} page={currentPage} setPage={handlePageChange}/>
      </div>
      
      <div className='MainContentContainer' style={{ opacity: isDownloading ? 0.5 : 1 }}>
        {renderContent()}
      </div>

      {currentPage < 4 && (
        <div className='MainSideBarChatContainer'>
          <SidebarChat 
            messages={messages} 
            sendMessage={sendMessage} 
            isProcessing={isProcessing || isDownloading} 
            lockChat={chatLock}
          />
        </div>
      )}
    </div>
  );
}

export default App;