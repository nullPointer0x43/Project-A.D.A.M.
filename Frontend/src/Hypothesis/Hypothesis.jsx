import AgentChat from "./chatAgent";
import InsightExplorer from "./insightExplorer"
import "../main.css"

const Hypothesis = ({ messages, sendMessage, progressMessage, insights }) => {
    return (
        <div className='hypothesis-container'>
            <AgentChat 
                messages={messages} 
                sendMessage={sendMessage} 
                isReady={true} 
                progress_message={progressMessage} 
            />
            <InsightExplorer data={insights} />
        </div>
    );
}

export default Hypothesis;