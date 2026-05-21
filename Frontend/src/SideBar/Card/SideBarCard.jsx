import "./SideBarCard.css"

function SideBarCard({ icon, text, expanded, value, completed, setter }) {
    return (
        <div 
            className={`sidebarcard-container ${(completed) ? "completed" : ""}`}
            onClick={() => setter(value)} 
            style={{"--card-width": expanded ? '170px' : '30px'}}
        >
            <div style={{ flexShrink: 0 }}>
                {icon}
            </div>
            <label className="icon-label" style={{ opacity: expanded ? 1 : 0, transition: 'opacity 0.2s' }}>
                {text}
            </label>
        </div>
    );
}

export default SideBarCard;