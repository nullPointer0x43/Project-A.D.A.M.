import "../main.css";
import { useCallback, useMemo, useState } from "react";
import SideBarCard from "./Card/SideBarCard";
import { sessionService } from "../services";
import { 
    IconMenu2, 
    IconRefresh,
    IconFileUnknownFilled, 
    IconPlayerPlayFilled, 
    IconZoomQuestion, 
    IconChartBar, 
    IconChartScatter3d, 
    IconReportAnalytics,
    IconLoader2 
} from '@tabler/icons-react';

function SideBar({ progress, page, setPage, isDownloading }) {
    const [expanded, setExpanded] = useState(false);
    const [spacers, setSpacers] = useState([1, 0, 0, 0, 0]);
    const [spacerProgress, setSpacerProgress] = useState(["0%", "0%", "0%", "0%", "0%"]);
    const [completed, setCompleted] = useState([1, 0, 0, 0, 0, 0]);
    const [isResetting, setIsResetting] = useState(false);

    const sidebarWidth = expanded ? '200px' : '60px';
    const justification = expanded ? "flex-start" : "center";

    useMemo(() => {
        setSpacers(prevSpacers => {
            const newSpacers = new Array(prevSpacers.length).fill(0);
            const index = Math.min(Math.floor(progress / 100), prevSpacers.length);
            if (index >= 0 && index < newSpacers.length) {
                newSpacers[index] = 1;
            }
            return newSpacers;
        });

        setCompleted(() => {
            const newCompleted = new Array(6).fill(0);
            const index = Math.min(Math.floor(progress / 100), 5);
            for (let i = 0; i <= index; i++) {
                newCompleted[i] = 1;
            }
            return newCompleted;
        });

        setSpacerProgress(prevProgress => {
            const newProgress = new Array(prevProgress.length).fill("0%");
            const index = Math.min(Math.floor(progress / 100), newProgress.length - 1);
            
            let i = 0;
            for (; i < index; i++) {
                newProgress[i] = "100%";
            }
            newProgress[i] = `${progress % 100}%`;

            return newProgress;
        });
    }, [progress]);

    const handleResetWorkspace = async () => {
        const confirmReset = window.confirm("Are you sure you want to completely clear this run? This will erase all dataset steps.");
        if (!confirmReset) return;

        try {
            setIsResetting(true);
            await sessionService.resetPipeline();
            
            window.location.reload(); 
        } catch (error) {
            alert(`Failed to reset pipeline workspace: ${error.message}`);
        } finally {
            setIsResetting(false);
        }
    };

    return (
        <div className="sidebar-container" style={{ 
            '--sidebar-width': sidebarWidth,
            '--justification': justification 
        }}>
            <SideBarCard 
                icon={<IconMenu2 stroke={3} color="white" size={30}/>} 
                text="Menu" 
                expanded={expanded} 
                completed={1} 
                setter={setExpanded} 
                value={!expanded}
            />

            <SideBarCard 
                icon={isResetting ? <IconLoader2 className="spinner" size={30}/> : <IconRefresh stroke={3} color="white" size={30}/>} 
                text="Clear Workspace" 
                expanded={expanded} 
                completed={1} 
                setter={handleResetWorkspace} 
                value={null}
                disabled={isResetting}
            />

            <div className="sidebar-progress-container">
                <SideBarCard 
                    icon={<IconPlayerPlayFilled color="white" size={30}/>} 
                    text="Analysis Start" 
                    expanded={expanded} 
                    completed={completed[0]} 
                    setter={setPage} 
                    value={completed[0] ? 0 : page}
                />

                <div 
                    className={`spacer ${spacerProgress[0] === "0%" ? "" : "spacer-active"}`} 
                    style={{ '--progress': spacerProgress[0], flexGrow: spacers[0] ? 1 : 0, transition: 'flex-grow 1s' }} 
                />

                <SideBarCard 
                    icon={<IconFileUnknownFilled color="white" size={30}/>} 
                    text="Missingness Analysis" 
                    expanded={expanded} 
                    completed={completed[1]} 
                    setter={setPage} 
                    value={completed[1] ? 1 : page}
                />
                
                <div 
                    className={`spacer ${spacerProgress[1] === "0%" ? "" : "spacer-active"}`} 
                    style={{ '--progress': spacerProgress[1], flexGrow: spacers[1] ? 1 : 0, transition: 'flex-grow 1s' }} 
                />

                <SideBarCard 
                    icon={<IconChartBar color="white" size={30}/>} 
                    text="Univariate Analysis" 
                    expanded={expanded} 
                    completed={completed[2]} 
                    setter={setPage} 
                    value={completed[2] ? 2 : page}
                />

                <div 
                    className={`spacer ${spacerProgress[2] === "0%" ? "" : "spacer-active"}`} 
                    style={{ '--progress': spacerProgress[2], flexGrow: spacers[2] ? 1 : 0, transition: 'flex-grow 1s' }} 
                />

                <SideBarCard 
                    icon={<IconChartScatter3d color="white" size={30}/>} 
                    text="Multivariate Analysis" 
                    expanded={expanded} 
                    completed={completed[3]} 
                    setter={setPage} 
                    value={completed[3] ? 3 : page}
                />

                <div 
                    className={`spacer ${spacerProgress[3] === "0%" ? "" : "spacer-active"}`} 
                    style={{ '--progress': spacerProgress[3], flexGrow: spacers[3] ? 1 : 0, transition: 'flex-grow 1s' }} 
                />

                <SideBarCard 
                    icon={<IconZoomQuestion color="white" size={30}/>} 
                    text="Hypothesis Testing" 
                    expanded={expanded} 
                    completed={completed[4]} 
                    setter={setPage} 
                    value={completed[4] ? 4 : page}
                />

                <div 
                    className={`spacer ${spacerProgress[4] === "0%" ? "" : "spacer-active"}`} 
                    style={{ '--progress': spacerProgress[4], flexGrow: spacers[4] ? 1 : 0, transition: 'flex-grow 1s' }} 
                />

                <SideBarCard 
                    icon={isDownloading ? <IconLoader2 className="spinner" size={30}/> : <IconReportAnalytics color="white" size={30}/>} 
                    text={isDownloading ? "Downloading..." : "Final Report"} 
                    expanded={expanded} 
                    completed={completed[5]} 
                    setter={setPage} 
                    value={completed[5] ? 5 : page} 
                    disabled={isDownloading} 
                />
            </div>
        </div>
    );
}

export default SideBar;