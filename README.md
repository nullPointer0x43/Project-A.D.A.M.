# Project A.D.A.M. : Secure, Agentic Data Analysis & Automated Profiling

## 1. Overview:

Project A.D.A.M. is a production-grade, containerized agent based platform designed to automate end-to-end exploratory data analysis (EDA), data cleaning, and statistical profiling. 

By combining deterministic programmatic pipelines with LLM reasoning, the system takes raw, uncurated datasets and automatically generates comprehensive, insights-driven data reports with minimal human intervention.

## 2. Table of Contents:

## 3. Agentic Orchestration: The LangGraph Architecture

### 3.1 State of LangGraph:
LangGraph follows a state based architecture, where the only common data passed along the graph between the nodes exists in a data structure called the state of the graph.

In this program the state of the graph is defined as follows:

```python
class GlobalState(BaseModel):
    current_info: Any = None
    current_data_info: DataAnalysis = DataAnalysis()
    log: str = ""
    user_input: str = ""

class DataAnalysis(BaseModel):
    file_path: str = Field(description="Path of the dataframe", default=None)
    targets: list[str] = Field(description="List of specific columns or variables to analyze", default=[])
    nulls: list[Union[str, int]] = Field(
        default=[],
        description="List of custom values to treat as NULL."
    )
    type_description: Dict[str, Tuple[str, Dict[str, str]]] = Field(
        description="Column name: (Type of data + Formatting)",
        default={}
    )
    page_data: Dict[int, Any] = Field(description="Page datas", default={})

    current_progress: int = Field(description="Current progress", default=0)
```

All of the fields described in the classes are pretty self explanatory and are description by their Field descriptions.

* **Current Info:** This variable describes any structures and data needed specific to the subgraph, hence it is set as type Any. The data structure changes based on the subgraph.
* **Current Data Info:** This is the persistent data being worked on and is passed throughout the graph so that each subgraph can perform its operations on the data.
* **Log:** This contains error logs and are used for routing between nodes in each of the subgraphs.
* **User Input:** This contains the user input pulled from the frontend.
* **Page Data:** This is a dictionary of the data required by the frontend for each page, such that it can be supplied whenever user switches between pages.

### 3.2 File and Target Input Subgraph:
#### 3.2.1 Aim:
The aim of this node is to take the file path, the targets for analysis (just for marking, in case this is ever used in combination with an entire Auto-ML pipeline), and verify if the target columns exist within the specified file.

### 3.2.2 State:
The rest of the state remains the same, but the `current_info` is changed to an object of type `PathInformation`

```python
class PathInformation(BaseModel):
    file_path: str = Field(
        default="",
        description="The source/input data file path to be read. Keywords: 'load', 'read', 'from', 'input', 'analyze'."
    )
    all_columns: List[str] = Field(description="List of all columns in the dataset", default=[])
    rows: int = Field(description="Total number of rows in the dataset", default=0)
    analysis_targets: list[str] = Field(
        default=[],
        description="A list of specific column names or variables to focus on (e.g., ['SalePrice', 'Age'])."
    )
    missing_fields: list[Literal["file_path", "analysis_targets"]] = Field(
        default=["file_path", "analysis_targets"],
        description="A tracking list of which required fields are currently empty."
    )
```

### 3.2.3 Graph Flow:

As of now the the system only supports excel and .csv files. The file box in the front-end restricts user inputs to files of these formats. Upon receiving the file path, the system prompts for the analysis targets. 

Depending on the user reply the graph-flow is routed to either, the parser node for parsing if it is a data related response or the convo node for general questions and conversation about the state or otherwise. (The routing is done by setting the `log` variable)

The convo node leads back to the prompt node for reprompting.

The parser node takes a human language command and modifies the targets accordingly (E.g. "Add Var-A to the targets" -> New State contains Var-A in the targets list). Upon parsing the data is sent to the loader node for verification.

The loader node's job is to load the dataframe uploaded and verify if all of the specified targets exist. If not it is routed back to the prompt node, to ask for the targets.

If the loader node verifies that all targets exist within the analysis file, it then continues to the post-prompt node, in which the user is given a quick summary of the current state (targets + file uploaded). 

The user may decide to change any of the data, in which case it is redirected to the parser node for re-parsing and verification, or the user may question the agent based on the current state, in which case it is redirected to the post prompt node.

The graph redirects back to the post prompt node from the convo node instead of the prompt node if the data is already verified.

```mermaid
graph TD
    %% Styling Configuration
    classDef client fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef api fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef state fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef sandbox fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef db fill:#ede7f6,stroke:#5e35b1,stroke-width:2px;

    %% Graph Nodes
    User([User: Dataset Upload]) :::client
    UI[React Dashboard / Nginx] :::client
    FastAPI[FastAPI Backend Broker] :::api
    
    subgraph LangGraph State Machine Loop
        Validate[Node 1: Data Validation & Schema Profiling] :::state
        RouteIntent{Node 2: Intent Router <br/> Ollama / Local LLM} :::state
        CodeGen[Node 3: Script Generation <br/> Gemini API] :::state
        Execute[Node 4: Code Execution Pipeline] :::state
    end

    subgraph Isolated Security Sandbox
        Sandbox[Sandbox Runtime Container <br/> Pandas / NumPy / SciPy] :::sandbox
    end

    subgraph Data & State Storage Tier
        MinIO[(MinIO S3 Object Store <br/> Raw Data / Reports)] :::db
        Postgres[(Postgres DB <br/> Run Logs & Metadata)] :::db
        Chroma[(ChromaDB <br/> RAG Pattern Index)] :::db
    end

    %% Node Connections / Flow
    User -->|1. HTTP Post / Websocket| UI
    UI --> FastAPI
    FastAPI -->|2. Stream Stream Data| MinIO
    
    %% Graph Loop Ingestion
    FastAPI -->|3. Initialize Graph State| Validate
    Validate -->|Update Schema Metadata| Postgres
    Validate --> RouteIntent
    
    %% Multi-Agent Routing
    RouteIntent -->|RAG Context Retrieval| Chroma
    RouteIntent -->|Generate Python Analysis Script| CodeGen
    CodeGen --> Execute
    
    %% Sandbox Isolation Ring
    Execute -->|4. Dispatch Untrusted Code| Sandbox
    Sandbox -->|Compute Advanced Statistics <br/> ANOVA / Chi-Square| Sandbox
    Sandbox -->|5. Return Logs & Dataframe State| Execute
    
    %% Final Exit State
    Execute -->|6. Error Catching / Loop Exit| RouteIntent
    RouteIntent -->|7. Final Report Compilation| HTMLReport[Jinja2 HTML / PDF Report Engine]:::api
    HTMLReport -->|Push Assets| MinIO
    HTMLReport -->|8. Push Streaming Output Logs| UI
```
