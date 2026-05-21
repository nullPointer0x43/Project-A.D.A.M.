# **Project A.D.A.M. : Secure, Agentic Data Analysis & Automated Profiling**

## **1. Overview:**

Project A.D.A.M. is a production-grade, containerized agent based platform designed to automate end-to-end exploratory data analysis (EDA), data cleaning, and statistical profiling. 

By combining deterministic programmatic pipelines with LLM reasoning, the system takes raw, uncurated datasets and automatically generates comprehensive, insights-driven data reports with minimal human intervention.

## **2. Table of Contents:**
1. **[Overview](#1-overview)**
2. **[Table of Contents](#2-table-of-contents)**
3. **[Agentic Orchestration](#3-agentic-orchestration-the-langgraph-architecture)**
    - **[State Description](#31-state-of-langgraph)**
    - **[File and Targets Subgraph](#32-file-and-target-input-subgraph)**

## **3. Agentic Orchestration: The LangGraph Architecture**

### **3.1 State of LangGraph:**
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

---

### **3.2 File and Target Input Subgraph:**
#### **3.2.1 Aim:**
The aim of this node is to take the file path, the targets for analysis (just for marking, in case this is ever used in combination with an entire Auto-ML pipeline), and verify if the target columns exist within the specified file.

#### **3.2.2 State:**
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

#### **3.2.3 Graph Flow:**

As of now the the system only supports excel and .csv files. The file box in the front-end restricts user inputs to files of these formats. Upon receiving the file path, the system prompts for the analysis targets. 

Depending on the user reply the graph-flow is routed to either, the parser node for parsing if it is a data related response or the convo node for general questions and conversation about the state or otherwise. (The routing is done by setting the `log` variable)

The convo node leads back to the prompt node for reprompting.

The parser node takes a human language command and modifies the targets accordingly (E.g. "Add Var-A to the targets" -> New State contains Var-A in the targets list). Upon parsing the data is sent to the loader node for verification.

The loader node's job is to load the dataframe uploaded and verify if all of the specified targets exist. If not it is routed back to the prompt node, to ask for the targets.

If the loader node verifies that all targets exist within the analysis file, it then continues to the post-prompt node, in which the user is given a quick summary of the current state (targets + file uploaded). 

The user may decide to change any of the data, in which case it is redirected to the parser node for re-parsing and verification, or the user may question the agent based on the current state, in which case it is redirected to the post prompt node.

The graph redirects back to the post prompt node from the convo node instead of the prompt node if the data is already verified.

#### **3.2.4 Graph Visualization:**
```mermaid
graph TD
    %% Nodes & Shapes
    START([START])
    PromptNode[prompt_node]
    InputNode[input_node]
    RouterNode{router_node}
    ParseNode{parse_node}
    MainNode{main_node}
    ConvoNode{convo_node}
    PostPromptNode[post_prompt_node]
    END([END])

    %% Standard Edges
    START --> PromptNode
    PromptNode --> InputNode
    InputNode --> RouterNode
    PostPromptNode --> InputNode

    %% Conditional Edges from router_node
    RouterNode -->|PARSE| ParseNode
    RouterNode -->|CONVO| ConvoNode
    RouterNode -->|END| END

    %% Conditional Edges from convo_node
    ConvoNode -->|s.log == 'Post Prompted'| PostPromptNode
    ConvoNode -->|Else| PromptNode

    %% Conditional Edges from parse_node
    ParseNode -->|missing_fields == True| PromptNode
    ParseNode -->|Else| MainNode

    %% Conditional Edges from main_node
    MainNode -->|'Successfully' in s.log| PostPromptNode
    MainNode -->|Else| PromptNode

    %% Apply Styles Separately
    class START,END startEnd;
    class PromptNode,InputNode,PostPromptNode nodeStyle;
    class RouterNode,ParseNode,MainNode,ConvoNode choice;
```

---

### **3.3 Disguised Null Identification Subgraph:**
#### **3.3.1 Aim:**
The aim of this node is to search and replace disguised nulls (E.g. -999, 'nothing', etc with the `pandas` standard `null`). It has a default list of nulls as follows, and additionally prompts the user to supply any addition nulls that might be present in the data.

```python
self.null_values = [
    "None", "nan", "NaN", "NAN", "null", "NULL", "undefined", "Undefined",
    "n/a", "N/A", "na", "NA", "n.a.", "N.A.", "n/p", "not available", "not applicable",
    "?", "-", "--", "---", ".", "...", "missing", "Missing", "MISSING", "unknown", "Unknown",
    "", " ", "  ", "\t", "\n", "none",
    -1, -99, -999, -9999, 0, 999, 9999, "999", "9999", "0", "-1",
    "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "0000-00-00", "01-01-1970", "1900-01-01", "NaT", "nat"
]
```

#### **3.3.2 State:**
The `current_info` is changed to an object of type `PathInformation`

```python
class NullCleanupInfo(BaseModel):
    nulls: list[Union[str, int]] = Field(
        default=[],
        description="List of values to treat as NULL."
    )

    nulls_report: list[Any] = Field(description="List of all the documents to add to vector db", default=[])
```

* **nulls_report:** it is just a list of the `documents` / `texts` which is later added to the vector database for querying. Each of the documents is of the format:

```python
txt = (
    f"'{null}' identified as a disguised null, found {int(total_sum)} times in columns: "
    f"{', '.join([str(i) for i in affected_cols])}"
)
```


#### **3.3.3 Graph Flow + Visualisation:**
The graph is almost identical to the File and Targets Subgraph, the only difference being that the `main node` (loader node) is replaced with the `null node`.

The `null node` replaces the suspected null with the `pandas` standard `null`. It also records how many of each null is found and in which all columns, so that it may be stored and queried later.

---

### **3.4 Type Validation Subgraph:**
#### **3.4.1 Aim:**
This subgraph, determines the type and format of a column, given its first few entries, suspected type (whatever `pandas` loaded it as) and column name. It then decides the intended type and any formatting (E.g. suffixes, prefixes and formats for datetime)


#### **3.4.2 State:**
The `current_info` now takes the class of `ColParsingData`.

```python
class ColParsingData(BaseModel):
    col_name: str = Field(description="Column name", default="")
    df_path: str = Field(description="File path", default="")
    col_type: Literal["numeric", "category", "text", "datetime", "timedelta"] = Field(description="Inferred type of column",
                                                                         default="numeric")
    col_formatting: Dict[str, str] = Field(
        description="Formatting steps (prefix, suffix, or datetime-format)",
        default_factory=dict
    )
    col_data: List[Any] = Field(description="List of first 20 elements in Column", default=[0] * 20)
    col_error: str = Field(description="Error occurred during parsing", default="")
    iterations: int = Field(description="Number of iterations performed", default=0)
    max_iterations: int = Field(description="Maximum number of iterations performed", default=5)

```
All of the variable are described by their Field descriptions.

#### **3.4.3 Graph Flow:**
The Subgraph uses an llm to find the suspected type and formatting data of each column. It is a very small and simple subgraph which is visualised below, and is run for each and every column.


#### **3.4.4 Graph Visualization:**
```mermaid
graph TD
    %% Nodes & Shapes
    START([START])
    InferAndFix[infer_and_fix]
    ValidateLogic{validate_logic}
    END([END])

    %% Standard Edges
    START --> InferAndFix
    InferAndFix --> ValidateLogic

    %% Conditional Edges
    ValidateLogic -->|loop| InferAndFix
    ValidateLogic -->|done| END
```

---

### **3.5 Missingness Analysis Subgraph:**
#### **3.5.1 Aim:**
This subgraph, determines the type and severity of missingness of a column. Based on this data, it proposes the method of imputation. It then executes the imputation to fill in all the missing values.

#### **3.5.2 Theory:**
The decision of type of imputation is based on severity and type of missigness. There are 3 types of missingnesses:
1. **MCAR:** The probability of a data point being missing is entirely independent of both the observed data and the unobserved missing values themselves. It is pure, unbiased noise.
$$P(\text{Missing} \mid \text{Observed}, \text{Unobserved}) = P(\text{Missing})$$
2. **MAR:** The missingness is not random, but it can be completely explained by other observed variables in the dataset. The missing values do not depend on the missing values themselves, but on some other known columns.
$$P(\text{Missing} \mid \text{Observed}, \text{Unobserved}) = P(\text{Missing} \mid \text{Observed})$$
3. **MNAR**: The probability of missingness depends directly on the hypothetical value itself, or on unobserved factors. The reason it is missing is bound to the missing information.
$$P(\text{Missing} \mid \text{Observed}, \text{Unobserved}) \neq P(\text{Missing} \mid \text{Observed})$$

The imputations performed can be summarised in the following `3D matrix` of 
1. **Severity:** `[0, 5]` , `(5, 30]`, `(30, 60]`, `(60, 100]`
2. **Type of missingness:** `MAR`, `MAR`, `MNAR`
3. **Type of data:** `numeric`, `categoric`, `datetime`, `timedelta`

hence it is a `4 x 3 x 4` matrix.

* For `numeric` datatype the imputations are as follows:

|        | MCAR          | MAR                | MNAR          |
|--------|---------------|--------------------|---------------|
| 0-5    | Simple Median | Conditional Median | Simple Median |
| 5-30   | Simple Median | Conditional Median | Escalate      |
| 30-60  | MICE          | Regressor          | Escalate      |
| 60-100 | Drop          | Drop               | Drop          |

* For `categorical` datatype the imputations are as follows:

|        | MCAR        | MAR              | MNAR         |
|--------|-------------|------------------|--------------|
| 0-5    | Simple Mode | Conditional Mode | Constant UNK |
| 5-30   | Simple Mode | Conditional Mode | Escalate     |
| 30-60  | Regressor   | Regressor        | Escalate     |
| 60-100 | Drop        | Drop             | Drop         |

* For `datetime` datatype the imputations are as follows:

|        | MCAR                  | MAR                      | MNAR                  |
|--------|-----------------------|--------------------------|-----------------------|
| 0-5    | Constant Forward-fill | Conditional Forward-fill | Constant Forward-fill |
| 5-20   | Constant Forward-fill | Conditional Forward-fill | Escalate              |
| 20-60  | Simple Median         | Conditional Median       | Escalate              |
| 60-100 | Drop                  | Drop                     | Drop                  |

* For `timedelta` datatype the imputations are as follows:

|        | MCAR          | MAR                | MNAR          |
|--------|---------------|--------------------|---------------|
| 0-5    | Simple Median | Conditional Median | Simple Median |
| 5-20   | Simple Median | Conditional Median | Escalate      |
| 20-60  | KNN           | Conditional Median | Escalate      |
| 60-100 | Drop          | Drop               | Drop          |