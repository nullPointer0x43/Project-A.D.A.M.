from typing import Union, Dict, Tuple, Literal, List, Any
from pydantic import BaseModel, Field

# =====================================================================
# CORE DATA STATES
# =====================================================================

# DATA ANALYSIS STRUCT
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


# GLOBAL DATA STATE
class GlobalState(BaseModel):
    current_info: Any = None
    current_data_info: DataAnalysis = DataAnalysis()
    log: str = ""
    user_input: str = ""


class SubgraphState(BaseModel):
    current_info: Any
    current_data_info: DataAnalysis = DataAnalysis()
    log: str = ""
    user_input: str = ""


class AnalystState(BaseModel):
    user_input: str = ""
    documents: List[str] = []
    route: str = ""
    code: str = ""
    result: str = ""
    final_summary: str = ""
    title: str = ""
    subdir_path: str = ""
    generated_files: List[str] = []
    errors: str = ""
    iterations: int = 0
    data_path: str = ""


# =====================================================================
# COLUMN PARSING NODE STRUCTS
# =====================================================================

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


# =====================================================================
# FILE TARGET NODE STRUCTS
# =====================================================================

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


class FileRouterPromptOutput(BaseModel):
    route: Literal["CONVO", "APPROVE", "FILE", "COL"] = Field(description="Which node to route to next", default="CONVO")


class FileParserOutput(BaseModel):
    analysisTargets: List[str] = Field(description="List of specific column names or variables to analyze", default=[])


# =====================================================================
# NULL ANALYSIS NODE STRUCTS
# =====================================================================

class NullCleanupInfo(BaseModel):
    nulls: list[Union[str, int]] = Field(
        default=[],
        description="List of values to treat as NULL."
    )

    nulls_report: list[Any] = Field(description="List of all the documents to add to vector db", default=[])


class NullRouterPromptOutput(BaseModel):
    route: Literal["CONVO", "APPROVE", "PARSE"] = Field(description="Which node to route to next", default="CONVO")


class NullParserOutput(BaseModel):
    nulls: List[Union[int, str]] = Field(description="List of all the custom nulls defined", default=[])


# =====================================================================
# AGENT PLATFORM / ROUTER STRUCTS
# =====================================================================

class MultiQuery(BaseModel):
    queries: List[str] = Field(default=[])


class RouterOutput(BaseModel):
    route: Literal["CODE", "RAG"]
    logic: str


class CodeOutput(BaseModel):
    chain_of_thought: str
    code: str