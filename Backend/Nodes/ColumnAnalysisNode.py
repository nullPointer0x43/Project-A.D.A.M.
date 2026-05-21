import numpy as np
import pandas as pd
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from tqdm import tqdm
from Nodes.DataStructs import DataAnalysis, ColParsingData
from Nodes.Prompts import column_inference_prompt

class SubgraphState(BaseModel):
    current_info: ColParsingData = ColParsingData()
    current_data_info: DataAnalysis = DataAnalysis()
    log: str = ""
    user_input: str = ""

class ColumnSmartParser:
    def __init__(self, llm, thread_id, minio_client):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(ColParsingData)
        self.minio_client = minio_client
        self.thread_id = thread_id

    def build(self):
        builder = StateGraph(SubgraphState)
        builder.add_node("infer_and_fix", self.infer_and_fix)
        builder.add_node("validate_logic", self.validate_logic)

        builder.set_entry_point("infer_and_fix")
        builder.add_edge("infer_and_fix", "validate_logic")

        builder.add_conditional_edges(
            "validate_logic",
            self.decide_path,
            {"loop": "infer_and_fix", "done": END}
        )
        return builder.compile()

    def infer_and_fix(self, state: SubgraphState):
        sample_data = state.current_info.col_data[:20]

        system_prompt = column_inference_prompt.format(
            col_name = state.current_info.col_name,
            col_data = sample_data,
            col_error = state.current_info.col_error if state.current_info.col_error else "None",
            data_schematic = state.current_info.model_json_schema()
        )

        res = self.structured_llm.invoke([
            ("system", system_prompt),
            ("human", f"Analyze column: {state.current_info.col_name}")
        ])

        state.current_info.col_type = res.col_type
        state.current_info.col_formatting = res.col_formatting
        return state

    def validate_logic(self, state: SubgraphState):
        df = self.minio_client.get_df(f"{self.thread_id}/TYPE_VALIDATION/Processed.parquet", self.thread_id)
        col_name = state.current_info.col_name
        col_type = state.current_info.col_type
        fmt = state.current_info.col_formatting
        try:
            series = df[col_name].astype(str).str.strip()

            if col_type == "numeric":
                s = series
                if fmt.get("prefix"):
                    s = s.str.replace(fmt["prefix"], "", regex=False)
                if fmt.get("suffix"):
                    s = s.str.replace(fmt["suffix"], "", regex=False)
                s = s.str.replace(r'[,$%]', '', regex=True)
                df[col_name] = pd.to_numeric(s, errors='raise')
                state.current_data_info.type_description[col_name] = (
                    state.current_info.col_type,
                    state.current_info.col_formatting
                )
            elif col_type == "datetime":
                target_format = fmt.get("datetime-format")
                try:
                    dt_series = pd.to_datetime(series, format=target_format, errors='raise')
                except Exception:
                    dt_series = pd.to_datetime(series, format='mixed', dayfirst=True)

                raw_strings = series.astype(str).str.strip()

                has_time = raw_strings.str.contains(r'\d{1,2}:\d{2}')
                has_day = raw_strings.str.contains(r'\d{1,2}[-/\.]|\d{1,2}\s+[A-Za-z]')
                has_month = raw_strings.str.contains(r'[A-Za-z]{3}|[-/\.]\d{1,2}[-/\.]')
                has_year = raw_strings.str.contains(r'\d{4}')

                type_descriptions = {}

                if target_format:
                    if any(token in target_format for token in ["%H", "%M"]):
                        df[f"{col_name}_mins_pd"] = np.where(
                            has_time,
                            (dt_series.dt.hour * 60) + dt_series.dt.minute,
                            np.nan
                        )
                        type_descriptions[f"{col_name}_mins_pd"] = ("datetime", {"datetime-format": "minutes_in_day"})

                    if "%d" in target_format:
                        df[f"{col_name}_day"] = np.where(has_day, dt_series.dt.day, np.nan)
                        type_descriptions[f"{col_name}_day"] = ("datetime", {"datetime-format": "%d"})

                    if any(tok in target_format for tok in ["%b", "%B", "%m"]):
                        df[f"{col_name}_month"] = np.where(has_month, dt_series.dt.month, np.nan)
                        type_descriptions[f"{col_name}_month"] = ("datetime", {"datetime-format": "%m"})

                    if any(tok in target_format for tok in ["%Y", "%y"]):
                        df[f"{col_name}_year"] = np.where(has_year, dt_series.dt.year, np.nan)
                        type_descriptions[f"{col_name}_year"] = ("datetime", {"datetime-format": "%Y"})

                    df.drop(columns=[col_name], inplace=True)
                else:
                    df[col_name] = dt_series
                    type_descriptions[col_name] = ("datetime", {"datetime-format": target_format})
                for key, desc in type_descriptions.items():
                    if col_name in state.current_data_info.targets:
                        state.current_data_info.targets.append(key)
                    state.current_data_info.type_description[key] = desc
                if col_name in state.current_data_info.targets:
                    state.current_data_info.targets.remove(col_name)
            elif col_type == "timedelta":
                s = series.str.replace(r'\b[hH]\b', 'hours', regex=True) \
                    .str.replace(r'\b[mM]\b', 'minutes', regex=True) \
                    .str.replace(r'\b[sS]\b', 'seconds', regex=True)
                s = s.str.replace(r'(\d)([a-zA-Z])', r'\1 \2', regex=True)
                s = s.str.replace(r'\b[hH]\b', 'hours', regex=True) \
                    .str.replace(r'\b[mM]\b', 'minutes', regex=True)

                td_series = pd.to_timedelta(s, errors='coerce')
                if td_series.isna().all():
                    raise ValueError(f"Could not parse duration from samples: {series.head(3).tolist()}")
                df[col_name] = td_series.dt.total_seconds() / 60.0
                state.current_info.col_type = "numeric"
                state.current_info.col_formatting["unit"] = "minutes"
                state.current_info.col_formatting["original_type"] = "timedelta"
                state.current_data_info.type_description[col_name] = (
                    state.current_info.col_type,
                    state.current_info.col_formatting
                )
            elif col_type in ["category", "text"]:
                df[col_name] = series
                state.current_data_info.type_description[col_name] = (
                    state.current_info.col_type,
                    state.current_info.col_formatting
                )

            state.current_info.col_error = ""

            self.minio_client.write_df(df, f"{self.thread_id}/TYPE_VALIDATION/Processed.parquet", self.thread_id, False)
        except Exception as e:
            error_msg = str(e)
            state.current_info.col_error = error_msg
            state.current_info.iterations += 1

        return state

    @staticmethod
    def decide_path(state: SubgraphState):
        if state.current_info.col_error == "":
            return "done"
        if state.current_info.iterations >= state.current_info.max_iterations:
            return "done"
        return "loop"


async def runColumnParsing(column_parser, config, manager, minio_client, add_doc, state:SubgraphState):
    thread_id = config["metadata"]["thread_id"]

    df = minio_client.get_df(f"{thread_id}/TYPE_VALIDATION/Processed.parquet", thread_id)
    columns = df.columns

    type_description = {}
    state.current_data_info.page_data[1]["col_analysis"] = []

    for column in tqdm(columns):
        snapshot = await column_parser.ainvoke({
            "current_info": {"col_name": column, "col_data": df[column].dropna().tolist()[:20]},
            "user_input": ""
        }, config)
        type_description_col = snapshot["current_data_info"].type_description
        data = []
        try:
            for key in type_description_col:
                data.append({
                    "display_name": key,
                    "data_type": type_description_col[key][0],
                    "formatting": type_description_col[key][1],
                    "analysis_status": "pending",
                    "imputation_status": "pending"
                })
                type_description[key] = type_description_col[key]
        except Exception:
            type_description[column] = df.dtypes[column]
            data.append({
                "display_name": column,
                "data_type": df.dtypes[column][0],
                "formatting": "No special Formatting",
                "analysis_status": "pending",
                "imputation_status": "pending"
            })
        state.current_data_info.page_data[1]["col_analysis"] += data

        await manager.push(state.current_data_info.page_data[1], thread_id, "dashboard_data")
        await manager.push(100 / len(columns) / 3, thread_id, "status_update")

    df = minio_client.get_df(f"{thread_id}/TYPE_VALIDATION/Processed.parquet", thread_id)
    minio_client.write_df(df, f"{thread_id}/TYPE_VALIDATION/Processed.parquet", thread_id)
    minio_client.write_df(df, f"{thread_id}/MISSINGNESS_IMPUTED/Processed.parquet", thread_id)
    minio_client.write_pickle(type_description, f"{thread_id}/TYPE_VALIDATION/Type_Description.p", thread_id)

    for col, info in type_description.items():
        result = ", ".join([f"{k}: {v}" for k, v in info[1].items()])
        txt = f"Column '{col}' is stored as {info[0]} with inferred format: {result}."
        add_doc(txt, {"source": "types", "column": col}, f"type_{col}")

    state.current_data_info.type_description = type_description
    return state
