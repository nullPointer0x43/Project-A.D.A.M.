import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from Nodes.DataStructs import SubgraphState

def MCAR_tests(col, alpha, df, type_description):
    missingDistribution = df[df[col].isnull()]
    notMissingDistribution = df[~df[col].isnull()]
    for col1 in df.columns:
        md_vals = missingDistribution[col1].dropna()
        nmd_vals = notMissingDistribution[col1].dropna()
        if col1 == col or len(md_vals) < 5 or len(nmd_vals) < 5:
            continue
        dtype = type_description[col1][0]
        if dtype == "numeric":
            score = ks_2samp(md_vals, nmd_vals)[1]
        elif dtype == "categorical":
            crosstab = pd.crosstab(df[col].isnull(), df[col1])
            score = chi2_contingency(crosstab)[1] if crosstab.shape[1] > 1 else 1.0
        else:
            score = 1.0
        if score < alpha: return False
    return True

def MAR_tests(target_col, df, type_description, threshold=0.65):
    num = [i for i in df.columns if type_description[i][0] in ["numeric", "datetime", "timedelta"]]
    cat = [i for i in df.columns if type_description[i][0] in ["category"]]
    mask = df[target_col].isnull().astype(int)
    df = df.drop(columns=[target_col])
    if type_description[target_col][0] in ["numeric", "datetime", "timedelta"]: num.remove(target_col)
    elif type_description[target_col][0] == "category": cat.remove(target_col)
    preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), num),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat)
    ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(f_classif, k=min(10, len(df.columns)))),
        ('clf', HistGradientBoostingClassifier(max_iter=100))
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scores = cross_val_score(model, df, mask, scoring='roc_auc', cv=2)
    return bool(np.mean(scores) > threshold)

def colAnalysisHelper(col, df, type_description):
    col_metadata = {
        "col": col, "dtype": "", "mechanism": "", "severity": 0, "impute": False,
        "impute_type": "", "impute_value": "", "flag": False, "escalate": False, "drop": False
    }
    severity = df[col].isnull().sum() / len(df) * 100
    dtype = type_description[col][0]
    mechanism = "MCAR" if MCAR_tests(col, 0.05, df, type_description) else \
        ("MAR" if dtype != "text" and MAR_tests(col, df, type_description) else "MNAR")
    col_metadata["dtype"] = dtype
    col_metadata["mechanism"] = mechanism
    col_metadata["severity"] = severity.item()
    if dtype == "numeric":
        if mechanism in ["MCAR", "MAR"]:
            if severity < 60:
                col_metadata["impute"] = True
                col_metadata["impute_type"] = "Simple Statistical" if (severity < 30 and mechanism == "MCAR") else ("Conditional Statistical" if severity < 30 else "Model Based")
                col_metadata["impute_value"] = "Median" if severity < 30 else ("MICE" if mechanism == "MCAR" else "Regressor")
                col_metadata["flag"] = True
            else:
                col_metadata["drop"] = True
        else:
            if severity < 5:
                col_metadata["impute"] = True
                col_metadata["impute_type"] = "Simple Statistical"
                col_metadata["impute_value"] = "Median"
                col_metadata["flag"] = True
            elif severity < 60: col_metadata["escalate"] = True
            else: col_metadata["drop"] = True
    elif dtype == "categorical":
        if mechanism in ["MCAR", "MAR"]:
            if severity < 60:
                col_metadata["impute"] = True
                col_metadata["impute_type"] = "Simple Statistical" if (severity < 30 and mechanism == "MCAR") else ("Conditional Statistical" if severity < 30 else "Model Based")
                col_metadata["impute_value"] = "Mode" if severity < 30 else "Regressor"
                col_metadata["flag"] = True
            else:
                col_metadata["drop"] = True
        else:
            if severity < 5:
                col_metadata["impute"] = True
                col_metadata["impute_type"] = "Constant"
                col_metadata["impute_value"] = "Unknown"
                col_metadata["flag"] = True
            elif severity < 60: col_metadata["escalate"] = True
            else: col_metadata["drop"] = True
    elif dtype == "datetime":
        if mechanism in ["MCAR", "MAR"]:
            if severity < 20:
                col_metadata["impute"] = True
                col_metadata["impute_type"] = "Constant" if mechanism == "MCAR" else "Conditional Statistical"
                col_metadata["impute_value"] = "ForwardFill"
                col_metadata["flag"] = True
            elif severity < 60:
                col_metadata["impute"] = True
                col_metadata["impute_type"] = "Simple Statistical" if mechanism == "MCAR" else "Conditional Statistical"
                col_metadata["impute_value"] = "Median"
                col_metadata["flag"] = True
            else:
                col_metadata["drop"] = True
        else:
            if severity < 5:
                col_metadata["impute"] = True
                col_metadata["impute_type"] = "Constant"
                col_metadata["impute_value"] = "ForwardFill"
                col_metadata["flag"] = True
            elif severity < 60: col_metadata["escalate"] = True
            else: col_metadata["drop"] = True
    elif dtype == "timedelta":
        if mechanism in ["MCAR", "MAR"]:
            if severity < 20:
                col_metadata["impute"] = True
                col_metadata["impute_type"] = "Simple Statistical" if mechanism == "MCAR" else "Conditional Statistical"
                col_metadata["impute_value"] = "Median"
                col_metadata["flag"] = True
            elif severity < 60:
                col_metadata["impute"] = True
                col_metadata["impute_type"] = "Model Based"
                col_metadata["impute_value"] = "KNN" if mechanism == "MCAR" else "Regressor"
                col_metadata["flag"] = True
            else:
                col_metadata["drop"] = True
        else:
            if severity < 5:
                col_metadata["impute"] = True
                col_metadata["impute_type"] = "Simple Statistical"
                col_metadata["impute_value"] = "Median"
                col_metadata["flag"] = True
            elif severity < 60: col_metadata["escalate"] = True
            else: col_metadata["drop"] = True
    else:
        if severity > 60: col_metadata["drop"] = True
        else:
            col_metadata["impute"] = True
            col_metadata["impute_type"] = "Constant"
            col_metadata["impute_value"] = "Unknown"
            col_metadata["flag"] = True
    return col_metadata

async def colAnalysis(df, type_description, page_data, manager, thread_id):
    page_data_map = {page_data["col_analysis"][i]["display_name"]: i for i in range(len(page_data["col_analysis"]))}
    col_report = []
    for col in tqdm(df.columns):
        if df[col].isnull().sum() > 0:
            col_report.append(colAnalysisHelper(col, df, type_description))
            idx = page_data_map[col]
            page_data["col_analysis"][idx]["missingness_severity"] = col_report[-1]["severity"]
            page_data["col_analysis"][idx]["missingness_type"] = col_report[-1]["mechanism"]
            page_data["col_analysis"][idx]["imputation_suggested"] = col_report[-1]["impute_type"]
            page_data["col_analysis"][idx]["analysis_status"] = "complete"
        else:
            idx = page_data_map[col]
            page_data["col_analysis"][idx]["missingness_severity"] = 0
            page_data["col_analysis"][idx]["missingness_type"] = "-"
            page_data["col_analysis"][idx]["imputation_suggested"] = "None"
            page_data["col_analysis"][idx]["analysis_status"] = "complete"
            page_data["col_analysis"][idx]["imputation_status"] = "complete"
        await manager.push(100 / len(df.columns) / 3, thread_id, "status_update")
        await manager.push(page_data, thread_id, "dashboard_data")
    return pd.DataFrame(col_report), page_data

def calculate_eta_squared(target, group_col):
    df_temp = pd.DataFrame({'target': target, 'group': group_col}).dropna()
    grand_mean = df_temp['target'].mean()
    ss_between = 0
    for category in df_temp['group'].unique():
        group_data = df_temp[df_temp['group'] == category]['target']
        ss_between += len(group_data) * (group_data.mean() - grand_mean) ** 2
    ss_total = np.sum((df_temp['target'] - grand_mean) ** 2)
    return ss_between / ss_total if ss_total != 0 else 0

def cramers_v(x, y):
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def select_best_anchor(target_col, df, type_description, threshold=0.2):
    scores = {}
    for col in df.columns:
        if col == target_col: continue
        if type_description[col][0] in ("numeric", "datetime", "timedelta"):
            if type_description[target_col][0] in ("numeric", "datetime", "timedelta"):
                scores[col] = abs(df[target_col].corr(df[col], method='spearman'))
            elif type_description[target_col][0] == "category":
                scores[col] = calculate_eta_squared(df[col], df[target_col])
            else: scores[col] = 0
        elif type_description[col][0] == "category":
            if type_description[target_col][0] in ("numeric", "datetime", "timedelta"):
                scores[col] = calculate_eta_squared(df[target_col], df[col])
            elif type_description[target_col][0] == "category":
                scores[col] = cramers_v(df[target_col], df[col])
            else: scores[col] = 0
        else: scores[col] = 0
    if not scores: return None
    best_anchor = max(scores, key=scores.get)
    return best_anchor if scores[best_anchor] >= threshold else None


async def colExecute(df, col_metadata, page_data, manager, thread_id, type_description):
    page_data_map = {page_data["col_analysis"][i]["display_name"]: i for i in range(len(page_data["col_analysis"]))}
    cols_to_drop = col_metadata[col_metadata["drop"]]["col"]
    dropped_df = df[cols_to_drop]
    df.drop(columns=cols_to_drop, inplace=True)
    col_flags = df[col_metadata[col_metadata["flag"]]["col"]].isnull().astype(int).add_suffix('_flag')

    constant_imputes = col_metadata[col_metadata["impute"] & (col_metadata["impute_type"] == "Constant")]
    for _, col in constant_imputes.iterrows():
        df[col["col"]] = df[col["col"]].fillna(
            "Unknown" if col["impute_value"] == "Unknown" else df[col["col"]].ffill())
        page_data["col_analysis"][page_data_map[col['col']]]["imputation_status"] = "complete"
        await manager.push(page_data, thread_id, "dashboard_data")
        await manager.push(100 / len(df.columns) / 3, thread_id, "status_update")

    simple_statistical_imputes = col_metadata[
        col_metadata["impute"] & (col_metadata["impute_type"] == "Simple Statistical")]
    for _, col in simple_statistical_imputes.iterrows():
        col_name = col["col"]
        imputer = SimpleImputer(strategy='most_frequent' if col["impute_value"] == "Mode" else 'median')
        df[col_name] = imputer.fit_transform(df[[col_name]]).ravel()
        page_data["col_analysis"][page_data_map[col_name]]["imputation_status"] = "complete"
        await manager.push(page_data, thread_id, "dashboard_data")
        await manager.push(100 / len(df.columns) / 3, thread_id, "status_update")

    conditional_statistical_imputes = col_metadata[
        col_metadata["impute"] & (col_metadata["impute_type"] == "Conditional Statistical")]
    for _, col in conditional_statistical_imputes.iterrows():
        col_name = col["col"]
        anchor_column = select_best_anchor(col_name, df, type_description)
        if anchor_column:
            if col["impute_value"] == "Median":
                df[col_name] = df[col_name].fillna(df.groupby(anchor_column)[col_name].transform('median'))
            elif col["impute_value"] == "Mode":
                df[col_name] = df[col_name].fillna(df.groupby(anchor_column)[col_name].transform(
                    lambda x: x.mode()[0] if not x.mode().empty else np.nan))
            elif col["impute_value"] == "ForwardFill":
                df[col_name] = df.groupby(anchor_column)[col_name].ffill()
            page_data["col_analysis"][page_data_map[col_name]]["imputation_status"] = "complete"
            await manager.push(page_data, thread_id, "dashboard_data")
            await manager.push(100 / len(df.columns) / 3, thread_id, "status_update")
        if not anchor_column or df[col_name].isnull().sum() > 0:
            imputer = SimpleImputer(strategy='median' if col["impute_value"] == "Median" else 'most_frequent')
            df[col_name] = imputer.fit_transform(df[[col_name]]).ravel()
            page_data["col_analysis"][page_data_map[col_name]]["imputation_status"] = "complete"
            await manager.push(page_data, thread_id, "dashboard_data")
            await manager.push(100 / len(df.columns) / 3, thread_id, "status_update")

    numerical = [i for i in df.columns if type_description[i][0] in ["numeric", "datetime", "timedelta"]]
    categorical = [i for i in df.columns if type_description[i][0] in ["category"]]
    model_imputes = col_metadata[col_metadata["impute"] & (col_metadata["impute_type"] == "Model Based")]

    if not model_imputes.empty:
        imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), max_iter=10,
                                   random_state=0)
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical),
            ('num', 'passthrough', numerical)
        ], remainder='drop')

        preprocessed_df = preprocessor.fit_transform(df[numerical + categorical])
        feature_names = preprocessor.get_feature_names_out()
        preprocessed_df = pd.DataFrame(preprocessed_df, columns=feature_names, index=df.index)

        mapping = {}
        for name, transformer, cols in preprocessor.transformers_:
            for col in cols:
                full_name = next((f for f in feature_names if f.endswith(f"__{col}") or f == col), None)
                if full_name: mapping[col] = full_name

        model_imputes_cols_raw = model_imputes["col"].tolist()

        # FIX Part 1: Convert raw model target columns to their transformed feature names map strictly
        model_imputes_name_transformed = [mapping[c] for c in model_imputes_cols_raw if c in mapping]

        corr_matrix = preprocessed_df.corr().abs()
        model_imputes_support_cols = corr_matrix[model_imputes_name_transformed].max(axis=1).nlargest(50).index.tolist()

        # FIX Part 2: Combine using only valid string titles found in preprocessed_df's columns
        model_imputes_final_features = list(set(model_imputes_support_cols + model_imputes_name_transformed))

        # Fit and transform using the verified transformed feature array list safely
        imputer.fit(preprocessed_df[model_imputes_final_features].sample(n=min(len(df), 1000), random_state=42))
        full_imputed = imputer.transform(preprocessed_df[model_imputes_final_features])
        imputed_df_subset = pd.DataFrame(full_imputed, columns=model_imputes_final_features, index=df.index)

        cat_transformer = preprocessor.named_transformers_['cat']
        cat_cols = preprocessor.transformers_[0][2]

        for col_name in model_imputes_cols_raw:
            if col_name in mapping:
                transformed_col_name = mapping[col_name]
                if col_name in categorical:
                    col_idx = cat_cols.index(col_name)
                    categories = cat_transformer.categories_[col_idx]

                    # Prevent floating precision values from breaking boundary indices
                    imputed_indices = imputed_df_subset[transformed_col_name].round().clip(0,
                                                                                           len(categories) - 1).astype(
                        int)
                    df[col_name] = [categories[i] for i in imputed_indices]
                else:
                    df[col_name] = imputed_df_subset[transformed_col_name]

            page_data["col_analysis"][page_data_map[col_name]]["imputation_status"] = "complete"
            await manager.push(page_data, thread_id, "dashboard_data")
            await manager.push(100 / len(df.columns) / 3, thread_id, "status_update")

    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        for col in df.columns:
            if df[col].isnull().any():
                if col in categorical:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                else:
                    df[col] = df[col].fillna(df[col].median())

    df = pd.concat([df, col_flags], axis=1)
    additional_type_descriptions = {i: ("category", {}) for i in col_flags}
    return df, dropped_df, additional_type_descriptions, page_data

async def runMissingnessNode(manager, config, minio_client, add_doc, state: SubgraphState):
    thread_id = config["metadata"]["thread_id"]
    df = minio_client.get_df(f"{thread_id}/MISSINGNESS_IMPUTED/Processed.parquet", thread_id)
    type_description = state.current_data_info.type_description
    col_advice, state.current_data_info.page_data[1] = await colAnalysis(df, type_description, state.current_data_info.page_data[1], manager, thread_id)
    df, dropped_df, additional_types, state.current_data_info.page_data[1] = await colExecute(df, col_advice, state.current_data_info.page_data[1], manager, thread_id, type_description)
    for i in additional_types:
        state.current_data_info.type_description[i] = additional_types[i]
    minio_client.write_pickle(state.current_data_info.type_description, f"{thread_id}/TYPE_VALIDATION/Type_Description.p", thread_id)
    minio_client.write_df(col_advice, f"{thread_id}/MISSINGNESS_IMPUTED/Missingness_report.parquet", thread_id)
    minio_client.write_df(df, f"{thread_id}/MISSINGNESS_IMPUTED/Processed.parquet", thread_id)
    minio_client.write_df(dropped_df, f"{thread_id}/MISSINGNESS_IMPUTED/Dropped_columns.parquet", thread_id)
    for i, row in tqdm(col_advice.iterrows()):
        txt = (f"Column '{row['col']}' ({row['dtype']}) has {row['mechanism']} missingness. "
               f"Imputed via {row['impute_type']} with value {row['impute_value']}. Severity: {row['severity']}.")
        if row["drop"]: txt += " Column was dropped."
        add_doc(txt, {"source": "missingness", "column": row['col']}, f"miss_{i}")
    await manager.push(2, thread_id, "page_change")
    await manager.push(0.34, thread_id, "status_update")
    state.current_data_info.current_progress += 67
    return state