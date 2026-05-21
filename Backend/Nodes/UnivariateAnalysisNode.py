import diptest
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import shapiro, normaltest
from Nodes.DataStructs import SubgraphState


def numericColumnAnalysis(col_data: pd.Series, alpha=0.05):
    data = dict()
    data["low-variance"] = bool(col_data.var() <= 0.05)
    data["is-id"] = bool(col_data.var() > 0.95)

    if col_data.apply(lambda x: x > 0).all():
        data["signage"] = "positive"
    elif col_data.apply(lambda x: x < 0).all():
        data["signage"] = "negative"
    else:
        data["signage"] = "mixed"
    data["zero"] = bool(col_data.apply(lambda x: x == 0).any())

    data["mean"] = col_data.mean().item()
    data["median"] = col_data.median().item()
    data["mode"] = col_data.mode().tolist()
    data["std"] = col_data.std().item()
    data["var"] = col_data.var().item()
    data["skew"] = col_data.skew().item()
    data["kurtosis"] = col_data.kurtosis().item()

    data["coeff_of_variation"] = data["std"] / data["mean"]
    data["MAD"] = (col_data - data["mean"]).median().item()

    data["percentiles"] = {i: col_data.quantile(i).item() for i in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]}
    data["iqr"] = (col_data.quantile(0.75) - col_data.quantile(0.25)).item()
    data["range"] = (col_data.max() - col_data.min()).item()
    data["extreme-outliers"] = data["percentiles"][0.99] > 3 * data["iqr"] and col_data.max() > 3 * data["iqr"]

    stat, p_value = diptest.diptest(col_data.values)
    if abs(data["skew"]) > 0.5:
        if data["skew"] < 0:
            data["distribution_type"] = "left-skewed"
        elif data["skew"] > 0:
            data["distribution_type"] = "right-skewed"
    else:
        if p_value < alpha:
            data["distribution_type"] = "multi-modal"
        elif abs(data["kurtosis"] + 1.2) < 0.1:
            data["distribution_type"] = "uniform"
        else:
            data["distribution_type"] = "symmetric"

    normal_p = shapiro(col_data.values).pvalue if len(col_data) < 5000 else normaltest(col_data.values).pvalue
    data["normal"] = bool(normal_p > alpha)

    data["IQR_outliers"] = col_data[(col_data < data["percentiles"][0.25] - 1.5 * data["iqr"]) |
                                    (col_data > data["percentiles"][0.75] + 1.5 * data["iqr"])]
    data["IQR_pc"] = len(data["IQR_outliers"]) / len(col_data)
    data["Z_outliers"] = col_data[(col_data - data["mean"]) / data["std"] > 3]
    data["Z_pc"] = len(data["Z_outliers"]) / len(col_data)
    data["outlier_flag"] = data["IQR_pc"] > 0.05 or data["Z_pc"] > 0.05

    sorted_series = col_data.sort_values()
    data["bottom_5"] = sorted_series.head(5)
    data["top_5"] = sorted_series.tail(5)

    if -0.5 < data["skew"] < 0.5:
        data["transform"] = "None"
    elif 0.5 <= data["skew"] < 1:
        data["transform"] = "sqrt"
    elif 1 < data["skew"] < 10:
        if data["signage"] == "positive":
            c = 1 if data["zero"] else 0
        else:
            c = col_data.min() + 1
        data["transform"] = f"log;{c}"
    elif data["skew"] > 10:
        data["transform"] = "reciprocal"
    elif -1 < data["skew"] < -0.5:
        data["transform"] = "sqr"
    else:
        data["transform"] = "reflect-log"

    return data


def categoricalColumnAnalysis(col_data: pd.Series, target=False):
    data = dict()

    frequency_table = pd.DataFrame(index=col_data.unique(), columns=["count", "percent"])
    frequency_table["count"] = col_data.value_counts()
    frequency_table["percent"] = col_data.value_counts() / len(col_data) * 100
    data["frequency"] = frequency_table

    data["rare"] = frequency_table[frequency_table["percent"] < 1].index.tolist()
    data["suggested_merge_categories"] = frequency_table[frequency_table["count"] < 50].index.tolist()

    data["cardinality"] = col_data.nunique()
    data["cardinality_after_merge"] = col_data.nunique() - len(data["suggested_merge_categories"]) + 1

    if data["cardinality"] == 2:
        data["cardinality_tier"] = "binary"
        data["encoding"] = "label"
    elif 2 < data["cardinality"] <= 10:
        data["cardinality_tier"] = "low"
        data["encoding"] = "OHE"
    elif 10 < data["cardinality"] <= 50:
        data["cardinality_tier"] = "medium"
        data["encoding"] = "target"
    elif 50 < data["cardinality"] <= 200:
        data["cardinality_tier"] = "high"
        data["encoding"] = "hashing"
    else:
        data["cardinality_tier"] = "very-high"
        data["encoding"] = "hashing"

    if data["cardinality_after_merge"]:
        if 2 < data["cardinality_after_merge"] <= 10:
            data["encoding"] += " or OHE after merge"
        elif 10 < data["cardinality_after_merge"] <= 50:
            data["encoding"] = " or target after merge"

    data["binary_flag"] = data["cardinality_tier"] == "binary"
    data["high_card_flag"] = data["cardinality_tier"] == "high" or data["cardinality_tier"] == "very-high"
    data["suspected_text"] = data["cardinality_tier"] == "very-high" and col_data.apply(lambda x: len(x)).mean > 150

    sorted_vals = frequency_table.sort_values("percent", ascending=False)
    data["top-percentages"] = {i: (sorted_vals.index.tolist()[:i], sum(sorted_vals["percent"].tolist()[:i])) for i in
                               [1, 3, 5]}

    data["entropy"] = frequency_table["percent"].apply(lambda x: - (x * np.log2(x / 100) / 100)).sum().item()
    data["entropy"] /= np.log2(data["cardinality"]).item()
    data["gini"] = (1 - frequency_table["percent"].apply(lambda x: (x / 100) ** 2).sum()).item()
    data["gini"] /= (1 - 1 / data["cardinality"])

    if target:
        data["imbalance_ratio"] = (frequency_table["count"].max() / frequency_table["count"].min()).item()
        if data["imbalance_ratio"] < 1.5:
            data["imbalance_tier"] = "Balanced"
            data["metric"] = "None"
            data["handling"] = "None"
        elif data["imbalance_ratio"] < 3:
            data["imbalance_tier"] = "Slight Imbalanced"
            data["metric"] = "accuracy + F1"
            data["handling"] = "Class Weighting"
        elif data["imbalance_ratio"] < 10:
            data["imbalance_tier"] = "Moderate Imbalanced"
            data["metric"] = "F1 + Precision-Recall-AUC"
            data["handling"] = "Class Weighting + SMOTE"
        elif data["imbalance_ratio"] < 100:
            data["imbalance_tier"] = "Severe Imbalanced"
            data["metric"] = "Precision-Recall-AUC + ROC-AUC"
            data["handling"] = "SMOTE + ADASYN"
        else:
            data["imbalance_tier"] = "Extremely Imbalanced"
            data["metric"] = "Precision-Recall-AUC + MCC"
            data["handling"] = "SMOTE + BalancedBaggingClassifier + EasyEnsemble + Flag"

    return data


async def push_univariate_to_pipe(univariate_results, df, manager, thread_id):
    formatted_data = {}

    for col, stats in univariate_results.items():
        col_type = stats.get("type", "numeric" if "mean" in stats else "category")
        entry = {
            "type": col_type,
            "is-id": stats.get("is-id", False),
            "outlier_flag": stats.get("outlier_flag", False),
        }

        if col_type == "numeric":
            entry.update({
                "mean": round(stats.get("mean", 0), 2),
                "median": round(stats.get("median", 0), 2),
                "std": round(stats.get("std", 0), 2),
                "skew": round(stats.get("skew", 0), 2),
                "kurtosis": round(stats.get("kurtosis", 0), 2),
                "iqr": round(stats.get("iqr", 0), 2),
                "percentiles": {str(k): round(v, 2) for k, v in stats.get("percentiles", {}).items()},
                "distribution_type": stats.get("distribution_type", "unknown"),
                "normal": stats.get("normal", False),
                "Z_pc": round(stats.get("Z_pc", 0), 4),
                "transform": stats.get("transform", "None"),
            })

            counts, bin_edges = np.histogram(df[col].dropna(), bins=10)
            entry["chart_data"] = [
                {"bin": f"{round(bin_edges[i], 1)}-{round(bin_edges[i + 1], 1)}", "count": int(counts[i])}
                for i in range(len(counts))
            ]
        else:
            entry.update({
                "cardinality": stats.get("cardinality"),
                "cardinality_tier": stats.get("cardinality_tier"),
                "encoding": stats.get("encoding"),
                "entropy": round(stats.get("entropy", 0), 3),
                "gini": round(stats.get("gini", 0), 3),
                "imbalance_ratio": round(stats.get("imbalance_ratio", 0), 2) if "imbalance_ratio" in stats else None,
                "imbalance_tier": stats.get("imbalance_tier", "Balanced"),
                "handling": stats.get("handling", "None"),
                "rare": stats.get("rare", []),
            })

            freq = df[col].value_counts().head(10)
            entry["chart_data"] = [
                {"name": str(k), "value": int(v)} for k, v in freq.items()
            ]

        formatted_data[col] = entry

    await manager.push(formatted_data, thread_id, "dashboard_data")
    return formatted_data


async def runUnivariateNode(manager, config, minio_client, add_doc, state: SubgraphState):
    thread_id = config["metadata"]["thread_id"]
    df = minio_client.get_df(f"{thread_id}/MISSINGNESS_IMPUTED/Processed.parquet", thread_id)
    type_description = state.current_data_info.type_description
    univariateData = {}

    for col in tqdm(df.columns):
        if type_description[col][0] == "category":
            univariateData[col] = categoricalColumnAnalysis(df[col])
        elif type_description[col][0] in ["numeric", "datetime", "timedelta"]:
            univariateData[col] = numericColumnAnalysis(df[col])

    state.current_data_info.page_data[2] = await push_univariate_to_pipe(univariateData, df, manager, thread_id)
    minio_client.write_pickle(univariateData, f"{thread_id}/UNIVARIATE/Analysis_data.p", thread_id)

    for col, stats in univariateData.items():
        stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
        txt = f"Univariate statistics for '{col}': {stats_str}."
        add_doc(txt, {"source": "univariate", "column": col}, f"uni_{col}")

    await manager.push(100, thread_id, "status_update")
    await manager.push(3, thread_id, "page_change")

    state.current_data_info.current_progress += 100
    return state