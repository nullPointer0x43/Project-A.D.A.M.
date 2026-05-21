import itertools
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, spearmanr, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from Nodes.DataStructs import SubgraphState


def get_cramers_v_magnitude(v, df_small):
    thresh = {1: (0.1, 0.3, 0.5), 2: (0.07, 0.21, 0.35)}.get(df_small, (0.06, 0.17, 0.29))
    if v < thresh[0]: return "Negligible"
    if v < thresh[1]: return "Small"
    if v < thresh[2]: return "Medium"
    return "Large"


def get_num_cat_magnitude(effect_val, effect_type):
    abs_val = abs(effect_val)
    if effect_type == "Cohen's d":
        if abs_val < 0.2: return "Negligible"
        if abs_val < 0.5: return "Small"
        if abs_val < 0.8: return "Medium"
        return "Large"
    if effect_type == "Rank-Biserial r":
        if abs_val < 0.1: return "Negligible"
        if abs_val < 0.3: return "Small"
        if abs_val < 0.5: return "Medium"
        return "Large"
    if effect_type in ["Omega-Sq", "Epsilon-Sq"]:
        if abs_val < 0.01: return "Negligible"
        if abs_val < 0.06: return "Small"
        if abs_val < 0.14: return "Medium"
        return "Large"
    return "N/A"


def catCatAnalysis(df, pair):
    contingency = pd.crosstab(df[pair[0]], df[pair[1]])
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    sparse_cells = (expected < 5).any()
    if sparse_cells:
        if contingency.shape == (2, 2):
            _, final_p = fisher_exact(contingency)
            test_used = "Fisher Exact"
        else:
            _, final_p, _, _ = chi2_contingency(contingency, lambda_="log-likelihood")
            test_used = "G-Test"
    else:
        final_p = p_chi
        test_used = "Pearson Chi-Square"
    n = contingency.sum().sum()
    phi2 = max(0, (chi2 / n) - ((contingency.shape[1] - 1) * (contingency.shape[0] - 1)) / (n - 1))
    r_corr = contingency.shape[0] - ((contingency.shape[0] - 1) ** 2) / (n - 1)
    k_corr = contingency.shape[1] - ((contingency.shape[1] - 1) ** 2) / (n - 1)
    v_val = np.sqrt(phi2 / min((k_corr - 1), (r_corr - 1)))
    df_small = min(contingency.shape) - 1
    return {
        'p_value': float(final_p),
        'effect_val': float(v_val),
        'effect_type': "Cramer's V",
        'magnitude': get_cramers_v_magnitude(v_val, df_small),
        'test_used': test_used
    }


def numNumAnalysis(df, pair, normality_profile):
    col1, col2 = pair
    data = df[[col1, col2]].dropna()
    is_normal = normality_profile.get(col1, False) and normality_profile.get(col2, False)
    if is_normal:
        coeff, p_val = pearsonr(data[col1], data[col2])
        method = 'Pearson'
    else:
        coeff, p_val = spearmanr(data[col1], data[col2])
        method = 'Spearman'
    return {
        'p_value': float(p_val),
        'effect_val': float(coeff),
        'effect_type': f"{method} r",
        'magnitude': "Large" if abs(coeff) > 0.5 else "Medium" if abs(coeff) > 0.3 else "Small",
        'test_used': f"{method} Correlation"
    }


def numCatAnalysis(df, pair, normality_profile):
    num_col, cat_col = pair
    clean_df = df[[num_col, cat_col]].dropna()
    if clean_df[num_col].nunique() <= 1 or clean_df[cat_col].nunique() < 2:
        return {
            'p_value': 1.0,
            'test_used': 'Skipped: Low Variance',
            'effect_val': 0.0,
            'effect_type': 'N/A',
            'magnitude': 'None'
        }
    group_data = clean_df.groupby(cat_col)[num_col]
    groups = [g.values for _, g in group_data if len(g) > 1]
    k = len(groups)
    if k < 2:
        return {
            'p_value': 1.0,
            'test_used': 'Skipped: Insufficient Groups',
            'effect_val': 0.0,
            'effect_type': 'N/A',
            'magnitude': 'None'
        }
    is_normal = bool(normality_profile.get(num_col, False))
    if all(np.var(g) > 0 for g in groups):
        _, p_lev = stats.levene(*groups)
        equal_var = bool(p_lev > 0.05)
    else:
        equal_var = False
    res_data = {'pair_type': 'Num-Cat'}
    if k == 2:
        g1, g2 = groups[0], groups[1]
        if is_normal:
            test = stats.ttest_ind(g1, g2, equal_var=equal_var)
            res_data.update({'test_used': "t-test" if equal_var else "Welch's t-test", 'p_value': float(test.pvalue)})
            sd_pool = np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
            eff = float((np.mean(g1) - np.mean(g2)) / sd_pool) if sd_pool != 0 else 0.0
            res_data.update({'effect_val': eff, 'effect_type': "Cohen's d"})
        else:
            u_res = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            eff = float(1 - (2 * u_res.statistic) / (len(g1) * len(g2)))
            res_data.update({'test_used': "Mann-Whitney U", 'p_value': float(u_res.pvalue), 'effect_val': eff,
                             'effect_type': "Rank-Biserial r"})
    else:
        if is_normal:
            if equal_var:
                anova = stats.f_oneway(*groups)
                ss_between = sum(len(g) * (np.mean(g) - np.mean(clean_df[num_col])) ** 2 for g in groups)
                ss_total = sum((clean_df[num_col] - np.mean(clean_df[num_col])) ** 2)
                ms_error = (ss_total - ss_between) / (len(clean_df) - k)
                eff = float((ss_between - (k - 1) * ms_error) / (ss_total + ms_error))
                res_data.update({'test_used': 'ANOVA', 'p_value': float(anova.pvalue), 'effect_val': max(0.0, eff),
                                 'effect_type': 'Omega-Sq'})
            else:
                model = ols(f"Q('{num_col}') ~ C(Q('{cat_col}'))", data=clean_df).fit(cov_type='HC3')
                a_table = sm.stats.anova_lm(model, robust='HC3')
                res_data.update(
                    {'test_used': "Welch's ANOVA", 'p_value': float(a_table["PR(>F)"][0]), 'effect_val': 0.0,
                     'effect_type': 'N/A'})
        else:
            kw = stats.kruskal(*groups)
            n = len(clean_df)
            eff = float(kw.statistic / ((n ** 2 - 1) / (n + 1)))
            res_data.update({'test_used': 'Kruskal-Wallis', 'p_value': float(kw.pvalue), 'effect_val': eff,
                             'effect_type': 'Epsilon-Sq'})
    res_data['magnitude'] = get_num_cat_magnitude(res_data.get('effect_val', 0), res_data.get('effect_type', 'N/A'))
    return res_data


async def runMultivariateNode(manager, config, minio_client, add_doc, state: SubgraphState = None):
    thread_id = config["metadata"]["thread_id"]
    df = minio_client.get_df(f"{thread_id}/MISSINGNESS_IMPUTED/Processed.parquet", thread_id)
    df = df.drop(columns=[c for c in df.columns if c.lower().endswith('flag')])

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)

    type_description = state.current_data_info.type_description
    categorical = [c for c in df.columns if type_description[c][0] == "category"]
    numerical = [c for c in df.columns if type_description[c][0] in ["numeric", "datetime", "timedelta"]]
    univariate_data = minio_client.get_pickle(f"{thread_id}/UNIVARIATE/Analysis_data.p", thread_id)
    normality_profile = {c: univariate_data[c]["normal"] for c in numerical}

    analysis_results = []
    all_pairs = [
        ('Cat-Cat', list(itertools.combinations(categorical, 2))),
        ('Num-Num', list(itertools.combinations(numerical, 2))),
        ('Num-Cat', list(itertools.product(numerical, categorical)))
    ]

    total_pairs = sum(len(p[1]) for p in all_pairs)
    processed = 0

    for p_type, pairs in all_pairs:
        for p in tqdm(pairs, desc=f"Analyzing {p_type}"):
            processed += 1
            await manager.push((processed / total_pairs) * 100, thread_id, "status_update")
            try:
                if p_type == 'Cat-Cat':
                    res = catCatAnalysis(df, p)
                elif p_type == 'Num-Num':
                    res = numNumAnalysis(df, p, normality_profile)
                else:
                    res = numCatAnalysis(df, p, normality_profile)
                res.update({'pair': p, 'pair_type': p_type})
                analysis_results.append(res)
            except Exception:
                continue

    p_values = [r['p_value'] for r in analysis_results]
    rejected, p_adj, _, _ = multipletests(p_values, method='fdr_bh')

    final_data = {}
    for i, r in enumerate(analysis_results):
        pair = r.pop("pair")
        pair_str = f"('{pair[0]}', '{pair[1]}')"
        final_data[pair_str] = {
            "pair_type": r.get("pair_type", "Unknown"),
            "test_used": r.get("test_used", "N/A"),
            "effect_val": float(r.get("effect_val", 0.0)),
            "p_adj": float(p_adj[i]),
            "magnitude": r.get("magnitude", "N/A"),
            "rejected": bool(rejected[i])
        }

    minio_client.write_pickle(final_data, f"{thread_id}/MULTIVARIATE/Analysis_data.p", thread_id)

    for pair, value in final_data.items():
        txt = f"Multivariate relationship between {pair}: {value}."
        add_doc(txt, {"source": "multivariate", "pair": str(pair)}, f"multi_{str(pair)}")

    state.current_data_info.page_data[3] = final_data
    await manager.push(final_data, thread_id, "dashboard_data")
    await manager.push(4, thread_id, "page_change")
    await manager.push(200, thread_id, "status_update")
    state.current_data_info.current_progress += 200
    return state