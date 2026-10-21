
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Polymer Series Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #F7F9FC;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E6EAF0;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        .app-hero {
            background: linear-gradient(135deg, #0E223D 0%, #163A63 100%);
            border-radius: 18px;
            padding: 26px 30px;
            margin-bottom: 18px;
            box-shadow: 0 10px 28px rgba(24, 46, 78, 0.10);
        }

        .app-hero h1 {
            color: white;
            font-size: 2rem;
            line-height: 1.15;
            margin: 0;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .app-hero p {
            color: rgba(255,255,255,0.82);
            margin: 8px 0 0 0;
            font-size: 0.98rem;
        }

        .eyebrow {
            color: #A8C4E6;
            text-transform: uppercase;
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            margin-bottom: 7px;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E6EAF0;
            border-radius: 14px;
            padding: 16px 18px;
            min-height: 98px;
            box-shadow: 0 3px 12px rgba(18, 38, 63, 0.035);
        }

        .metric-label {
            color: #687587;
            font-size: 0.78rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.055em;
            margin-bottom: 6px;
        }

        .metric-value {
            color: #10243E;
            font-size: 1.45rem;
            font-weight: 720;
            line-height: 1.1;
        }

        .metric-subtext {
            color: #8A96A6;
            font-size: 0.78rem;
            margin-top: 5px;
        }

        .section-card {
            background: #FFFFFF;
            border: 1px solid #E6EAF0;
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 3px 12px rgba(18, 38, 63, 0.03);
        }

        .status-pill {
            display: inline-block;
            background: #EAF4EE;
            color: #2B6843;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.76rem;
            font-weight: 650;
        }

        .muted-pill {
            display: inline-block;
            background: #EEF2F7;
            color: #536274;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.76rem;
            font-weight: 650;
            margin-right: 6px;
            margin-bottom: 5px;
        }

        .small-note {
            color: #738195;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        h2, h3 {
            color: #10243E;
            letter-spacing: -0.012em;
        }

        div[data-testid="stButton"] > button {
            border-radius: 10px;
            font-weight: 650;
            min-height: 42px;
        }

        div[data-testid="stDownloadButton"] > button {
            border-radius: 10px;
            font-weight: 650;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #E6EAF0;
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stFileUploader"] {
            background: white;
            border-radius: 12px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #FFFFFF;
            padding: 5px;
            border: 1px solid #E6EAF0;
            border-radius: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding-left: 18px;
            padding-right: 18px;
        }

        .stTabs [aria-selected="true"] {
            background: #EFF4FA;
        }

        hr {
            border-color: #E8ECF2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-hero">
        <div class="eyebrow">LCMS polymer screening</div>
        <h1>Polymer Series Explorer</h1>
        <p>Map homologous polymer series across an LCMS feature table, connect companion adduct forms, and inspect every assignment interactively.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

MONOISOTOPIC_MASS = {
    "H": 1.00782503223,
    "C": 12.00000000000,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Na": 22.9897692820,
    "Mg": 23.985041697,
    "Al": 26.98153853,
    "Si": 27.97692653465,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "K": 38.9637064864,
    "Ca": 39.962590863,
    "Fe": 55.93493633,
    "Br": 78.9183376,
    "I": 126.904468,
}

PROTON_MASS = 1.007276466621
ELECTRON_MASS = 0.000548579909

ADDUCT_LIBRARY = {
    "[M+H]+": {
        "shift": PROTON_MASS,
        "charge": 1,
        "polarity": "positive",
        "description": "Protonated",
    },
    "[M+Na]+": {
        "shift": MONOISOTOPIC_MASS["Na"] - ELECTRON_MASS,
        "charge": 1,
        "polarity": "positive",
        "description": "Sodiated",
    },
    "[M+K]+": {
        "shift": MONOISOTOPIC_MASS["K"] - ELECTRON_MASS,
        "charge": 1,
        "polarity": "positive",
        "description": "Potassiated",
    },
    "[M+NH4]+": {
        "shift": (
            MONOISOTOPIC_MASS["N"]
            + 4 * MONOISOTOPIC_MASS["H"]
            - ELECTRON_MASS
        ),
        "charge": 1,
        "polarity": "positive",
        "description": "Ammoniated",
    },
    "[M-H]-": {
        "shift": -PROTON_MASS,
        "charge": -1,
        "polarity": "negative",
        "description": "Deprotonated",
    },
}

REPEAT_LIBRARY = {
    "CH2 / methylene": {
        "full_name": "Methylene increment",
        "formula": "CH2",
        "notes": "Common 14.01565 Da hydrocarbon homologous increment",
    },
    "PEG / EO": {
        "full_name": "Polyethylene glycol / ethylene oxide",
        "formula": "C2H4O",
        "notes": "Ethylene oxide repeat",
    },
    "PPG / PO": {
        "full_name": "Polypropylene glycol / propylene oxide",
        "formula": "C3H6O",
        "notes": "Propylene oxide repeat",
    },
    "PE": {
        "full_name": "Polyethylene",
        "formula": "C2H4",
        "notes": "Ethylene repeat",
    },
    "PP": {
        "full_name": "Polypropylene",
        "formula": "C3H6",
        "notes": "Propylene repeat",
    },
    "PS": {
        "full_name": "Polystyrene",
        "formula": "C8H8",
        "notes": "Styrene repeat",
    },
    "PMMA": {
        "full_name": "Poly(methyl methacrylate)",
        "formula": "C5H8O2",
        "notes": "Methyl methacrylate repeat",
    },
    "PLA": {
        "full_name": "Poly(lactic acid)",
        "formula": "C3H4O2",
        "notes": "Lactyl repeat",
    },
    "Nylon 6": {
        "full_name": "Nylon 6",
        "formula": "C6H11NO",
        "notes": "Caprolactam derived repeat",
    },
    "PDMS": {
        "full_name": "Polydimethylsiloxane",
        "formula": "C2H6OSi",
        "notes": "Dimethylsiloxane repeat",
    },
    "PET": {
        "full_name": "Polyethylene terephthalate",
        "formula": "C10H8O4",
        "notes": "Ethylene terephthalate repeat",
    },
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def formula_mass(formula: str) -> float:
    if not formula:
        raise ValueError("Formula is empty.")

    formula = formula.strip()
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    rebuilt = "".join(element + count for element, count in tokens)

    if rebuilt != formula:
        raise ValueError("Use a simple formula such as C2H4O, C8H8, or C2H6OSi.")

    total = 0.0
    for element, count_text in tokens:
        if element not in MONOISOTOPIC_MASS:
            raise ValueError(f"Element {element} is not in the built in mass table.")
        count = int(count_text) if count_text else 1
        total += MONOISOTOPIC_MASS[element] * count

    return total


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "mz": ["mz", "m/z", "mass_to_charge", "mass to charge"],
        "rt": ["rt", "retention_time", "retention time", "retentiontime"],
        "intensity": ["intensity", "height", "abundance", "area", "peak_area", "peak area"],
    }

    lowered = {str(c).strip().lower(): c for c in df.columns}
    rename = {}

    for canonical, names in aliases.items():
        for name in names:
            if name in lowered:
                rename[lowered[name]] = canonical
                break

    out = df.rename(columns=rename).copy()

    if "mz" not in out.columns:
        raise ValueError("A column named m/z or mz is required.")

    out["mz"] = pd.to_numeric(out["mz"], errors="coerce")
    out["rt"] = (
        pd.to_numeric(out["rt"], errors="coerce")
        if "rt" in out.columns
        else np.nan
    )
    out["intensity"] = (
        pd.to_numeric(out["intensity"], errors="coerce")
        if "intensity" in out.columns
        else 1.0
    )

    out = out.dropna(subset=["mz"]).reset_index(drop=True)
    out["feature_id"] = np.arange(1, len(out) + 1)

    return out[["feature_id", "mz", "rt", "intensity"]]


def find_series(
    features: pd.DataFrame,
    repeat_mass: float,
    charge: int,
    ppm_tolerance: float,
    min_members: int,
    max_repeat_gap: int,
    rt_tolerance,
):
    """Find connected feature groups consistent with repeat_mass / charge spacing."""
    df = features.sort_values("mz").reset_index(drop=True).copy()
    spacing = repeat_mass / abs(charge)
    mz = df["mz"].to_numpy()
    n = len(df)
    graph = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            delta = mz[j] - mz[i]
            max_delta = spacing * max_repeat_gap * (1 + ppm_tolerance * 1e-6)
            if delta > max_delta:
                break

            repeats = int(round(delta / spacing))
            if repeats < 1 or repeats > max_repeat_gap:
                continue

            target = repeats * spacing
            ppm_error_value = (delta - target) / target * 1e6

            if abs(ppm_error_value) <= ppm_tolerance:
                if rt_tolerance is not None:
                    rt_i = df.loc[i, "rt"]
                    rt_j = df.loc[j, "rt"]
                    if pd.notna(rt_i) and pd.notna(rt_j):
                        if abs(rt_j - rt_i) > rt_tolerance:
                            continue

                graph[i].add(j)
                graph[j].add(i)

    visited = set()
    components = []

    for start_node in range(n):
        if start_node in visited or start_node not in graph:
            continue

        queue = deque([start_node])
        visited.add(start_node)
        comp = []

        while queue:
            node = queue.popleft()
            comp.append(node)
            for nxt in graph[node]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        if len(comp) >= min_members:
            components.append(sorted(comp, key=lambda x: mz[x]))

    components.sort(key=len, reverse=True)
    rows = []

    for local_series_number, comp in enumerate(components, start=1):
        base_idx = comp[0]
        base_mz = mz[base_idx]

        for idx in comp:
            repeat_number = int(round((mz[idx] - base_mz) / spacing))
            predicted = base_mz + repeat_number * spacing
            ppm_from_series = (
                0.0
                if repeat_number == 0
                else (mz[idx] - predicted) / predicted * 1e6
            )

            row = df.loc[idx].to_dict()
            row.update({
                "local_series_number": local_series_number,
                "repeat_index": repeat_number,
                "charge": int(charge),
                "repeat_mass": float(repeat_mass),
                "spacing_mz": float(spacing),
                "series_ppm_error": float(ppm_from_series),
            })
            rows.append(row)

    return pd.DataFrame(rows)


def add_kendrick(df: pd.DataFrame, repeat_mass: float):
    out = df.copy()
    nominal = round(repeat_mass)
    out["kendrick_mass"] = out["mz"] * nominal / repeat_mass
    out["kendrick_nominal_mass"] = np.round(out["kendrick_mass"])
    out["kmd"] = out["kendrick_nominal_mass"] - out["kendrick_mass"]
    return out


def series_shape_metrics(series_df: pd.DataFrame):
    idx = sorted(set(int(v) for v in series_df["repeat_index"].dropna()))
    members = len(idx)

    if members <= 1:
        continuity = 0.0
        missing = 0
    else:
        gaps = np.diff(idx)
        continuity = float(np.mean(gaps == 1))
        missing = int(np.sum(np.maximum(gaps - 1, 0)))

    ppm_values = np.abs(pd.to_numeric(series_df["series_ppm_error"], errors="coerce"))
    median_abs_ppm = float(np.nanmedian(ppm_values)) if ppm_values.notna().any() else np.nan
    max_abs_ppm = float(np.nanmax(ppm_values)) if ppm_values.notna().any() else np.nan

    rt_corr = np.nan
    rt_data = series_df[["repeat_index", "rt"]].dropna()
    if len(rt_data) >= 3 and rt_data["rt"].nunique() > 1:
        ranks_x = rt_data["repeat_index"].rank().to_numpy()
        ranks_y = rt_data["rt"].rank().to_numpy()
        if np.std(ranks_x) > 0 and np.std(ranks_y) > 0:
            rt_corr = float(np.corrcoef(ranks_x, ranks_y)[0, 1])

    return {
        "members": members,
        "continuity_fraction": continuity,
        "missing_repeats": missing,
        "median_abs_ppm": median_abs_ppm,
        "max_abs_ppm": max_abs_ppm,
        "rt_rank_correlation": rt_corr,
    }


def dedupe_charge_hypotheses(results: pd.DataFrame, ppm_tolerance: float):
    """Keep the most plausible charge when multiple charge hypotheses use the same features."""
    if results.empty or "series_id" not in results.columns:
        return results

    candidates = []
    for series_id, sdf in results.groupby("series_id"):
        metrics = series_shape_metrics(sdf)
        ppm_component = 0.0 if np.isnan(metrics["median_abs_ppm"]) else max(
            0.0, 1.0 - metrics["median_abs_ppm"] / max(ppm_tolerance, 1e-9)
        )
        score = (
            60.0 * metrics["continuity_fraction"]
            + 25.0 * min(metrics["members"] / 8.0, 1.0)
            + 15.0 * ppm_component
        )
        candidates.append({
            "series_id": series_id,
            "repeat_unit": sdf["repeat_unit"].iloc[0],
            "search_origin": sdf.get("search_origin", pd.Series(["Targeted"])).iloc[0],
            "features": frozenset(int(v) for v in sdf["feature_id"]),
            "score": score,
            "charge": int(sdf["charge"].iloc[0]),
        })

    candidates.sort(key=lambda x: (x["score"], -x["charge"]), reverse=True)
    kept = []

    for cand in candidates:
        reject = False
        for existing in kept:
            if (
                existing["repeat_unit"] != cand["repeat_unit"]
                or existing["search_origin"] != cand["search_origin"]
            ):
                continue

            union = cand["features"] | existing["features"]
            intersection = cand["features"] & existing["features"]
            if not union or not intersection:
                continue
            jaccard = len(intersection) / len(union)
            containment = len(intersection) / min(len(cand["features"]), len(existing["features"]))
            # A wrong charge often explains every 2nd or 3rd member of a better
            # series. Containment catches that nested-subset artifact while still
            # allowing distinct, partially overlapping families.
            if jaccard >= 0.80 or containment >= 0.80:
                reject = True
                break

        if not reject:
            kept.append(cand)

    keep_ids = {item["series_id"] for item in kept}
    return results[results["series_id"].isin(keep_ids)].copy()


def run_multi_repeat_search(
    features,
    search_units,
    charges,
    ppm_tolerance,
    min_members,
    max_repeat_gap,
    rt_tolerance,
    search_origin="Targeted",
):
    all_results = []

    for search_unit in search_units:
        for charge in charges:
            result = find_series(
                features=features,
                repeat_mass=search_unit["mass"],
                charge=int(charge),
                ppm_tolerance=ppm_tolerance,
                min_members=min_members,
                max_repeat_gap=max_repeat_gap,
                rt_tolerance=rt_tolerance,
            )

            if result.empty:
                continue

            result = add_kendrick(result, search_unit["mass"])
            result["repeat_unit"] = search_unit["label"]
            result["repeat_formula"] = search_unit.get("formula", "")
            result["repeat_description"] = search_unit.get("description", "")
            result["search_origin"] = search_origin
            result["series_id"] = result["local_series_number"].apply(
                lambda x: (
                    f'{search_unit["label"]} | z{int(charge)} | '
                    f'Series {int(x)}'
                )
            )
            all_results.append(result)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    return dedupe_charge_hypotheses(combined, ppm_tolerance)


def build_summary(results: pd.DataFrame):
    if results.empty:
        return pd.DataFrame()

    rows = []
    for series_id, sdf in results.groupby("series_id"):
        metrics = series_shape_metrics(sdf)
        rows.append({
            "repeat_unit": sdf["repeat_unit"].iloc[0],
            "repeat_formula": sdf["repeat_formula"].iloc[0],
            "repeat_mass": float(sdf["repeat_mass"].iloc[0]),
            "charge": int(sdf["charge"].iloc[0]),
            "series_id": series_id,
            "members": int(metrics["members"]),
            "continuity_fraction": metrics["continuity_fraction"],
            "missing_repeats": metrics["missing_repeats"],
            "median_abs_ppm": metrics["median_abs_ppm"],
            "max_abs_ppm": metrics["max_abs_ppm"],
            "rt_rank_correlation": metrics["rt_rank_correlation"],
            "mz_min": float(sdf["mz"].min()),
            "mz_max": float(sdf["mz"].max()),
            "rt_min": float(sdf["rt"].min()) if sdf["rt"].notna().any() else np.nan,
            "rt_max": float(sdf["rt"].max()) if sdf["rt"].notna().any() else np.nan,
            "total_intensity": float(sdf["intensity"].sum()),
        })

    return pd.DataFrame(rows).sort_values(
        ["repeat_unit", "members", "mz_min"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def ppm_error(observed, expected):
    return (observed - expected) / expected * 1e6


def find_adduct_series_links(
    results: pd.DataFrame,
    selected_adducts,
    ppm_tolerance: float,
    rt_tolerance: float,
    min_matches: int = 3,
):
    """Link separate z=1 homologous series that are supported companion adduct forms."""
    if results.empty or len(selected_adducts) < 2:
        return pd.DataFrame(), pd.DataFrame()

    link_rows = []
    member_rows = []
    seen = set()

    for repeat_unit in results["repeat_unit"].unique():
        ru = results[results["repeat_unit"] == repeat_unit]
        series_ids = list(ru["series_id"].unique())

        for i in range(len(series_ids)):
            a_id = series_ids[i]
            a = ru[ru["series_id"] == a_id].copy()
            if int(a["charge"].iloc[0]) != 1:
                continue

            for j in range(i + 1, len(series_ids)):
                b_id = series_ids[j]
                b = ru[ru["series_id"] == b_id].copy()
                if int(b["charge"].iloc[0]) != 1:
                    continue

                for adduct_a in selected_adducts:
                    for adduct_b in selected_adducts:
                        if adduct_a == adduct_b:
                            continue

                        info_a = ADDUCT_LIBRARY[adduct_a]
                        info_b = ADDUCT_LIBRARY[adduct_b]
                        if info_a["polarity"] != info_b["polarity"]:
                            continue

                        expected_delta = info_b["shift"] - info_a["shift"]
                        if expected_delta <= 0:
                            continue

                        matches = []
                        for _, row_a in a.iterrows():
                            expected_b_mz = row_a["mz"] + expected_delta
                            candidates = b.copy()
                            candidates["abs_ppm"] = np.abs(
                                (candidates["mz"] - expected_b_mz)
                                / expected_b_mz
                                * 1e6
                            )
                            candidates = candidates[
                                candidates["abs_ppm"] <= ppm_tolerance
                            ]
                            if candidates.empty:
                                continue

                            if pd.notna(row_a["rt"]) and candidates["rt"].notna().any():
                                candidates = candidates[
                                    np.abs(candidates["rt"] - row_a["rt"])
                                    <= rt_tolerance
                                ]
                            if candidates.empty:
                                continue

                            best = candidates.sort_values("abs_ppm").iloc[0]
                            matches.append({
                                "repeat_unit": repeat_unit,
                                "series_a": a_id,
                                "candidate_adduct_a": adduct_a,
                                "series_b": b_id,
                                "candidate_adduct_b": adduct_b,
                                "feature_a": int(row_a["feature_id"]),
                                "feature_b": int(best["feature_id"]),
                                "mz_a": float(row_a["mz"]),
                                "mz_b": float(best["mz"]),
                                "rt_a": float(row_a["rt"]) if pd.notna(row_a["rt"]) else np.nan,
                                "rt_b": float(best["rt"]) if pd.notna(best["rt"]) else np.nan,
                                "repeat_index_a": int(row_a["repeat_index"]),
                                "repeat_index_b": int(best["repeat_index"]),
                                "delta_mz": float(best["mz"] - row_a["mz"]),
                                "ppm_error": float(
                                    ppm_error(best["mz"] - row_a["mz"], expected_delta)
                                ),
                                "rt_difference": (
                                    float(abs(best["rt"] - row_a["rt"]))
                                    if pd.notna(best["rt"]) and pd.notna(row_a["rt"])
                                    else np.nan
                                ),
                            })

                        unique_pairs = {(m["feature_a"], m["feature_b"]) for m in matches}
                        if len(unique_pairs) < min_matches:
                            continue

                        key = (repeat_unit, a_id, adduct_a, b_id, adduct_b)
                        if key in seen:
                            continue
                        seen.add(key)

                        link_rows.append({
                            "repeat_unit": repeat_unit,
                            "series_a": a_id,
                            "candidate_adduct_a": adduct_a,
                            "series_b": b_id,
                            "candidate_adduct_b": adduct_b,
                            "matched_members": len(unique_pairs),
                            "expected_delta_mz": expected_delta,
                            "median_observed_delta_mz": np.median(
                                [m["delta_mz"] for m in matches]
                            ),
                            "median_abs_ppm": np.median(
                                [abs(m["ppm_error"]) for m in matches]
                            ),
                            "median_rt_difference": np.nanmedian(
                                [m["rt_difference"] for m in matches]
                            ),
                        })
                        member_rows.extend(matches)

    return pd.DataFrame(link_rows), pd.DataFrame(member_rows)


def build_series_adduct_map(links: pd.DataFrame):
    candidates = defaultdict(set)
    if links.empty:
        return {}

    for _, row in links.iterrows():
        candidates[row["series_a"]].add(row["candidate_adduct_a"])
        candidates[row["series_b"]].add(row["candidate_adduct_b"])

    output = {}
    for series_id, labels in candidates.items():
        if len(labels) == 1:
            output[series_id] = next(iter(labels))
        else:
            output[series_id] = "Ambiguous: " + " / ".join(sorted(labels))
    return output


def build_family_groups(series_ids, links: pd.DataFrame):
    parent = {sid: sid for sid in series_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if not links.empty:
        for _, row in links.iterrows():
            if row["series_a"] in parent and row["series_b"] in parent:
                union(row["series_a"], row["series_b"])

    roots = []
    for sid in series_ids:
        root = find(sid)
        if root not in roots:
            roots.append(root)

    root_to_group = {root: f"Family {i + 1}" for i, root in enumerate(roots)}
    return {sid: root_to_group[find(sid)] for sid in series_ids}


def annotate_summary_with_adducts(summary, links):
    out = summary.copy()
    out["adduct_evidence"] = "No companion adduct series identified"
    if links.empty:
        return out

    evidence_map = defaultdict(list)
    for _, row in links.iterrows():
        evidence_map[row["series_a"]].append(
            f'{row["candidate_adduct_a"]} linked to '
            f'{row["series_b"]} ({row["candidate_adduct_b"]})'
        )
        evidence_map[row["series_b"]].append(
            f'{row["candidate_adduct_b"]} linked to '
            f'{row["series_a"]} ({row["candidate_adduct_a"]})'
        )

    for idx, row in out.iterrows():
        if row["series_id"] in evidence_map:
            out.loc[idx, "adduct_evidence"] = "; ".join(
                evidence_map[row["series_id"]]
            )
    return out


def linked_series_ids(selected_series, links: pd.DataFrame):
    linked = set()
    if not selected_series or links.empty:
        return linked

    for _, row in links.iterrows():
        if row["series_a"] == selected_series:
            linked.add(row["series_b"])
        elif row["series_b"] == selected_series:
            linked.add(row["series_a"])
    return linked


def linked_relationship_table(selected_series, links: pd.DataFrame):
    if not selected_series or links.empty:
        return pd.DataFrame()

    rows = []
    for _, row in links.iterrows():
        if row["series_a"] == selected_series:
            rows.append({
                "Selected adduct": row["candidate_adduct_a"],
                "Linked series": row["series_b"],
                "Linked adduct": row["candidate_adduct_b"],
                "Matched members": row["matched_members"],
                "Median |ppm|": row["median_abs_ppm"],
                "Median RT difference": row["median_rt_difference"],
            })
        elif row["series_b"] == selected_series:
            rows.append({
                "Selected adduct": row["candidate_adduct_b"],
                "Linked series": row["series_a"],
                "Linked adduct": row["candidate_adduct_a"],
                "Matched members": row["matched_members"],
                "Median |ppm|": row["median_abs_ppm"],
                "Median RT difference": row["median_rt_difference"],
            })
    return pd.DataFrame(rows)


def candidate_formulae(
    target_mass: float,
    allowed_elements,
    ppm_tolerance: float = 5.0,
    max_results: int = 5,
):
    """Generate plausible elemental formulas for an unknown repeat mass."""
    allowed = [e for e in allowed_elements if e in MONOISOTOPIC_MASS]
    if not allowed or target_mass <= 0:
        return []

    use_h = "H" in allowed
    heavy = [e for e in allowed if e != "H"]
    caps = {
        "C": min(45, int(target_mass / MONOISOTOPIC_MASS["C"]) + 1),
        "N": 8,
        "O": 12,
        "F": 8,
        "Si": 5,
        "P": 4,
        "S": 5,
        "Cl": 4,
        "Br": 2,
        "I": 1,
    }

    # Put carbon first, then the lighter/common heteroatoms. H is solved from residual mass.
    order = [e for e in ["C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"] if e in heavy]
    results = []
    tolerance_da = target_mass * ppm_tolerance * 1e-6

    def recurse(pos, composition, current_mass, hetero_count):
        if current_mass > target_mass + tolerance_da:
            return

        if pos == len(order):
            comp = dict(composition)
            if use_h:
                residual = target_mass - current_mass
                h_count = int(round(residual / MONOISOTOPIC_MASS["H"]))
                if h_count < 0:
                    return
                comp["H"] = h_count
                calc_mass = current_mass + h_count * MONOISOTOPIC_MASS["H"]
            else:
                calc_mass = current_mass

            error_ppm = (calc_mass - target_mass) / target_mass * 1e6
            if abs(error_ppm) > ppm_tolerance:
                return

            c = comp.get("C", 0)
            si = comp.get("Si", 0)
            n = comp.get("N", 0)
            p = comp.get("P", 0)
            x = comp.get("F", 0) + comp.get("Cl", 0) + comp.get("Br", 0) + comp.get("I", 0)
            h = comp.get("H", 0)

            # Loose valence sanity check. Repeat units can be unusual, so avoid over-filtering.
            h_max = 2 * (c + si) + 2 + n + p - x + 6
            if use_h and h > max(h_max, 4):
                return

            dbe = 1 + c + si - h / 2 + n / 2 + p / 2 - x / 2
            if dbe < -1.0:
                return

            parts = []
            for element in ["C", "H"] + sorted(e for e in comp if e not in {"C", "H"}):
                count = comp.get(element, 0)
                if count <= 0:
                    continue
                parts.append(element if count == 1 else f"{element}{count}")
            formula = "".join(parts)
            if not formula:
                return

            results.append({
                "formula": formula,
                "calculated_mass": calc_mass,
                "ppm_error": error_ppm,
                "dbe": dbe,
                "atom_count": sum(comp.values()),
            })
            return

        element = order[pos]
        element_mass = MONOISOTOPIC_MASS[element]
        max_by_mass = int((target_mass + tolerance_da - current_mass) / element_mass)
        cap = min(caps.get(element, max_by_mass), max_by_mass)

        for count in range(cap + 1):
            new_hetero = hetero_count
            if element != "C":
                new_hetero += count
            if new_hetero > 10:
                break

            if count:
                composition[element] = count
            elif element in composition:
                composition.pop(element)

            recurse(
                pos + 1,
                composition,
                current_mass + count * element_mass,
                new_hetero,
            )

        composition.pop(element, None)

    recurse(0, {}, 0.0, 0)

    # Remove duplicate formula strings and sort by mass accuracy, then simplicity.
    best = {}
    for row in results:
        formula = row["formula"]
        if formula not in best or abs(row["ppm_error"]) < abs(best[formula]["ppm_error"]):
            best[formula] = row

    ordered = sorted(
        best.values(),
        key=lambda r: (abs(r["ppm_error"]), r["atom_count"], abs(r["dbe"])),
    )
    return ordered[:max_results]


def known_adduct_mass_differences(selected_adducts):
    diffs = []
    for i, a in enumerate(selected_adducts):
        for b in selected_adducts[i + 1:]:
            ia = ADDUCT_LIBRARY[a]
            ib = ADDUCT_LIBRARY[b]
            if ia["polarity"] != ib["polarity"]:
                continue
            diffs.append(abs(ib["shift"] - ia["shift"]))
    return diffs


def discover_repeat_masses(
    features: pd.DataFrame,
    min_repeat_mass: float,
    max_repeat_mass: float,
    bin_width: float,
    min_direct_pairs: int,
    max_candidates: int,
    ppm_tolerance: float,
    min_members: int,
    max_repeat_gap: int,
    rt_tolerance,
    charges,
    selected_adducts,
    targeted_units,
    formula_elements,
    formula_ppm_tolerance: float,
    max_formula_candidates: int,
):
    """Discover, validate, score, and interpret recurring unknown repeat masses."""
    df = features.sort_values("mz").reset_index(drop=True).copy()
    mz = df["mz"].to_numpy()
    rt = df["rt"].to_numpy()

    if len(df) < 2:
        return pd.DataFrame(), pd.DataFrame()

    bins = defaultdict(list)
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            delta = mz[j] - mz[i]
            if delta > max_repeat_mass:
                break
            if delta < min_repeat_mass:
                continue

            if rt_tolerance is not None and pd.notna(rt[i]) and pd.notna(rt[j]):
                if abs(rt[j] - rt[i]) > rt_tolerance:
                    continue

            bin_id = int(round(delta / bin_width))
            bins[bin_id].append(delta)

    raw_rows = []
    for bin_id, values in bins.items():
        if len(values) < min_direct_pairs:
            continue
        candidate_mass = float(np.median(values))
        if not (min_repeat_mass <= candidate_mass <= max_repeat_mass):
            continue
        raw_rows.append({
            "candidate_mass": candidate_mass,
            "direct_pair_support": len(values),
            "bin_center": bin_id * bin_width,
            "spread_da": float(np.ptp(values)) if len(values) > 1 else 0.0,
        })

    if not raw_rows:
        return pd.DataFrame(), pd.DataFrame()

    raw = pd.DataFrame(raw_rows).sort_values(
        ["direct_pair_support", "candidate_mass"], ascending=[False, True]
    )
    max_support = max(float(raw["direct_pair_support"].max()), 1.0)

    # Merge adjacent bins representing the same candidate.
    deduped = []
    for _, row in raw.iterrows():
        mass = float(row["candidate_mass"])
        if any(
            abs(mass - ex["candidate_mass"])
            <= max(bin_width * 1.5, mass * ppm_tolerance * 1e-6 * 3)
            for ex in deduped
        ):
            continue
        deduped.append(row.to_dict())

    adduct_diffs = known_adduct_mass_differences(selected_adducts)
    targeted_masses = [float(u["mass"]) for u in targeted_units]
    validated = []
    results_by_mass = {}

    for candidate in deduped[: max(40, max_candidates * 8)]:
        mass = float(candidate["candidate_mass"])
        label = f"Discovered {mass:.5f} Da"
        unit = {
            "label": label,
            "formula": "",
            "mass": mass,
            "description": "Automatically discovered recurring mass shift",
        }
        result = run_multi_repeat_search(
            features=features,
            search_units=[unit],
            charges=charges,
            ppm_tolerance=ppm_tolerance,
            min_members=min_members,
            max_repeat_gap=max_repeat_gap,
            rt_tolerance=rt_tolerance,
            search_origin="Discovered",
        )
        if result.empty:
            continue

        metrics = [series_shape_metrics(sdf) for _, sdf in result.groupby("series_id")]
        median_continuity = float(np.median([m["continuity_fraction"] for m in metrics]))
        med_ppm_values = [m["median_abs_ppm"] for m in metrics if not np.isnan(m["median_abs_ppm"])]
        median_abs_ppm = float(np.median(med_ppm_values)) if med_ppm_values else np.nan
        series_count = int(result["series_id"].nunique())
        assigned_features = int(result["feature_id"].nunique())

        support_score = 25.0 * min(candidate["direct_pair_support"] / max_support, 1.0)
        connected_score = 20.0 * min(series_count / 2.0, 1.0)
        continuity_score = 25.0 * median_continuity
        mass_score = 20.0 * max(
            0.0,
            1.0 - (0.0 if np.isnan(median_abs_ppm) else median_abs_ppm) / max(ppm_tolerance, 1e-9),
        )
        feature_score = 10.0 * min(assigned_features / max(min_members * 2, 1), 1.0)
        score = support_score + connected_score + continuity_score + mass_score + feature_score

        adduct_artifact = any(
            abs(mass - diff) <= max(bin_width * 2, diff * ppm_tolerance * 1e-6 * 5)
            for diff in adduct_diffs
        )
        if adduct_artifact:
            score -= 25.0

        matches_targeted = None
        for unit_target in targeted_units:
            target_mass = float(unit_target["mass"])
            if abs(mass - target_mass) <= max(
                bin_width * 2, target_mass * ppm_tolerance * 1e-6 * 5
            ):
                matches_targeted = unit_target["label"]
                score -= 8.0
                break

        formulas = candidate_formulae(
            target_mass=mass,
            allowed_elements=formula_elements,
            ppm_tolerance=formula_ppm_tolerance,
            max_results=max_formula_candidates,
        )
        formula_text = "; ".join(
            f'{f["formula"]} ({f["ppm_error"]:+.2f} ppm)' for f in formulas
        )

        row = {
            **candidate,
            "candidate_series": series_count,
            "assigned_features": assigned_features,
            "median_continuity": median_continuity,
            "median_abs_ppm": median_abs_ppm,
            "adduct_difference_flag": adduct_artifact,
            "matches_targeted": matches_targeted or "",
            "candidate_formulas": formula_text,
            "discovery_score_raw": max(0.0, min(100.0, score)),
            "harmonic_of": "",
            "possible_subharmonic_of": "",
        }
        validated.append(row)
        results_by_mass[mass] = result

    if not validated:
        return pd.DataFrame(), pd.DataFrame()

    # Penalize likely harmonics and subharmonics using continuity + score.
    validated = sorted(validated, key=lambda r: r["candidate_mass"])
    for i, small in enumerate(validated):
        for j in range(i + 1, len(validated)):
            large = validated[j]
            ratio = large["candidate_mass"] / small["candidate_mass"]
            multiple = int(round(ratio))
            if multiple < 2 or multiple > max_repeat_gap:
                continue
            target = small["candidate_mass"] * multiple
            tol = max(bin_width * 2.5, target * ppm_tolerance * 1e-6 * 6)
            if abs(large["candidate_mass"] - target) > tol:
                continue

            if (
                small["discovery_score_raw"] >= large["discovery_score_raw"] - 5
                and small["median_continuity"] >= large["median_continuity"] - 0.10
            ):
                large["discovery_score_raw"] = max(0.0, large["discovery_score_raw"] - 20.0)
                large["harmonic_of"] = f'{small["candidate_mass"]:.5f}'
            elif large["median_continuity"] > small["median_continuity"] + 0.20:
                small["discovery_score_raw"] = max(0.0, small["discovery_score_raw"] - 12.0)
                small["possible_subharmonic_of"] = f'{large["candidate_mass"]:.5f}'

    candidate_table = pd.DataFrame(validated).rename(
        columns={"discovery_score_raw": "discovery_score"}
    )
    candidate_table = candidate_table.sort_values(
        ["discovery_score", "median_continuity", "direct_pair_support"],
        ascending=[False, False, False],
    ).head(max_candidates).reset_index(drop=True)

    retained_masses = list(candidate_table["candidate_mass"])
    retained_results = []
    for mass in retained_masses:
        result = results_by_mass.get(float(mass))
        if result is not None and not result.empty:
            retained_results.append(result)

    combined = pd.concat(retained_results, ignore_index=True) if retained_results else pd.DataFrame()
    return candidate_table, combined


def merge_targeted_and_discovered_results(targeted, discovered, targeted_units, ppm_tolerance):
    frames = []
    if targeted is not None and not targeted.empty:
        frames.append(targeted.copy())

    if discovered is not None and not discovered.empty:
        d = discovered.copy()
        targeted_masses = [float(u["mass"]) for u in targeted_units]
        if targeted_masses:
            keep = []
            for _, row in d.iterrows():
                mass = float(row["repeat_mass"])
                duplicate = any(
                    abs(mass - tm) <= tm * ppm_tolerance * 1e-6 * 5
                    for tm in targeted_masses
                )
                keep.append(not duplicate)
            d = d[np.array(keep, dtype=bool)]
        if not d.empty:
            frames.append(d)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_series_confidence(
    summary: pd.DataFrame,
    results: pd.DataFrame,
    adduct_links: pd.DataFrame,
    discovered_candidates: pd.DataFrame,
    ppm_tolerance: float,
    min_members: int,
):
    out = summary.copy()
    linked = set()
    if not adduct_links.empty:
        linked.update(adduct_links["series_a"].tolist())
        linked.update(adduct_links["series_b"].tolist())

    discovery_score_by_mass = {}
    if not discovered_candidates.empty:
        for _, row in discovered_candidates.iterrows():
            discovery_score_by_mass[float(row["candidate_mass"])] = float(row["discovery_score"])

    scores = []
    labels = []
    components = []

    for _, row in out.iterrows():
        member_score = 20.0 * min(row["members"] / max(min_members + 4, 1), 1.0)
        continuity_score = 20.0 * float(row["continuity_fraction"])
        median_ppm = row["median_abs_ppm"]
        ppm_ratio = 0.0 if pd.isna(median_ppm) else min(float(median_ppm) / max(ppm_tolerance, 1e-9), 1.0)
        mass_score = 25.0 * (1.0 - ppm_ratio)

        rt_corr = row["rt_rank_correlation"]
        rt_score = 7.5 if pd.isna(rt_corr) else 15.0 * min(abs(float(rt_corr)), 1.0)
        adduct_score = 10.0 if row["series_id"] in linked else 5.0

        if row.get("search_origin", "Targeted") == "Discovered":
            repeat_support = 5.0
            for mass, dscore in discovery_score_by_mass.items():
                if abs(float(row["repeat_mass"]) - mass) <= max(mass * ppm_tolerance * 1e-6 * 5, 0.01):
                    repeat_support = 10.0 * dscore / 100.0
                    break
        else:
            repeat_support = 10.0

        score = max(0.0, min(100.0, member_score + continuity_score + mass_score + rt_score + adduct_score + repeat_support))
        label = "High" if score >= 80 else "Medium" if score >= 60 else "Low"

        scores.append(score)
        labels.append(label)
        components.append(
            f"members {member_score:.0f}/20; continuity {continuity_score:.0f}/20; "
            f"mass {mass_score:.0f}/25; RT {rt_score:.0f}/15; "
            f"adduct {adduct_score:.0f}/10; repeat {repeat_support:.0f}/10"
        )

    out["confidence_score"] = scores
    out["confidence_label"] = labels
    out["confidence_components"] = components
    return out


def reconstruct_neutral_masses(results: pd.DataFrame, summary: pd.DataFrame):
    out = results.copy()
    adduct_map = summary.set_index("series_id")["candidate_adduct"].to_dict()
    family_map = summary.set_index("series_id")["family_group"].to_dict()
    conf_map = summary.set_index("series_id")["confidence_score"].to_dict()
    conf_label_map = summary.set_index("series_id")["confidence_label"].to_dict()

    out["candidate_adduct"] = out["series_id"].map(adduct_map).fillna("Unassigned")
    out["family_group"] = out["series_id"].map(family_map)
    out["confidence_score"] = out["series_id"].map(conf_map)
    out["confidence_label"] = out["series_id"].map(conf_label_map)

    neutral = []
    decharged = []
    for _, row in out.iterrows():
        z = int(row["charge"])
        ion_mass = float(row["mz"]) * abs(z)
        decharged.append(ion_mass)

        adduct = row["candidate_adduct"]
        if z == 1 and adduct in ADDUCT_LIBRARY:
            neutral.append(float(row["mz"]) - ADDUCT_LIBRARY[adduct]["shift"])
        else:
            neutral.append(np.nan)

    out["decharged_ion_mass"] = decharged
    out["neutral_mass"] = neutral
    return out


def end_group_analysis(series_df: pd.DataFrame):
    if series_df.empty or len(series_df) < 2:
        return None

    sdf = series_df.sort_values("repeat_index").copy()
    x = sdf["repeat_index"].to_numpy(dtype=float)

    if sdf["neutral_mass"].notna().sum() >= 2:
        valid = sdf["neutral_mass"].notna().to_numpy()
        y = sdf.loc[valid, "neutral_mass"].to_numpy(dtype=float)
        x_fit = x[valid]
        basis = "Neutral mass"
    else:
        y = sdf["decharged_ion_mass"].to_numpy(dtype=float)
        x_fit = x
        basis = "Decharged ion mass"

    if len(y) < 2 or np.std(x_fit) == 0:
        return None

    slope, intercept = np.polyfit(x_fit, y, 1)
    predicted = slope * x_fit + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    theoretical = float(sdf["repeat_mass"].iloc[0])
    slope_ppm = (slope - theoretical) / theoretical * 1e6
    residual = intercept % theoretical if theoretical > 0 else np.nan

    return {
        "basis": basis,
        "fitted_repeat_mass": slope,
        "repeat_mass_ppm_error": slope_ppm,
        "observed_index_zero_intercept": intercept,
        "residual_mod_repeat": residual,
        "r2": r2,
    }


def add_intercept_summary(summary: pd.DataFrame, results: pd.DataFrame):
    out = summary.copy()
    values = {}
    for series_id, sdf in results.groupby("series_id"):
        analysis = end_group_analysis(sdf)
        values[series_id] = analysis

    for column in [
        "fitted_repeat_mass",
        "repeat_fit_ppm_error",
        "observed_index_zero_intercept",
        "residual_mod_repeat",
        "repeat_fit_r2",
        "intercept_basis",
    ]:
        out[column] = np.nan if column != "intercept_basis" else ""

    for idx, row in out.iterrows():
        analysis = values.get(row["series_id"])
        if not analysis:
            continue
        out.loc[idx, "fitted_repeat_mass"] = analysis["fitted_repeat_mass"]
        out.loc[idx, "repeat_fit_ppm_error"] = analysis["repeat_mass_ppm_error"]
        out.loc[idx, "observed_index_zero_intercept"] = analysis["observed_index_zero_intercept"]
        out.loc[idx, "residual_mod_repeat"] = analysis["residual_mod_repeat"]
        out.loc[idx, "repeat_fit_r2"] = analysis["r2"]
        out.loc[idx, "intercept_basis"] = analysis["basis"]
    return out


def selected_series_from_plot_event(event, trace_series_map):
    if event is None:
        return None
    try:
        points = event.selection.points
    except Exception:
        try:
            points = event["selection"]["points"]
        except Exception:
            points = []
    if not points:
        return None

    point = points[0]
    curve_number = point.get("curve_number")
    if curve_number is None:
        curve_number = point.get("curveNumber")
    if curve_number is None:
        return None
    if 0 <= int(curve_number) < len(trace_series_map):
        return trace_series_map[int(curve_number)]
    return None


def make_all_series_figure(results, summary, selected_series, adduct_links, adduct_member_matches):
    use_rt = results["rt"].notna().all()
    family_groups = summary.set_index("series_id")["family_group"].to_dict()
    adduct_labels = summary.set_index("series_id")["candidate_adduct"].to_dict()
    confidence = summary.set_index("series_id")["confidence_score"].to_dict()

    families = list(dict.fromkeys(summary["family_group"].tolist()))
    palette = (
        px.colors.qualitative.Safe
        + px.colors.qualitative.Bold
        + px.colors.qualitative.Set2
        + px.colors.qualitative.Dark24
    )
    family_colors = {family: palette[i % len(palette)] for i, family in enumerate(families)}
    dash_map = {
        "[M+H]+": "solid",
        "[M+Na]+": "dash",
        "[M+K]+": "dot",
        "[M+NH4]+": "dashdot",
        "[M-H]-": "longdash",
    }
    linked_ids = linked_series_ids(selected_series, adduct_links)

    fig = go.Figure()
    trace_series_map = []

    for series_id in summary["series_id"]:
        sdf = results[results["series_id"] == series_id].sort_values("repeat_index").copy()
        family = family_groups.get(series_id, series_id)
        adduct = adduct_labels.get(series_id, "Unassigned")
        color = family_colors[family]
        score = confidence.get(series_id, np.nan)
        charge = int(sdf["charge"].iloc[0])

        is_selected = series_id == selected_series
        is_linked = series_id in linked_ids
        if is_selected:
            width, marker_size, opacity = 5, 10, 1.0
        elif is_linked:
            width, marker_size, opacity = 3.5, 8, 0.95
        else:
            width, marker_size, opacity = 1.8, 6, 0.38 if selected_series else 0.82

        dash = dash_map.get(adduct, "solid")
        if use_rt:
            x = sdf["rt"]
            x_title = "Retention time"
        else:
            x = sdf["repeat_index"]
            x_title = "Repeat index"

        customdata = np.column_stack([
            [series_id] * len(sdf),
            [family] * len(sdf),
            [adduct] * len(sdf),
            [charge] * len(sdf),
            [f"{score:.1f}" if pd.notna(score) else ""] * len(sdf),
            sdf["repeat_unit"].astype(str),
            sdf["repeat_formula"].astype(str),
            sdf["feature_id"].astype(int),
            sdf["repeat_index"].astype(int),
        ])

        fig.add_trace(go.Scatter(
            x=x,
            y=sdf["mz"],
            mode="lines+markers",
            name=f"{family} · z{charge} · {adduct}",
            customdata=customdata,
            line=dict(color=color, width=width, dash=dash),
            marker=dict(color=color, size=marker_size, line=dict(width=1, color="#FFFFFF")),
            opacity=opacity,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Family: %{customdata[1]}<br>"
                "Adduct: %{customdata[2]}<br>"
                "Charge: z%{customdata[3]}<br>"
                "Confidence: %{customdata[4]}<br>"
                "Repeat: %{customdata[5]} · %{customdata[6]}<br>"
                "Feature: %{customdata[7]}<br>"
                "Repeat index: %{customdata[8]}<br>"
                "m/z: %{y:.5f}<extra></extra>"
            ),
            showlegend=False,
        ))
        trace_series_map.append(series_id)

    if selected_series and not adduct_member_matches.empty and use_rt:
        connector_rows = adduct_member_matches[
            (adduct_member_matches["series_a"] == selected_series)
            | (adduct_member_matches["series_b"] == selected_series)
        ]
        for _, row in connector_rows.iterrows():
            family = family_groups.get(selected_series, selected_series)
            fig.add_trace(go.Scatter(
                x=[row["rt_a"], row["rt_b"]],
                y=[row["mz_a"], row["mz_b"]],
                mode="lines",
                line=dict(color=family_colors[family], width=1.1, dash="dot"),
                opacity=0.30,
                hoverinfo="skip",
                showlegend=False,
            ))

    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=25, r=20, t=30, b=25),
        xaxis_title=x_title,
        yaxis_title="m/z",
        hovermode="closest",
        clickmode="event+select",
        dragmode="select",
        selectionrevision="series-selection",
        uirevision="series-overview",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F5", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F5", zeroline=False)
    return fig, trace_series_map, use_rt


def make_neutral_family_figure(family_results: pd.DataFrame):
    neutral = family_results[family_results["neutral_mass"].notna()].copy()
    if neutral.empty:
        return None

    use_rt = neutral["rt"].notna().all()
    fig = go.Figure()
    for series_id, sdf in neutral.groupby("series_id"):
        sdf = sdf.sort_values("repeat_index")
        x = sdf["rt"] if use_rt else sdf["repeat_index"]
        adduct = sdf["candidate_adduct"].iloc[0]
        fig.add_trace(go.Scatter(
            x=x,
            y=sdf["neutral_mass"],
            mode="lines+markers",
            name=adduct,
            hovertemplate=(
                f"<b>{adduct}</b><br>"
                "Neutral mass: %{y:.5f}<br>"
                + ("RT: %{x:.3f}<br>" if use_rt else "Index: %{x}<br>")
                + "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=15, r=15, t=20, b=15),
        xaxis_title="Retention time" if use_rt else "Repeat index",
        yaxis_title="Reconstructed neutral mass",
        legend_title="Adduct",
    )
    return fig

def metric_card(label, value, subtext=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("### Analysis setup")
    st.caption("Configure the screen, then run or rerun without reloading your data.")

    with st.expander("1. Repeat units", expanded=True):
        st.caption("Select every repeat unit to search in the same run.")

        selected_builtin = []

        for short_name, info in REPEAT_LIBRARY.items():
            checked = st.checkbox(
                f'{short_name}  ·  {info["formula"]}',
                value=short_name in ["PEG / EO", "PPG / PO"],
                key=f"repeat_{short_name}",
                help=info["full_name"],
            )
            if checked:
                selected_builtin.append(short_name)

        st.markdown("---")

        use_custom = st.checkbox("Add custom repeat unit", value=False)

        custom_name = st.text_input(
            "Custom name",
            value="Custom",
            disabled=not use_custom,
        )

        custom_formula = st.text_input(
            "Custom formula",
            value="C2H4O",
            disabled=not use_custom,
        )

        custom_error = None
        custom_mass = None

        if use_custom:
            try:
                custom_mass = formula_mass(custom_formula)
                st.caption(f"Exact mass: {custom_mass:.6f} Da")
            except Exception as exc:
                custom_error = str(exc)
                st.error(custom_error)

        st.markdown("---")

        discover_any = st.checkbox(
            "Find any homologous series",
            value=False,
            help=(
                "Search the feature table for recurring mass differences even when "
                "no repeat unit is specified."
            ),
        )

        if discover_any:
            st.caption(
                "Discovery mode ranks recurring direct mass shifts, then validates "
                "each candidate by building connected homologous series."
            )

            discovery_min_mass = st.number_input(
                "Minimum discovered repeat mass (Da)",
                min_value=1.0,
                max_value=500.0,
                value=10.0,
                step=1.0,
            )
            discovery_max_mass = st.number_input(
                "Maximum discovered repeat mass (Da)",
                min_value=2.0,
                max_value=1000.0,
                value=250.0,
                step=5.0,
            )
            discovery_max_candidates = st.number_input(
                "Maximum discovered repeat units",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
            )
            discovery_min_pairs = st.number_input(
                "Minimum direct pair support",
                min_value=2,
                max_value=100,
                value=3,
                step=1,
            )
            discovery_bin_width = st.number_input(
                "Discovery mass bin width (Da)",
                min_value=0.0005,
                max_value=0.1000,
                value=0.0050,
                step=0.0005,
                format="%.4f",
            )

            st.markdown("##### Candidate repeat formulas")
            infer_formulas = st.checkbox(
                "Infer candidate formulas for discovered repeats",
                value=True,
            )
            formula_elements = st.multiselect(
                "Allowed elements",
                ["C", "H", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"],
                default=["C", "H", "N", "O", "Si"],
                disabled=not infer_formulas,
            )
            formula_ppm_tolerance = st.number_input(
                "Formula tolerance (ppm)",
                min_value=0.5,
                max_value=50.0,
                value=5.0,
                step=0.5,
                disabled=not infer_formulas,
            )
            max_formula_candidates = st.number_input(
                "Formula candidates per repeat",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                disabled=not infer_formulas,
            )
        else:
            discovery_min_mass = 10.0
            discovery_max_mass = 250.0
            discovery_max_candidates = 10
            discovery_min_pairs = 3
            discovery_bin_width = 0.005
            infer_formulas = True
            formula_elements = ["C", "H", "N", "O", "Si"]
            formula_ppm_tolerance = 5.0
            max_formula_candidates = 5

    with st.expander("2. Adducts", expanded=True):
        st.caption("All common adducts are included by default.")

        selected_adducts = []

        for adduct_name, info in ADDUCT_LIBRARY.items():
            checked = st.checkbox(
                f'{adduct_name}  ·  {info["description"]}',
                value=True,
                key=f"adduct_{adduct_name}",
            )
            if checked:
                selected_adducts.append(adduct_name)

    with st.expander("3. Search settings", expanded=False):
        auto_charge = st.checkbox(
            "Automatic charge detection",
            value=True,
            help=(
                "Search multiple charge-state hypotheses and keep the most plausible "
                "assignment when the same features can be explained more than one way."
            ),
        )

        if auto_charge:
            max_charge = st.number_input(
                "Maximum charge state",
                min_value=1,
                max_value=6,
                value=4,
                step=1,
            )
            charges = list(range(1, int(max_charge) + 1))
        else:
            manual_charge = st.selectbox(
                "Charge state",
                [1, 2, 3, 4, 5, 6],
                index=0,
            )
            charges = [int(manual_charge)]

        ppm_tolerance = st.number_input(
            "Repeat spacing tolerance (ppm)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.5,
        )

        min_members = st.number_input(
            "Minimum members in a series",
            min_value=2,
            max_value=50,
            value=4,
            step=1,
        )

        max_repeat_gap = st.number_input(
            "Maximum skipped repeat units",
            min_value=1,
            max_value=50,
            value=8,
            step=1,
        )

        use_rt = st.checkbox(
            "Apply RT proximity filter to series search",
            value=False,
        )

        rt_tolerance_value = st.number_input(
            "Max series RT difference",
            min_value=0.0,
            value=3.0,
            step=0.1,
            disabled=not use_rt,
        )

        rt_tolerance = rt_tolerance_value if use_rt else None

        st.markdown("##### Adduct matching")

        adduct_ppm_tolerance = st.number_input(
            "Adduct matching tolerance (ppm)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.5,
        )

        adduct_rt_tolerance = st.number_input(
            "Adduct RT matching tolerance",
            min_value=0.0,
            value=0.20,
            step=0.05,
        )

    with st.expander("Reference libraries", expanded=False):
        st.markdown("**Repeat units**")
        for short_name, info in REPEAT_LIBRARY.items():
            st.caption(
                f'{short_name}: {info["formula"]} · '
                f'{formula_mass(info["formula"]):.6f} Da'
            )

        st.markdown("**Adducts**")
        for adduct_name, info in ADDUCT_LIBRARY.items():
            st.caption(
                f'{adduct_name}: {info["description"]} · '
                f'{info["polarity"]}'
            )


# ------------------------------------------------------------
# Data input
# ------------------------------------------------------------

upload_col, info_col = st.columns([1.5, 1])

with upload_col:
    st.markdown("### Data")
    uploaded = st.file_uploader(
        "Upload LCMS feature table",
        type=["csv"],
        key="feature_file",
        help="CSV with at minimum an m/z or mz column. RT and intensity are recommended.",
    )

with info_col:
    st.markdown("### Expected input")
    st.markdown(
        """
        <div class="section-card">
            <div class="small-note">
                <b>Required:</b> m/z<br>
                <b>Recommended:</b> RT, intensity<br><br>
                Common column aliases such as <i>retention_time</i>, <i>area</i>,
                <i>height</i>, and <i>abundance</i> are recognized automatically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded is None:
    st.info(
        "Upload a CSV feature table to begin. The included example file contains "
        "multiple polymer families and adduct forms."
    )
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    features = normalize_columns(raw)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()


# ------------------------------------------------------------
# Search configuration summary
# ------------------------------------------------------------

search_units = []

for short_name in selected_builtin:
    info = REPEAT_LIBRARY[short_name]
    search_units.append({
        "label": short_name,
        "formula": info["formula"],
        "mass": formula_mass(info["formula"]),
        "description": info["full_name"],
    })

if use_custom and custom_error is None and custom_mass is not None:
    custom_label = custom_name.strip() if custom_name.strip() else "Custom"
    search_units.append({
        "label": custom_label,
        "formula": custom_formula.strip(),
        "mass": custom_mass,
        "description": "User defined repeat unit",
    })


st.markdown("### Search summary")

summary_cols = st.columns(4)
with summary_cols[0]:
    metric_card("Features loaded", f"{len(features):,}", "Input feature table")
with summary_cols[1]:
    metric_card(
        "m/z range",
        f"{features.mz.min():.1f}–{features.mz.max():.1f}",
        "Observed features",
    )
with summary_cols[2]:
    metric_card(
        "Repeat search",
        len(search_units) + (1 if discover_any else 0),
        f"{len(search_units)} targeted" + (" + discovery" if discover_any else ""),
    )
with summary_cols[3]:
    metric_card(
        "Adducts",
        len(selected_adducts),
        "Selected for evaluation",
    )

st.write("")

repeat_badge_parts = []
if search_units:
    repeat_badge_parts.extend(
        f'<span class="muted-pill">{unit["label"]} · {unit["formula"]}</span>'
        for unit in search_units
    )
if discover_any:
    repeat_badge_parts.append('<span class="status-pill">Find any homologous series</span>')
repeat_badges = " ".join(repeat_badge_parts) if repeat_badge_parts else '<span class="muted-pill">No repeat search selected</span>'

if selected_adducts:
    adduct_badges = " ".join(
        f'<span class="muted-pill">{name}</span>'
        for name in selected_adducts
    )
else:
    adduct_badges = '<span class="muted-pill">No adducts selected</span>'

st.markdown(
    f"""
    <div class="section-card">
        <div class="metric-label">Active repeat units</div>
        <div>{repeat_badges}</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Active adducts</div>
        <div>{adduct_badges}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

button_label = (
    "Rerun analysis"
    if "search_results" in st.session_state
    else "Run analysis"
)

run_search = st.button(
    button_label,
    type="primary",
    use_container_width=True,
    disabled=(
        (len(search_units) == 0 and not discover_any)
        or len(selected_adducts) == 0
        or custom_error is not None
        or (discover_any and discovery_min_mass >= discovery_max_mass)
    ),
)

if run_search:
    with st.spinner("Running targeted and discovery homologous-series screening..."):
        targeted_results = run_multi_repeat_search(
            features=features,
            search_units=search_units,
            charges=charges,
            ppm_tolerance=ppm_tolerance,
            min_members=int(min_members),
            max_repeat_gap=int(max_repeat_gap),
            rt_tolerance=rt_tolerance,
            search_origin="Targeted",
        )

        discovered_candidates = pd.DataFrame()
        discovered_results = pd.DataFrame()

        if discover_any:
            discovered_candidates, discovered_results = discover_repeat_masses(
                features=features,
                min_repeat_mass=float(discovery_min_mass),
                max_repeat_mass=float(discovery_max_mass),
                bin_width=float(discovery_bin_width),
                min_direct_pairs=int(discovery_min_pairs),
                max_candidates=int(discovery_max_candidates),
                ppm_tolerance=float(ppm_tolerance),
                min_members=int(min_members),
                max_repeat_gap=int(max_repeat_gap),
                rt_tolerance=rt_tolerance,
                charges=charges,
                selected_adducts=selected_adducts,
                targeted_units=search_units,
                formula_elements=(formula_elements if infer_formulas else []),
                formula_ppm_tolerance=float(formula_ppm_tolerance),
                max_formula_candidates=int(max_formula_candidates),
            )

        results = merge_targeted_and_discovered_results(
            targeted_results,
            discovered_results,
            targeted_units=search_units,
            ppm_tolerance=float(ppm_tolerance),
        )

        adduct_links, adduct_member_matches = find_adduct_series_links(
            results=results,
            selected_adducts=selected_adducts,
            ppm_tolerance=adduct_ppm_tolerance,
            rt_tolerance=adduct_rt_tolerance,
            min_matches=max(2, min(3, int(min_members))),
        )

    st.session_state["search_results"] = results
    st.session_state["discovered_candidates"] = discovered_candidates
    st.session_state["adduct_links"] = adduct_links
    st.session_state["adduct_member_matches"] = adduct_member_matches
    st.session_state["last_search_units"] = search_units
    st.session_state["last_discover_any"] = discover_any
    st.session_state["last_selected_adducts"] = selected_adducts
    st.session_state["last_charges"] = charges
    st.session_state["last_ppm_tolerance"] = float(ppm_tolerance)
    st.session_state["last_min_members"] = int(min_members)
    st.session_state["selected_series_id"] = None
    st.session_state["search_run_counter"] = (
        st.session_state.get("search_run_counter", 0) + 1
    )

if "search_results" not in st.session_state:
    st.stop()

results = st.session_state["search_results"]
adduct_links = st.session_state.get("adduct_links", pd.DataFrame())
adduct_member_matches = st.session_state.get(
    "adduct_member_matches",
    pd.DataFrame(),
)
discovered_candidates = st.session_state.get(
    "discovered_candidates",
    pd.DataFrame(),
)
last_search_units = st.session_state.get("last_search_units", [])
last_discover_any = st.session_state.get("last_discover_any", False)
last_selected_adducts = st.session_state.get("last_selected_adducts", [])
last_charges = st.session_state.get("last_charges", [1])
last_ppm_tolerance = st.session_state.get("last_ppm_tolerance", 5.0)
last_min_members = st.session_state.get("last_min_members", 4)
run_number = st.session_state.get("search_run_counter", 1)

if results.empty:
    st.warning("No candidate homologous series met the current criteria.")
    st.stop()

summary = build_summary(results)
summary = annotate_summary_with_adducts(summary, adduct_links)
series_ids = summary["series_id"].tolist()

series_adduct_map = build_series_adduct_map(adduct_links)
family_map = build_family_groups(series_ids, adduct_links)

summary["candidate_adduct"] = summary["series_id"].map(
    lambda sid: series_adduct_map.get(sid, "Unassigned")
)
summary["family_group"] = summary["series_id"].map(family_map)
series_origin_map = (
    results[["series_id", "search_origin"]]
    .drop_duplicates()
    .set_index("series_id")["search_origin"]
    .to_dict()
)
summary["search_origin"] = summary["series_id"].map(series_origin_map)

summary = add_series_confidence(
    summary=summary,
    results=results,
    adduct_links=adduct_links,
    discovered_candidates=discovered_candidates,
    ppm_tolerance=last_ppm_tolerance,
    min_members=last_min_members,
)
results = reconstruct_neutral_masses(results, summary)
summary = add_intercept_summary(summary, results)

if (
    st.session_state.get("selected_series_id") not in series_ids
):
    st.session_state["selected_series_id"] = series_ids[0]

st.markdown(
    f'<span class="status-pill">Analysis complete · Run #{run_number}</span>',
    unsafe_allow_html=True,
)
st.write("")


# ------------------------------------------------------------
# Main result tabs
# ------------------------------------------------------------

tab_overview, tab_series, tab_adducts, tab_export = st.tabs(
    ["Overview", "Interactive Series Map", "Adduct Relationships", "Results & Export"]
)

with tab_overview:
    st.markdown("### Analysis overview")

    result_cols = st.columns(5)
    with result_cols[0]:
        metric_card(
            "Repeat search",
            len(last_search_units) + (1 if last_discover_any else 0),
            f"{len(last_search_units)} targeted" + (" + discovery" if last_discover_any else ""),
        )
    with result_cols[1]:
        metric_card(
            "Candidate series",
            len(series_ids),
            "Across all repeat units",
        )
    with result_cols[2]:
        metric_card(
            "High confidence",
            int((summary["confidence_label"] == "High").sum()),
            "Score ≥ 80",
        )
    with result_cols[3]:
        metric_card(
            "Charge states",
            ", ".join(f"z{z}" for z in sorted(summary["charge"].unique())),
            "Detected hypotheses",
        )
    with result_cols[4]:
        metric_card(
            "Adduct relationships",
            0 if adduct_links.empty else len(adduct_links),
            "Evidence supported links",
        )

    st.write("")
    st.markdown("#### Detection by repeat unit")

    repeat_summary = (
        results.groupby(["search_origin", "repeat_unit", "repeat_formula"], as_index=False)
        .agg(
            candidate_series=("series_id", "nunique"),
            assigned_feature_hits=("feature_id", "count"),
        )
    )

    st.dataframe(
        repeat_summary,
        use_container_width=True,
        hide_index=True,
    )

    if last_discover_any:
        st.markdown("#### Automatically discovered repeat masses")
        if discovered_candidates.empty:
            st.info(
                "Discovery mode did not find a recurring mass shift that produced "
                "a qualifying homologous series under the current criteria."
            )
        else:
            discovery_display = discovered_candidates.rename(
                columns={
                    "candidate_mass": "Repeat mass (Da)",
                    "discovery_score": "Discovery score",
                    "direct_pair_support": "Direct pair support",
                    "candidate_series": "Series found",
                    "assigned_features": "Assigned features",
                    "median_continuity": "Continuity",
                    "median_abs_ppm": "Median |ppm|",
                    "candidate_formulas": "Candidate formulas",
                    "harmonic_of": "Harmonic of",
                    "possible_subharmonic_of": "Possible subharmonic of",
                    "adduct_difference_flag": "Adduct-difference flag",
                    "matches_targeted": "Matches targeted",
                }
            )
            st.dataframe(
                discovery_display[
                    [
                        "Repeat mass (Da)",
                        "Discovery score",
                        "Candidate formulas",
                        "Direct pair support",
                        "Series found",
                        "Assigned features",
                        "Continuity",
                        "Median |ppm|",
                        "Harmonic of",
                        "Possible subharmonic of",
                        "Adduct-difference flag",
                        "Matches targeted",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("#### All candidate series")
    overview_cols = [
        "confidence_score",
        "confidence_label",
        "search_origin",
        "family_group",
        "repeat_unit",
        "repeat_formula",
        "charge",
        "series_id",
        "candidate_adduct",
        "members",
        "continuity_fraction",
        "missing_repeats",
        "median_abs_ppm",
        "rt_rank_correlation",
        "fitted_repeat_mass",
        "repeat_fit_ppm_error",
        "repeat_fit_r2",
        "residual_mod_repeat",
        "adduct_evidence",
    ]
    st.dataframe(
        summary[overview_cols],
        use_container_width=True,
        hide_index=True,
    )


with tab_series:
    st.markdown("### Interactive series map")
    st.caption(
        "Every connected line is one candidate homologous series. Automatic charge detection, "
        "confidence scoring, linked adduct families, and neutral reconstruction are shown together."
    )

    selected_series = st.session_state["selected_series_id"]
    left_panel, right_panel = st.columns([2.25, 1], gap="large")

    with left_panel:
        st.markdown("#### All detected homologous series")
        st.markdown(
            """
            <div class="section-card">
                <div class="small-note">
                    Click a marker to select the complete series. One line represents one charge/adduct
                    hypothesis. Linked z=1 adduct forms share a family color and remain separate lines;
                    dotted connectors identify coeluting members used as adduct evidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_all, trace_series_map, using_rt = make_all_series_figure(
            results=results,
            summary=summary,
            selected_series=selected_series,
            adduct_links=adduct_links,
            adduct_member_matches=adduct_member_matches,
        )

        plot_event = st.plotly_chart(
            fig_all,
            use_container_width=True,
            key=f"all_series_map_{run_number}",
            on_select="rerun",
            selection_mode="points",
            config={"displaylogo": False, "scrollZoom": True},
        )

        clicked_series = selected_series_from_plot_event(plot_event, trace_series_map)
        if clicked_series and clicked_series != st.session_state["selected_series_id"]:
            st.session_state["selected_series_id"] = clicked_series
            st.rerun()

        st.caption(
            "X axis = retention time." if using_rt
            else "Retention time was incomplete, so repeat index is used on the X axis."
        )

    with right_panel:
        selected_series = st.session_state["selected_series_id"]
        direct_series = st.selectbox(
            "Selected series",
            series_ids,
            index=series_ids.index(selected_series),
            key="direct_series_picker",
        )
        if direct_series != selected_series:
            st.session_state["selected_series_id"] = direct_series
            st.rerun()

        selected_series = st.session_state["selected_series_id"]
        selected = results[results["series_id"] == selected_series].sort_values("repeat_index").copy()
        selected_summary = summary[summary["series_id"] == selected_series].iloc[0]
        family_id = selected_summary["family_group"]
        candidate_adduct = selected_summary["candidate_adduct"]
        selected_origin = selected_summary["search_origin"]
        charge_value = int(selected_summary["charge"])

        st.markdown(
            f"""
            <div class="section-card">
                <div class="eyebrow">Selected homologous series</div>
                <div style="font-size:1.08rem;font-weight:750;color:#10243E;">{selected_series}</div>
                <div style="height:10px;"></div>
                <span class="status-pill">{selected_summary['confidence_label']} · {selected_summary['confidence_score']:.0f}/100</span>
                <span class="muted-pill">{family_id}</span>
                <span class="muted-pill">z{charge_value}</span>
                <span class="muted-pill">{candidate_adduct}</span>
                <span class="muted-pill">{selected_origin}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        detail_cols = st.columns(2)
        with detail_cols[0]:
            metric_card("Repeat unit", selected_summary["repeat_unit"], selected_summary["repeat_formula"] or "Unknown formula")
        with detail_cols[1]:
            metric_card("Members", int(selected_summary["members"]), f"Continuity {selected_summary['continuity_fraction']:.0%}")

        st.write("")
        with st.expander("Confidence score", expanded=False):
            st.write(selected_summary["confidence_components"])
            st.caption(
                "Confidence combines series size, consecutive homologs, mass accuracy, RT trend, "
                "adduct support, and whether the repeat was targeted or discovered."
            )

        if selected_origin == "Discovered" and not discovered_candidates.empty:
            candidate_row = discovered_candidates.iloc[
                np.argmin(np.abs(discovered_candidates["candidate_mass"] - selected_summary["repeat_mass"]))
            ]
            st.markdown("#### Unknown repeat interpretation")
            st.write(f"**Repeat mass:** {selected_summary['repeat_mass']:.6f} Da")
            if candidate_row.get("candidate_formulas", ""):
                st.write(f"**Candidate formulas:** {candidate_row['candidate_formulas']}")
            st.caption(f"Discovery score: {candidate_row['discovery_score']:.1f}/100")

        st.markdown("#### Linked adduct forms")
        selected_links = linked_relationship_table(selected_series, adduct_links)
        if selected_links.empty:
            if charge_value > 1:
                st.caption("Adduct deconvolution is currently restricted to supported z=1 H, Na, K, NH4, and deprotonated ion forms.")
            else:
                st.caption("No supported companion adduct series were detected for this line.")
        else:
            st.dataframe(selected_links, use_container_width=True, hide_index=True)

        family_series_ids = summary[summary["family_group"] == family_id]["series_id"].tolist()
        family_results = results[results["series_id"].isin(family_series_ids)].copy()

        neutral_fig = make_neutral_family_figure(family_results)
        if neutral_fig is not None:
            st.markdown("#### Neutral family reconstruction")
            st.plotly_chart(neutral_fig, use_container_width=True, config={"displaylogo": False})
            st.caption(
                "Companion adduct traces are converted back to neutral mass. Correctly linked ion forms should collapse onto the same neutral oligomer pattern."
            )

        st.markdown("#### Repeat / end group fit")
        intercept = end_group_analysis(selected)
        if intercept:
            fit_cols = st.columns(2)
            with fit_cols[0]:
                st.metric("Fitted repeat", f"{intercept['fitted_repeat_mass']:.6f} Da", f"{intercept['repeat_mass_ppm_error']:+.2f} ppm")
            with fit_cols[1]:
                st.metric("Fit R²", f"{intercept['r2']:.5f}")
            st.write(f"**Index 0 intercept:** {intercept['observed_index_zero_intercept']:.5f} Da")
            st.write(f"**Residual modulo repeat:** {intercept['residual_mod_repeat']:.5f} Da")
            st.caption(
                "The residual modulo the repeat is an end-group/intercept clue, not an absolute end-group assignment. Absolute end-group mass requires an anchored degree of polymerization."
            )

        st.markdown("#### Series members")
        member_cols = [
            "feature_id", "mz", "rt", "repeat_index", "intensity",
            "series_ppm_error", "decharged_ion_mass", "neutral_mass",
        ]
        st.dataframe(selected[member_cols], use_container_width=True, hide_index=True, height=270)

        st.markdown("#### Download")
        st.download_button(
            "Download selected series",
            data=selected.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_selected.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download linked adduct family",
            data=family_results.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_selected_family.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download all series",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_all_assignments.csv",
            mime="text/csv",
            use_container_width=True,
        )


with tab_adducts:
    st.markdown("### Adduct relationships")

    st.markdown(
        """
        <div class="section-card">
            <div class="small-note">
                Companion z=1 series are linked only when multiple homologous members show
                the expected exact adduct mass offset and similar retention times.
                Linked adducts are then reconstructed to neutral mass and collapsed into a shared family.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not adduct_links.empty:
        st.dataframe(
            adduct_links,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Matched feature pairs used as adduct evidence"):
            st.dataframe(
                adduct_member_matches,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "No companion adduct series met the current evidence criteria. "
            "A homologous series may still be valid when only one ion form is observed."
        )

    st.markdown("#### Adducts evaluated in last run")
    adduct_table = pd.DataFrame([
        {
            "Adduct": name,
            "Description": ADDUCT_LIBRARY[name]["description"],
            "Polarity": ADDUCT_LIBRARY[name]["polarity"],
            "m/z shift (z=1)": ADDUCT_LIBRARY[name]["shift"],
        }
        for name in last_selected_adducts
    ])

    st.dataframe(
        adduct_table,
        use_container_width=True,
        hide_index=True,
    )


with tab_export:
    st.markdown("### Results & export")

    st.markdown(
        """
        <div class="section-card">
            <div class="small-note">
                Export the complete feature assignments for downstream review.
                Adduct relationships are exported separately so the evidence layer
                remains distinct from the core homologous series assignments.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Complete candidate series table")
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    current_selected_series = st.session_state.get(
        "selected_series_id",
        series_ids[0],
    )
    export_selected = results[
        results["series_id"] == current_selected_series
    ].copy()

    export_family_id = summary.loc[
        summary["series_id"] == current_selected_series,
        "family_group",
    ].iloc[0]

    export_family_series = summary[
        summary["family_group"] == export_family_id
    ]["series_id"].tolist()

    export_family = results[
        results["series_id"].isin(export_family_series)
    ].copy()

    export_row1 = st.columns(2)

    with export_row1[0]:
        st.download_button(
            "Download all assignments",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_all_assignments.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_row1[1]:
        st.download_button(
            "Download series summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    export_row2 = st.columns(2)

    with export_row2[0]:
        st.download_button(
            "Download currently selected series",
            data=export_selected.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_selected.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_row2[1]:
        st.download_button(
            "Download selected linked adduct family",
            data=export_family.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_selected_family.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if not adduct_links.empty:
        st.download_button(
            "Download adduct relationships",
            data=adduct_links.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_adduct_relationships.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if last_discover_any and not discovered_candidates.empty:
        st.download_button(
            "Download discovered repeat masses",
            data=discovered_candidates.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_discovered_repeat_masses.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "Change any repeat unit, adduct, or search setting in the sidebar and "
        "click Rerun analysis. The uploaded feature table stays loaded."
    )
