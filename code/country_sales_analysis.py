"""Country-level line-sales analysis for the e-commerce project.

This module is intentionally limited to the team's country-sales contribution.
It treats the cleaned CSV files as the primary analysis inputs, preserves each
dataset's sales definition, creates comparable country labels only for verified
aliases, produces four report figures, and writes machine-readable validation
evidence.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
SOURCE_DIR = PROJECT_ROOT / "data" / "cleaning_artifacts"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "country_sales"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
VALIDATION_DIR = OUTPUT_DIR / "validation"

MONEY_ATOL = 1e-6
SHARE_ATOL = 1e-10

COUNTRY_MAPPING = {
    "EIRE": "Ireland",
    "USA": "United States",
    "RSA": "South Africa",
}
SPECIAL_LABELS = {"Unspecified", "European Community", "Channel Islands"}
COLORS = {
    "a": "#1f77b4",
    "b": "#d62728",
    "accent": "#2ca02c",
    "neutral": "#6b7280",
    "grid": "#d1d5db",
}


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    path: Path
    date_column: str


DATASETS = (
    DatasetConfig(
        key="dataset_a",
        label="Dataset A (08ecomm)",
        path=DATA_DIR / "dataset_a_08ecomm_cleaned.csv",
        date_column="InvoiceDate",
    ),
    DatasetConfig(
        key="dataset_b",
        label="Dataset B (Global Superstore)",
        path=DATA_DIR / "dataset_b_global_superstore_cleaned.csv",
        date_column="InvoiceDate",
    ),
)


def configure_plot_style() -> None:
    """Use a consistent, readable style for all report figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.55,
            "grid.linestyle": "--",
            "legend.frameon": False,
        }
    )


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Load one analysis-ready CSV and enforce the shared country schema."""
    df = pd.read_csv(config.path, encoding="utf-8-sig", low_memory=False)
    if "Country" not in df.columns:
        raise ValueError(f"{config.path.name} is missing Country")
    if "Sales" not in df.columns:
        raise ValueError(f"{config.path.name} is missing Sales")
    df["Country_Original"] = df["Country"]
    # Keep the cleaned/exported Sales field as the analysis measure.
    df["Sales"] = pd.to_numeric(df["Sales"], errors="raise")

    df["Country_Normalized"] = df["Country_Original"].astype("string").str.strip()
    df["Country_Comparable"] = df["Country_Normalized"].replace(COUNTRY_MAPPING)
    df["IsSpecialLabel"] = df["Country_Normalized"].isin(tuple(SPECIAL_LABELS))
    df["IsComparableCountry"] = ~df["IsSpecialLabel"]
    return df


def summarize_country_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line sales before return adjustment."""
    summary = (
        df.groupby(
            ["Country_Normalized", "Country_Comparable", "IsSpecialLabel"],
            dropna=False,
            as_index=False,
        )
        .agg(TotalSales=("Sales", "sum"))
        .rename(columns={"Country_Normalized": "Country"})
        .sort_values(["TotalSales", "Country"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total_sales = summary["TotalSales"].sum()
    summary["Rank"] = np.arange(1, len(summary) + 1)
    summary["SalesShareAll"] = summary["TotalSales"] / total_sales
    summary["CumulativeSalesShareAll"] = summary["SalesShareAll"].cumsum()
    return summary


def comparable_country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only confirmed country labels for cross-dataset comparison."""
    comparable = df.loc[df["IsComparableCountry"]].copy()
    summary = (
        comparable.groupby("Country_Comparable", as_index=False)
        .agg(ComparableSales=("Sales", "sum"))
        .rename(columns={"Country_Comparable": "Country"})
        .sort_values(["ComparableSales", "Country"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total = summary["ComparableSales"].sum()
    summary["ComparableSalesShare"] = summary["ComparableSales"] / total
    return summary


def high_quantity_or_price_sensitivity(
    df: pd.DataFrame, dataset_label: str
) -> pd.DataFrame:
    """Compare country-sales concentration before and after the outlier flag."""
    if "HighQuantityOrPrice" not in df.columns:
        raise ValueError(
            "HighQuantityOrPrice is required for this sensitivity analysis"
        )
    flagged = df["HighQuantityOrPrice"].fillna(False).astype(bool)
    rows = []
    for label, view in [
        ("All retained sales lines", df),
        ("Excluding HighQuantityOrPrice lines", df.loc[~flagged]),
    ]:
        summary = summarize_country_sales(view)
        rows.append(
            {
                "Dataset": dataset_label,
                "Scenario": label,
                "Rows": int(len(view)),
                "TotalSales": float(view["Sales"].sum()),
                "HighQuantityOrPriceRows": (
                    int(flagged.to_numpy().sum())
                    if label.startswith("All")
                    else 0
                ),
                "HighQuantityOrPriceSales": (
                    float(df.loc[flagged, "Sales"].sum())
                    if label.startswith("All")
                    else 0.0
                ),
                "HighQuantityOrPriceShareOfAllSales": (
                    float(df.loc[flagged, "Sales"].sum() / df["Sales"].sum())
                    if label.startswith("All")
                    else 0.0
                ),
                "TopLabel": str(summary.iloc[0]["Country"]),
                "TopLabelSalesShare": float(summary.iloc[0]["SalesShareAll"]),
                "Top5SalesShare": float(
                    summary.head(5)["SalesShareAll"].to_numpy().sum()
                ),
                "LabelsFor80Percent": int(
                    np.argmax(summary["CumulativeSalesShareAll"].to_numpy() >= 0.8)
                    + 1
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_dataset(df: pd.DataFrame, config: DatasetConfig) -> dict[str, Any]:
    """Run internal checks for one dataset."""
    grouped_total = float(
        df.groupby("Country_Normalized", dropna=False)
        .agg(Sales=("Sales", "sum"))["Sales"]
        .to_numpy()
        .sum()
    )
    pivot = df.pivot_table(
        index="Country_Normalized", values="Sales", aggfunc="sum", dropna=False
    )
    groupby = cast(
        pd.Series,
        df.groupby("Country_Normalized", dropna=False)["Sales"].sum(),
    ).sort_index()
    shares = summarize_country_sales(df)["SalesShareAll"]
    parsed_dates = pd.to_datetime(df[config.date_column], errors="coerce")
    sales_array = df["Sales"].to_numpy(dtype=float)
    result: dict[str, Any] = {
        "dataset": config.label,
        "rows": int(len(df)),
        "valid_sales_numeric": bool(pd.api.types.is_numeric_dtype(df["Sales"])),
        "missing_country_count": int(
            df["Country_Original"].isna().to_numpy().sum()
        ),
        "blank_country_count": int(
            (df["Country_Normalized"].fillna("") == "").to_numpy().sum()
        ),
        "negative_sales_count": int(
            (df["Sales"] < 0).to_numpy().sum()
        ),
        "non_finite_sales_count": int((~np.isfinite(sales_array)).sum()),
        "invalid_date_count": int(parsed_dates.isna().to_numpy().sum()),
        "row_level_sales_total": float(df["Sales"].to_numpy().sum()),
        "grouped_sales_total": grouped_total,
        "grouped_total_matches_row_total": bool(
            np.isclose(df["Sales"].sum(), grouped_total, rtol=0, atol=MONEY_ATOL)
        ),
        "sales_share_sum": float(shares.to_numpy().sum()),
        "sales_share_sum_is_one": bool(
            np.isclose(shares.sum(), 1.0, rtol=0, atol=SHARE_ATOL)
        ),
        "groupby_matches_pivot_table": bool(
            np.allclose(
                groupby.to_numpy(),
                pivot["Sales"].sort_index().to_numpy(),
                rtol=0,
                atol=MONEY_ATOL,
            )
        ),
        "special_label_count": int(
            df.loc[df["IsSpecialLabel"], "Country_Normalized"].nunique()
        ),
        "comparable_country_count": int(
            df.loc[df["IsComparableCountry"], "Country_Comparable"].nunique()
        ),
    }
    result["all_internal_checks_pass"] = bool(
        result["valid_sales_numeric"]
        and result["missing_country_count"] == 0
        and result["blank_country_count"] == 0
        and result["negative_sales_count"] == 0
        and result["non_finite_sales_count"] == 0
        and result["invalid_date_count"] == 0
        and result["grouped_total_matches_row_total"]
        and result["sales_share_sum_is_one"]
        and result["groupby_matches_pivot_table"]
    )
    return result


def validate_artifact_flags(payload: dict[str, Any], required_fields: list[str]) -> bool:
    """Return True only when every required cleaning-artifact flag is PASS."""
    return all(payload.get(field) == "PASS" for field in required_fields)


def source_reconciliation(df: pd.DataFrame, config: DatasetConfig) -> dict[str, Any]:
    """Compare cleaned inputs with available source-level totals where possible."""
    if config.key == "dataset_a":
        manifest_path = SOURCE_DIR / "08ecomm_cleaning_summary.json"
        verification_path = SOURCE_DIR / "08ecomm_verification.json"
        audit_path = SOURCE_DIR / "08ecomm_cleaning_audit.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        verification = json.loads(
            verification_path.read_text(encoding="utf-8-sig")
        )
        audit = pd.read_csv(audit_path, encoding="utf-8-sig")
        cleaned_path = config.path
        source_bytes = cleaned_path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        independent = pd.read_csv(cleaned_path, encoding="utf-8-sig", low_memory=False)
        independent["Sales"] = pd.to_numeric(independent["Sales"], errors="raise")
        independent["Country_Normalized"] = (
            independent["Country"].astype("string").str.strip()
        )
        analysis_country = (
            df.groupby("Country_Normalized", dropna=False)
            .agg(analysis=("Sales", "sum"))
            .sort_index()
        )
        independent_country = (
            independent.groupby("Country_Normalized", dropna=False)
            .agg(independent=("Sales", "sum"))
            .sort_index()
        )
        country_join = (
            analysis_country.join(independent_country, how="outer").fillna(0)
        )
        country_diff = (country_join["analysis"] - country_join["independent"]).abs()
        return {
            "source": "08ecomm_cleaning_complete.zip / cleaning_summary.json",
            "source_reconciliation_available": True,
            "manifest_sales_rows": int(manifest["sales_rows"]),
            "manifest_sales_countries": int(
                manifest["sales_countries_including_unspecified"]
            ),
            "manifest_gross_positive_line_amount": float(
                manifest["gross_positive_line_amount"]
            ),
            "manifest_input_rows": int(manifest["input_rows"]),
            "manifest_duplicates_removed": int(manifest["exact_duplicates_removed"]),
            "manifest_excluded_rows": int(manifest["excluded_rows"]),
            "manifest_return_rows": int(manifest["return_rows"]),
            "audit_sales_rows": int(
                audit.loc[audit["Category"].eq("Valid positive sales"), "Rows"].iloc[0]
            ),
            "audit_return_rows": int(
                audit.loc[
                    audit["Category"].eq("Valid returns/cancellations"), "Rows"
                ].iloc[0]
            ),
            "cleaned_rows": int(len(df)),
            "cleaned_country_labels": len(df["Country_Normalized"].unique()),
            "source_sales_total": float(manifest["gross_positive_line_amount"]),
            "cleaned_sales_total": float(df["Sales"].to_numpy().sum()),
            "rows_match": int(manifest["sales_rows"]) == len(df),
            "sales_match": bool(
                np.isclose(
                    manifest["gross_positive_line_amount"],
                    df["Sales"].sum(),
                    rtol=0,
                    atol=MONEY_ATOL,
                )
            ),
            "country_labels_match": int(
                manifest["sales_countries_including_unspecified"]
            )
            == len(df["Country_Normalized"].unique()),
            "country_level_rows_match": len(country_join)
            == len(df["Country_Normalized"].unique()),
            "country_level_max_abs_difference": float(
                country_diff.to_numpy().max()
            ),
            "country_level_sales_match": bool(
                np.allclose(
                    country_join["analysis"],
                    country_join["independent"],
                    rtol=0,
                    atol=MONEY_ATOL,
                )
            ),
            "cleaned_input_sha256": source_hash,
            "artifact_checks_pass": bool(
                int(
                    audit.loc[
                        audit["Category"].eq("Valid positive sales"), "Rows"
                    ].iloc[0]
                )
                == int(manifest["sales_rows"])
                and int(
                    audit.loc[
                        audit["Category"].eq("Valid returns/cancellations"), "Rows"
                    ].iloc[0]
                )
                == int(manifest["return_rows"])
                and
                validate_artifact_flags(
                    manifest,
                    ["row_partition_check", "input_unchanged_sha256_check"],
                )
                and validate_artifact_flags(
                    verification,
                    [
                        "independent_csv_partition_check",
                        "all_exported_sales_values_decimal_check",
                        "all_exported_dates_parse_check",
                        "source_sample_value_check",
                    ],
                )
            ),
            "artifact_flag_details": {
                "row_partition_check": manifest["row_partition_check"],
                "input_unchanged_sha256_check": manifest[
                    "input_unchanged_sha256_check"
                ],
                "independent_csv_partition_check": verification[
                    "independent_csv_partition_check"
                ],
                "all_exported_sales_values_decimal_check": verification[
                    "all_exported_sales_values_decimal_check"
                ],
                "all_exported_dates_parse_check": verification[
                    "all_exported_dates_parse_check"
                ],
                "source_sample_value_check": verification[
                    "source_sample_value_check"
                ],
            },
        }
    manifest_path = SOURCE_DIR / "global_superstore_cleaning_summary.json"
    verification_path = SOURCE_DIR / "global_superstore_verification.json"
    audit_path = SOURCE_DIR / "global_superstore_cleaning_audit.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    verification = json.loads(verification_path.read_text(encoding="utf-8-sig"))
    audit = pd.read_csv(audit_path, encoding="utf-8-sig")
    independent = pd.read_csv(config.path, encoding="utf-8-sig", low_memory=False)
    independent["Country_Normalized"] = (
        independent["Country"].astype("string").str.strip()
    )
    independent["Sales"] = pd.to_numeric(independent["Sales"], errors="raise")
    analysis_country = (
        df.groupby("Country_Normalized", dropna=False)
        .agg(analysis=("Sales", "sum"))
        .sort_index()
    )
    source_country = (
        independent.groupby("Country_Normalized", dropna=False)
        .agg(source=("Sales", "sum"))
        .sort_index()
    )
    country_join = (
        analysis_country.join(source_country, how="outer").fillna(0)
    )
    return {
        "source": "Global Superstore cleaning artifacts",
        "source_reconciliation_available": True,
        "manifest_orders_input_rows": int(manifest["input_rows_by_sheet"]["Orders"]),
        "manifest_orders_output_rows": int(manifest["orders_output_rows"]),
        "manifest_countries": int(manifest["countries"]),
        "manifest_sales_total": float(manifest["totals"]["Sales"]),
        "audit_orders_output_rows": int(
            audit.loc[audit["Sheet"].eq("Orders"), "OutputRows"].iloc[0]
        ),
        "verification_all_original_order_fields": verification[
            "all_original_order_fields_verified"
        ],
        "verification_all_original_measure_totals": verification[
            "all_original_measure_totals_preserved"
        ],
        "cleaned_rows": int(len(df)),
        "source_sales_total": float(manifest["totals"]["Sales"]),
        "cleaned_sales_total": float(df["Sales"].to_numpy().sum()),
        "rows_match": int(manifest["orders_output_rows"]) == len(df),
        "sales_match": bool(
            np.isclose(
                manifest["totals"]["Sales"],
                df["Sales"].sum(),
                rtol=0,
                atol=MONEY_ATOL,
            )
        ),
        "country_labels_match": int(manifest["countries"])
        == len(df["Country_Normalized"].unique()),
        "country_level_rows_match": len(country_join)
        == len(df["Country_Normalized"].unique()),
        "country_level_max_abs_difference": float(
            (country_join["analysis"] - country_join["source"])
            .abs()
            .to_numpy()
            .max()
        ),
        "country_level_sales_match": bool(
            np.allclose(
                country_join["analysis"],
                country_join["source"],
                rtol=0,
                atol=MONEY_ATOL,
            )
        ),
        "artifact_checks_pass": bool(
            int(
                audit.loc[audit["Sheet"].eq("Orders"), "OutputRows"].iloc[0]
            )
            == int(manifest["orders_output_rows"])
            and
            validate_artifact_flags(
                manifest,
                [
                    "row_coverage_and_no_join_expansion",
                    "original_sales_quantity_profit_shipping_preserved",
                ],
            )
            and validate_artifact_flags(
                verification,
                [
                    "all_original_order_fields_verified",
                    "all_original_measure_totals_preserved",
                ],
            )
        ),
        "artifact_flag_details": {
            "row_coverage_and_no_join_expansion": manifest[
                "row_coverage_and_no_join_expansion"
            ],
            "original_sales_quantity_profit_shipping_preserved": manifest[
                "original_sales_quantity_profit_shipping_preserved"
            ],
            "all_original_order_fields_verified": verification[
                "all_original_order_fields_verified"
            ],
            "all_original_measure_totals_preserved": verification[
                "all_original_measure_totals_preserved"
            ],
        },
    }


def save_tables(
    frames: dict[str, pd.DataFrame], overview: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    overview.to_csv(
        TABLE_DIR / "dataset_overview.csv", index=False, encoding="utf-8-sig"
    )
    for key, summary in frames.items():
        summary.to_csv(
            TABLE_DIR / f"{key}_country_market_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary.head(15).to_csv(
            TABLE_DIR / f"{key}_top15.csv", index=False, encoding="utf-8-sig"
        )
    comparison.to_csv(
        TABLE_DIR / "comparable_country_share_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )


def annotate_bars(ax: Axes, bars: Any, percentages: bool = False) -> None:
    """Place compact numeric labels at the right edge of horizontal bars."""
    for bar in bars:
        value = bar.get_width()
        label = f"{value:.1f}%" if percentages else f"{value:,.0f}"
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            " " + label,
            va="center",
            fontsize=8,
        )


def plot_top15(summaries: dict[str, pd.DataFrame]) -> None:
    """Plot the top fifteen labels for each dataset with independent scales."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for ax, (key, summary), color in zip(
        axes, summaries.items(), [COLORS["a"], COLORS["b"]]
    ):
        view = summary.head(15).sort_values("TotalSales")
        bars = ax.barh(view["Country"], view["TotalSales"], color=color, alpha=0.88)
        annotate_bars(ax, bars)
        ax.set_title(
            "Dataset A (08ecomm)"
            if key == "dataset_a"
            else "Dataset B (Global Superstore)"
        )
        ax.set_xlabel("Line sales (source data units)")
        ax.set_ylabel("")
        ax.set_xlim(0, view["TotalSales"].max() * 1.22)
    fig.suptitle(
        "Top 15 Countries or Market Labels by Line Sales Before Return Adjustment",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig1_top15_sales.png", bbox_inches="tight")
    plt.close(fig)


def plot_pareto(summaries: dict[str, pd.DataFrame]) -> None:
    """Separate display bars from the cumulative curve on original ranks."""
    fig, axes = plt.subplots(
        len(summaries),
        2,
        figsize=(15, 11),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.45]},
    )
    axes = np.atleast_2d(axes)
    for row, ((key, summary), color) in enumerate(
        zip(summaries.items(), [COLORS["a"], COLORS["b"]])
    ):
        bar_ax, curve_ax = axes[row]
        top20 = summary.head(20)
        full_rank = np.arange(1, len(summary) + 1)
        bar_view = top20[["Country", "TotalSales"]].copy()
        if len(summary) > 20:
            bar_view = cast(
                pd.DataFrame,
                pd.concat(
                    [
                        bar_view,
                        pd.DataFrame(
                            [{
                                "Country": "Others",
                                "TotalSales": summary.iloc[20:]["TotalSales"].sum(),
                            }]
                        ),
                    ],
                    ignore_index=True,
                ),
            )
        # Keep the synthetic Others group on the bottom of the displayed
        # ranking so it cannot be mistaken for a leading geographic label.
        others = bar_view.loc[bar_view["Country"] == "Others"]
        ranked_labels = bar_view.loc[
            ~(bar_view["Country"] == "Others")
        ].sort_values("TotalSales")
        bar_view = cast(
            pd.DataFrame, pd.concat([others, ranked_labels], ignore_index=True)
        )
        bar_colors = [
            COLORS["neutral"] if country == "Others" else color
            for country in bar_view["Country"]
        ]
        bars = bar_ax.barh(
            bar_view["Country"], bar_view["TotalSales"], color=bar_colors, alpha=0.68
        )
        annotate_bars(bar_ax, bars)
        bar_ax.set_xlabel("Line sales (source data units)")
        bar_ax.set_ylabel("Top 20 labels plus Others")
        bar_ax.set_title(
            "Dataset A (08ecomm)" if key == "dataset_a" else "Dataset B (Global Superstore)"
        )
        curve_ax.plot(
            full_rank,
            summary["CumulativeSalesShareAll"] * 100,
            color=COLORS["accent"],
            linewidth=2.2,
        )
        curve_ax.axhline(80, color=COLORS["neutral"], linestyle=":", linewidth=1)
        # argmax returns the first full-data rank whose cumulative share reaches 80%.
        full_idx80 = (
            int(np.argmax(summary["CumulativeSalesShareAll"].to_numpy() >= 0.8)) + 1
        )
        curve_ax.scatter(
            [full_idx80],
            [
                summary["CumulativeSalesShareAll"].to_numpy()[full_idx80 - 1] * 100
            ],
            color=COLORS["accent"],
            zorder=4,
        )
        curve_ax.annotate(
            f"At least 80%: {full_idx80} "
            f"{'label' if full_idx80 == 1 else 'labels'}",
            (full_idx80, 80),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
        curve_ax.set_xlabel("Original country/market rank")
        curve_ax.set_ylabel("Cumulative sales share (%)")
        curve_ax.set_ylim(0, 105)
        curve_ax.set_title("Full ranked cumulative curve")
    fig.suptitle(
        "Sales Concentration by Country or Market Label",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig2_pareto_sales.png", bbox_inches="tight")
    plt.close(fig)


def plot_comparison(comparable: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Plot the leading union comparison with hollow markers for missing labels."""
    a, b = comparable.values()
    comparison = a.merge(b, on="Country", how="outer", suffixes=("_A", "_B"))
    comparison["UnionSelectionScore"] = (
        comparison[["ComparableSalesShare_A", "ComparableSalesShare_B"]]
        .fillna(0)
        .sum(axis=1)
    )
    view = (
        comparison.sort_values("UnionSelectionScore", ascending=False)
        .head(12)
        .sort_values("UnionSelectionScore")
    )
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    y = np.arange(len(view))
    a_present = view["ComparableSalesShare_A"].notna().to_numpy()
    b_present = view["ComparableSalesShare_B"].notna().to_numpy()
    a_values = view["ComparableSalesShare_A"].fillna(0).to_numpy() * 100
    b_values = view["ComparableSalesShare_B"].fillna(0).to_numpy() * 100
    for idx in range(len(view)):
        if a_present[idx] and b_present[idx]:
            ax.hlines(
                y[idx], a_values[idx], b_values[idx], color="#9ca3af", linewidth=2
            )
    ax.scatter(
        a_values[a_present], y[a_present], color=COLORS["a"], s=65, label="Dataset A"
    )
    ax.scatter(
        b_values[b_present], y[b_present], color=COLORS["b"], s=65, label="Dataset B"
    )
    if (~a_present).any():
        ax.scatter(
            np.zeros((~a_present).sum()),
            y[~a_present],
            facecolors="none",
            edgecolors=COLORS["a"],
            s=65,
            linewidths=1.5,
            label="Dataset A: not observed",
        )
    if (~b_present).any():
        ax.scatter(
            np.zeros((~b_present).sum()),
            y[~b_present],
            facecolors="none",
            edgecolors=COLORS["b"],
            s=65,
            linewidths=1.5,
            label="Dataset B: not observed",
        )
    ax.set_yticks(y, view["Country"])
    ax.set_xlabel("Share of all non-special comparable sales (%)")
    ax.set_title("Comparable Country Sales Shares Across the Two Datasets")
    ax.legend(loc="lower right")
    ax.set_xlim(left=-2)
    fig.savefig(FIG_DIR / "fig3_comparable_sales_share.png", bbox_inches="tight")
    plt.close(fig)
    return comparison.sort_values("UnionSelectionScore", ascending=False).reset_index(
        drop=True
    )


def plot_matched_country_comparison(
    comparable: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Compare the 31-country intersection with a matched denominator."""
    a, b = comparable.values()
    common = sorted(set(a["Country"]).intersection(b["Country"]))
    a_matched = a.loc[a["Country"].isin(common)].copy()
    b_matched = b.loc[b["Country"].isin(common)].copy()
    a_matched["MatchedSalesShare"] = (
        a_matched["ComparableSales"] / a_matched["ComparableSales"].sum()
    )
    b_matched["MatchedSalesShare"] = (
        b_matched["ComparableSales"] / b_matched["ComparableSales"].sum()
    )
    matched = a_matched[["Country", "MatchedSalesShare"]].merge(
        b_matched[["Country", "MatchedSalesShare"]],
        on="Country",
        suffixes=("_A", "_B"),
    )
    matched["MatchedSelectionScore"] = (
        matched["MatchedSalesShare_A"] + matched["MatchedSalesShare_B"]
    )
    view = matched.sort_values("MatchedSelectionScore", ascending=False).reset_index(
        drop=True
    )
    plot_view = view.sort_values("MatchedSelectionScore").copy()
    fig, ax = plt.subplots(figsize=(10, 11), constrained_layout=True)
    y = np.arange(len(plot_view))
    ax.hlines(
        y,
        plot_view["MatchedSalesShare_A"] * 100,
        plot_view["MatchedSalesShare_B"] * 100,
        color="#9ca3af",
        linewidth=1.6,
    )
    ax.scatter(
        plot_view["MatchedSalesShare_A"] * 100,
        y,
        color=COLORS["a"],
        s=42,
        label="Dataset A",
    )
    ax.scatter(
        plot_view["MatchedSalesShare_B"] * 100,
        y,
        color=COLORS["b"],
        s=42,
        label="Dataset B",
    )
    ax.set_yticks(y, plot_view["Country"])
    ax.set_xlabel("Share within the 31 matched countries (%)")
    ax.set_title("Sales Shares within the 31 Shared Geographic Labels")
    ax.legend(loc="lower right")
    ax.set_xlim(left=0)
    fig.savefig(FIG_DIR / "fig4_matched_country_shares.png", bbox_inches="tight")
    plt.close(fig)
    matched_stats = {
        "matched_country_count": int(len(matched)),
        "dataset_a_only_country_count": int(len(set(a["Country"]) - set(b["Country"]))),
        "dataset_b_only_country_count": int(len(set(b["Country"]) - set(a["Country"]))),
        "dataset_b_only_sales_share_all": float(
            b.loc[
                ~b["Country"].isin(tuple(a["Country"])), "ComparableSalesShare"
            ].to_numpy().sum()
        ),
        "dataset_a_only_sales_share_all": float(
            a.loc[
                ~a["Country"].isin(tuple(b["Country"])), "ComparableSalesShare"
            ].to_numpy().sum()
        ),
        "dataset_a_matched_denominator": float(a_matched["ComparableSales"].sum()),
        "dataset_b_matched_denominator": float(b_matched["ComparableSales"].sum()),
    }
    return view, matched_stats


def build_overview(
    frames: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    records = []
    for config in DATASETS:
        df = frames[config.key]
        summary = summaries[config.key]
        date_values = pd.to_datetime(df[config.date_column], errors="coerce")
        top = summary.iloc[0]
        records.append(
            {
                "Dataset": config.label,
                "Valid sales rows": len(df),
                "Country/market labels": summary["Country"].nunique(),
                "Total sales": df["Sales"].sum(),
                "Date start": date_values.min().date().isoformat(),
                "Date end": date_values.max().date().isoformat(),
                "Top label": top["Country"],
                "Top-label share": top["SalesShareAll"],
                "Special-label sales": df.loc[df["IsSpecialLabel"], "Sales"].sum(),
            }
        )
    return pd.DataFrame(records)


def save_failure_diagnostics(
    stage: str,
    error: Exception | str,
    validations: list[dict[str, Any]],
    reconciliations: list[dict[str, Any]],
) -> None:
    """Persist validation failure details before stopping the analysis."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "FAIL",
        "stage": stage,
        "error": str(error),
        "internal_validation": validations,
        "cleaning_artifact_reconciliation": reconciliations,
    }
    (VALIDATION_DIR / "validation_failure_diagnostics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def main() -> None:
    configure_plot_style()
    for directory in (FIG_DIR, TABLE_DIR, VALIDATION_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, pd.DataFrame] = {}
    validations: list[dict[str, Any]] = []
    reconciliations: list[dict[str, Any]] = []
    comparable: dict[str, pd.DataFrame] = {}

    try:
        for config in DATASETS:
            df = load_dataset(config)
            validation = validate_dataset(df, config)
            reconciliation = source_reconciliation(df, config)
            validations.append(validation)
            reconciliations.append(reconciliation)
            frames[config.key] = df
    except Exception as exc:
        save_failure_diagnostics(
            "dataset_processing", exc, validations, reconciliations
        )
        raise

    if not all(item["all_internal_checks_pass"] for item in validations):
        save_failure_diagnostics(
            "internal_validation", "Internal validation failed", validations, reconciliations
        )
        raise RuntimeError("Internal validation failed. Analysis outputs were not generated.")
    if not all(
        item["rows_match"]
        and item["sales_match"]
        and item.get("country_labels_match", True)
        and item.get("country_level_sales_match", True)
        and item.get("artifact_checks_pass", True)
        for item in reconciliations
    ):
        save_failure_diagnostics(
            "cleaning_artifact_reconciliation",
            "Cleaning-artifact reconciliation failed",
            validations,
            reconciliations,
        )
        raise RuntimeError(
            "Cleaning-artifact reconciliation failed. Analysis outputs were not generated."
        )

    # Build summaries only after all validation and artifact checks pass.
    # This fail-fast order prevents partially validated data from producing
    # report tables or figures.
    for config in DATASETS:
        df = frames[config.key]
        summaries[config.key] = summarize_country_sales(df)
        comparable[config.key] = comparable_country_summary(df)

    overview = build_overview(frames, summaries)
    comparison = plot_comparison(comparable)
    matched_comparison, comparison_stats = plot_matched_country_comparison(comparable)
    high_quantity_or_price = pd.concat(
        [
            high_quantity_or_price_sensitivity(
                frames["dataset_a"], "Dataset A (08ecomm)"
            ),
            high_quantity_or_price_sensitivity(
                frames["dataset_b"], "Dataset B (Global Superstore)"
            ),
        ],
        ignore_index=True,
    )
    save_tables(summaries, overview, comparison)
    high_quantity_or_price.to_csv(
        TABLE_DIR / "high_quantity_or_price_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matched_comparison.to_csv(
        TABLE_DIR / "matched_country_share_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([comparison_stats]).to_csv(
        TABLE_DIR / "comparison_set_summary.csv", index=False, encoding="utf-8-sig"
    )
    plot_top15(summaries)
    plot_pareto(summaries)

    validation_payload = {
        "internal_validation": validations,
        "source_reconciliation": reconciliations,
        "all_internal_checks_pass": all(
            item["all_internal_checks_pass"] for item in validations
        ),
        "all_source_reconciliations_match": all(
            item["rows_match"]
            and item["sales_match"]
            and item.get("country_labels_match", True)
            and item.get("country_level_sales_match", True)
            and item.get("artifact_checks_pass", True)
            for item in reconciliations
        ),
    }
    (VALIDATION_DIR / "validation_results.json").write_text(
        json.dumps(validation_payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(validations).to_csv(
        VALIDATION_DIR / "internal_validation.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(reconciliations).to_csv(
        VALIDATION_DIR / "source_reconciliation.csv", index=False, encoding="utf-8-sig"
    )

    report_payload = {
        "overview": overview.to_dict(orient="records"),
        "validation": validation_payload,
        "top15": {
            key: value.head(15).to_dict(orient="records")
            for key, value in summaries.items()
        },
        "comparison_top12": comparison.head(12).to_dict(orient="records"),
        "matched_country_comparison": matched_comparison.to_dict(orient="records"),
        "comparison_set_summary": comparison_stats,
        "high_quantity_or_price_sensitivity": high_quantity_or_price.to_dict(
            orient="records"
        ),
    }
    (OUTPUT_DIR / "analysis_results.json").write_text(
        json.dumps(report_payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report_payload, indent=2, default=str))


if __name__ == "__main__":
    main()
