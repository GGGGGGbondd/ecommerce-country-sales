"""Core functional tests for the country-sales contribution."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from country_sales_analysis import (  # noqa: E402
    COUNTRY_MAPPING,
    DATASETS,
    SPECIAL_LABELS,
    DatasetConfig,
    comparable_country_summary,
    high_quantity_or_price_sensitivity,
    load_dataset,
    source_reconciliation,
    summarize_country_sales,
    validate_dataset,
)


def synthetic_frame(
    countries: list[str | None],
    sales: list[float],
    dates: list[str] | None = None,
) -> pd.DataFrame:
    """Create a small normalized frame for validation checks."""
    dates = dates or ["2011-01-01"] * len(countries)
    return pd.DataFrame(
        {
            "Country_Original": countries,
            "Country_Normalized": countries,
            "Country_Comparable": countries,
            "IsSpecialLabel": [country in SPECIAL_LABELS for country in countries],
            "IsComparableCountry": [country not in SPECIAL_LABELS for country in countries],
            "Sales": sales,
            "InvoiceDate": dates,
        }
    )


class CountrySalesCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_a = load_dataset(DATASETS[0])
        cls.dataset_b = load_dataset(DATASETS[1])

    def test_invalid_data_detection(self) -> None:
        config = DatasetConfig(
            "test", "Test", Path("test.csv"), "InvoiceDate"
        )
        cases = [
            (synthetic_frame(["A", "B"], [10.0, -1.0]), "negative_sales_count"),
            (synthetic_frame(["A", ""], [10.0, 20.0]), "blank_country_count"),
            (synthetic_frame(["A", None], [10.0, 20.0]), "missing_country_count"),
            (synthetic_frame(["A", "B"], [10.0, np.inf]), "non_finite_sales_count"),
            (
                synthetic_frame(["A", "B"], [10.0, 20.0], ["2011-01-01", "bad"]),
                "invalid_date_count",
            ),
        ]
        for frame, field in cases:
            with self.subTest(field=field):
                result = validate_dataset(frame, config)
                self.assertEqual(result[field], 1)
                self.assertFalse(result["all_internal_checks_pass"])

    def test_country_label_treatment(self) -> None:
        self.assertEqual(COUNTRY_MAPPING["EIRE"], "Ireland")
        self.assertEqual(COUNTRY_MAPPING["USA"], "United States")
        self.assertEqual(COUNTRY_MAPPING["RSA"], "South Africa")
        actual_special = self.dataset_a.loc[self.dataset_a["IsSpecialLabel"]]
        self.assertEqual(set(actual_special["Country_Normalized"]), SPECIAL_LABELS)
        self.assertTrue(actual_special["Country_Original"].notna().all())
        self.assertTrue((actual_special["Sales"] > 0).all())

    def test_sales_aggregation(self) -> None:
        result = validate_dataset(self.dataset_b, DATASETS[1])
        summary = summarize_country_sales(self.dataset_b)
        self.assertTrue(result["groupby_matches_pivot_table"])
        self.assertTrue(result["grouped_total_matches_row_total"])
        self.assertTrue(result["sales_share_sum_is_one"])
        self.assertAlmostEqual(summary["TotalSales"].sum(), self.dataset_b["Sales"].sum())

    def test_cross_dataset_comparison(self) -> None:
        a = comparable_country_summary(self.dataset_a)
        b = comparable_country_summary(self.dataset_b)
        a_set, b_set = set(a["Country"]), set(b["Country"])
        self.assertEqual(len(a_set), 35)
        self.assertEqual(len(b_set), 147)
        self.assertEqual(len(a_set.intersection(b_set)), 31)
        self.assertEqual(len(a_set - b_set), 4)
        self.assertEqual(len(b_set - a_set), 116)

    def test_high_quantity_or_price_sensitivity(self) -> None:
        high_quantity_or_price = high_quantity_or_price_sensitivity(
            self.dataset_a, "Dataset A (08ecomm)"
        )
        self.assertEqual(
            int(high_quantity_or_price.iloc[0]["HighQuantityOrPriceRows"]),
            9918,
        )
        self.assertAlmostEqual(
            float(high_quantity_or_price.iloc[0]["HighQuantityOrPriceSales"]),
            2520542.210,
            places=3,
        )
        self.assertAlmostEqual(
            float(high_quantity_or_price.iloc[0]["HighQuantityOrPriceShareOfAllSales"]),
            0.2370925269,
            places=8,
        )
        self.assertAlmostEqual(
            float(high_quantity_or_price.iloc[1]["TotalSales"]),
            8110506.534,
            places=3,
        )
        self.assertAlmostEqual(
            float(high_quantity_or_price.iloc[1]["TopLabelSalesShare"]),
            0.8530905332,
            places=8,
        )
        self.assertAlmostEqual(
            float(high_quantity_or_price.iloc[1]["Top5SalesShare"]),
            0.9437280831,
            places=8,
        )
        self.assertEqual(int(high_quantity_or_price.iloc[1]["LabelsFor80Percent"]), 1)
        high_quantity_or_price_b = high_quantity_or_price_sensitivity(
            self.dataset_b, "Dataset B (Global Superstore)"
        )
        self.assertEqual(
            int(high_quantity_or_price_b.iloc[0]["HighQuantityOrPriceRows"]),
            954,
        )
        self.assertAlmostEqual(
            float(high_quantity_or_price_b.iloc[0]["HighQuantityOrPriceSales"]),
            1510273.54846,
            places=5,
        )
        self.assertAlmostEqual(
            float(high_quantity_or_price_b.iloc[0]["HighQuantityOrPriceShareOfAllSales"]),
            0.1194600214,
            places=8,
        )
        self.assertAlmostEqual(
            float(high_quantity_or_price_b.iloc[1]["TotalSales"]),
            11132228.36142,
            places=5,
        )
        self.assertAlmostEqual(
            float(high_quantity_or_price_b.iloc[1]["TopLabelSalesShare"]),
            0.1664525144,
            places=8,
        )
        self.assertAlmostEqual(
            float(high_quantity_or_price_b.iloc[1]["Top5SalesShare"]),
            0.4224101165,
            places=8,
        )
        self.assertEqual(
            int(high_quantity_or_price_b.iloc[1]["LabelsFor80Percent"]), 24
        )
        with self.assertRaises(ValueError):
            high_quantity_or_price_sensitivity(
                self.dataset_b.drop(columns=["HighQuantityOrPrice"]),
                "Dataset B (Global Superstore)",
            )

    def test_real_data_validation_and_cleaning_reconciliation(self) -> None:
        result_a = validate_dataset(self.dataset_a, DATASETS[0])
        result_b = validate_dataset(self.dataset_b, DATASETS[1])
        reconciliation_a = source_reconciliation(self.dataset_a, DATASETS[0])
        reconciliation_b = source_reconciliation(self.dataset_b, DATASETS[1])
        self.assertTrue(result_a["all_internal_checks_pass"])
        self.assertTrue(result_b["all_internal_checks_pass"])
        self.assertEqual(result_a["invalid_date_count"], 0)
        self.assertEqual(result_b["invalid_date_count"], 0)
        self.assertEqual(reconciliation_a["manifest_sales_rows"], 524877)
        self.assertEqual(reconciliation_b["manifest_orders_output_rows"], 51290)
        self.assertTrue(reconciliation_a["artifact_checks_pass"])
        self.assertTrue(reconciliation_b["artifact_checks_pass"])


if __name__ == "__main__":
    unittest.main()
