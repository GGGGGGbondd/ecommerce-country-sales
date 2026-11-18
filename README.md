# Ecommerce Group Project

This repository is the team submission workspace. The country-level sales contribution is currently included; the remaining team members can add their work in the reserved folders.

## Current country-sales contribution

- `code/country_sales_analysis.py`: annotated analysis, visualization, runtime validation, cleaning-artifact reconciliation, and fail-fast checks.
- `code/test_country_sales_analysis.py`: six focused automated tests.
- `data/cleaned/`: the two cleaned analysis inputs used by the program.
- `data/cleaning_artifacts/`: cleaning summaries, audit files, verification records, and the Global Superstore returns extract.
- `outputs/country_sales/`: figures, country-level tables, sensitivity results, and machine-readable validation output.
- `report/GROUP_NO_GROUP_NAME.pdf`: final English report section.
- `report/GROUP_NO_GROUP_NAME.tex`: source for the final English report section.

The report and program use the cleaned CSV files as the analysis inputs. The original raw source files are not duplicated in this submission copy; the `data/raw/` folder is retained as a slot for any source files needed by other team members.

## Team slots

- Add `customer_analysis.py`, `product_analysis.py`, and `time_analysis.py` under `code/`.
- Add their figures, tables, and validation output under `outputs/customer/`, `outputs/product/`, and `outputs/time/`.
- Add the final group presentation as `presentation/GROUP_NO_GROUP_NAME.pptx`.
- Add the assembled group report as `report/GROUP_NO_GROUP_NAME.pdf` when the team merges all sections. Keep the current country-sales source/report files until that merge is complete.

## Reproduce the country-sales contribution

Run the commands from this repository root with a Python environment containing pandas, NumPy, and Matplotlib:

```text
python code/country_sales_analysis.py
python -m unittest discover -s code -p "test_*.py" -v
```

The analysis reads only `data/cleaned/` and writes results under `outputs/country_sales/`.
