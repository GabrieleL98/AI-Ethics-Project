============================================================
FAIRNESS ANALYZER - UNIFIED MEASUREMENT MODULE
============================================================

Project Overview:
-----------------
This module provides a unified interface for fairness analysis by wrapping 
three major libraries: Fairlearn, AIF360, and Aequitas. It is designed 
to simplify the auditing of machine learning models and datasets, 
ensuring consistent output through structured results.

Key Features:
-------------
* Unified Interface: Access Fairlearn, AIF360, and Aequitas through a 
  single class (FairnessAnalyzer).
* Structured Outputs: Every metric returns a 'FairnessResult' with 
  point estimates, 95% Bootstrap Confidence Intervals, and effect sizes.
* Data Transformation: Built-in support for binning continuous variables 
  (e.g., Age) and creating intersectional features (e.g., Gender + Age).
* Statistical Rigor: 
    - 95% Bootstrap CI for all key metrics.
    - Benjamini-Hochberg (FDR) correction for intersectional audits.
* MLOps Ready: 
    - Native integration with MLflow for experiment tracking.
    - Custom assertion helpers for CI/CD pipelines (Pytest ready).
* Visualization: Automated generation of fairness reports with error bars.

Installation:
-------------
Ensure you have the following dependencies installed:

pip install pandas numpy scikit-learn matplotlib statsmodels 
pip install fairlearn aif360 aequitas mlflow

Usage Example:
--------------
Below is a quick snippet to get started (detailed in the provided Notebook):

    from fairness_analyzer import FairnessAnalyzer

    # 1. Initialize
    analyzer = FairnessAnalyzer(df="loan_approval_dataset.csv")
    analyzer.set_config(target_col="Loan_Approval_Status", 
                        sensitive_col="Gender", 
                        positive_label=1)

    # 2. Binning continuous features
    analyzer.bin_column(column_name="Age", 
                        bins=[0, 25, 35, 45, 55, 65, float('inf')],
                        labels=['0-25', '26-35', '36-45', '46-55', '56-65', '65+'],
                        new_column_name="Age_Group")

    # 3. Calculate Metrics
    results = analyzer.calculate_classification_metrics(engine="fairlearn")
    print(results['demographic_parity_difference'])

    # 4. Statistical Audit
    audit = analyzer.get_fairness_audit(n_bootstrap=1000)
    print(audit)

    # 5. Intersectional Analysis
    inter_df = analyzer.intersectional_audit(column_names=["Gender", "Age_Group"])

    # 6. Visualize
    analyzer.generate_report_visualizations(results, output_path="report.png")

Core Components:
----------------
- FairnessResult: A dataclass storing metric_name, value, 
  confidence_interval, effect_size, sample_sizes, and metadata.
- FairnessAnalyzer: The main entry point for data ingestion, 
  configuration, and metric calculation.

CI/CD Integration:
------------------
The module includes `assert_fairness`, allowing you to set 
thresholds in your testing pipelines:

    analyzer.assert_fairness(result, threshold=0.8, metric="effect_size")

Dataset Info:
-------------
This implementation was tested using the 'Loan Approval' dataset from Kaggle,
focusing on attributes such as Gender, Age, Income, and Credit Score.

============================================================
Developed as a robust framework for ethical AI auditing.