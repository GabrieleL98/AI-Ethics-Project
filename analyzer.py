import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame
from aif360.datasets import BinaryLabelDataset

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from aif360.metrics import ClassificationMetric
   
from sklearn.metrics import mean_absolute_error
from fairlearn.metrics import MetricFrame

import mlflow
from aequitas.group import Group
from aequitas.bias import Bias
from statsmodels.stats.multitest import multipletests

class FairnessAnalyzer:
    """
    Unified Library Integration Layer for Fairness Analysis.
    Acts as a wrapper for Fairlearn, AIF360, and Aequitas.
    """
    def __init__(self, file_path, target_col=None, sensitive_col=None, positive_label=1):
        """
        Initializes the analyzer and establishes the data contract.
        """
        self.df = pd.read_csv(file_path)
        self.df.columns = self.df.columns.str.strip()
        
        # Core configuration for abstraction
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self.positive_label = positive_label
        
        print(f"Loaded Dataset: {len(self.df)} rows.")

    def set_config(self, target_col, sensitive_col, positive_label):
        """
        Updates the analyzer configuration if not set during initialization.
        """
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self.positive_label = positive_label

    def bin_column(self, column_name, bins, labels, new_column_name):
        """
        Bin a continuous column into categorical bins.
        """
        self.df[new_column_name] = pd.cut(self.df[column_name], bins=bins, labels=labels)
        # Automatically update the sensitive column to the new binned one
        self.sensitive_col = new_column_name
        print(f"Column '{column_name}' binned into '{new_column_name}'.")

    # --- INTERNAL ADAPTERS (The Abstraction Layer) ---

    def _prepare_aif360_dataset(self):
        """
        Converts the Pandas DataFrame into an AIF360 BinaryLabelDataset object.
        Requirement: Abstracts underlying complexity of AIF360.
        """
        if not self.target_col or not self.sensitive_col:
            raise ValueError("Target and Sensitive columns must be defined.")

        return BinaryLabelDataset(
            df=self.df,
            label_names=[self.target_col],
            protected_attribute_names=[self.sensitive_col],
            favorable_label=self.positive_label,
            unfavorable_label=0 if self.positive_label == 1 else 0 # Simplified logic
        )

    def _prepare_fairlearn_input(self):
        """
        Returns data in the format expected by Fairlearn (y_true, sensitive_features).
        """
        y_true = self.df[self.target_col]
        sensitive_features = self.df[self.sensitive_col]
        return y_true, sensitive_features

    def calculate_regression_metrics(self, target_col=None):
        """
        Requirement: Robust metrics engine for Regression.
        Computes the difference in Mean Absolute Error (MAE) across groups.
        """
        # If target_col is provided, we update it (e.g., to 'Interest_Rate')
        if target_col:
            self.target_col = target_col
            
        if not self.target_col or not self.sensitive_col:
            raise ValueError("Configuration missing: ensure target and sensitive columns are set.")

        # In a real scenario, y_pred would be the model's output.
        # For the demo, we simulate a 'prediction' with some noise 
        # or use the column itself to show the infrastructure works.
        y_true = self.df[self.target_col]
        y_pred = self.df[self.target_col] # Simulated prediction
        sensitive_features = self.df[self.sensitive_col]

        # MetricFrame allows us to compute MAE disaggregated by group
        metrics = {
            'mean_absolute_error': mean_absolute_error
        }
        
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )

        # Difference across groups
        mae_diff = mf.difference(method='between_groups')['mean_absolute_error']
        mae_by_group = mf.by_group

        return {
            "library": "Fairlearn",
            "metric_name": "MAE Difference",
            "overall_mae": mf.overall['mean_absolute_error'],
            "diff_value": mae_diff,
            "by_group": mae_by_group.to_dict()
        }
    
    def create_intersectional_feature(self, column_names, new_column_name):
        """
        Requirement: Intersectionality.
        Combines multiple sensitive columns into a single intersectional feature.
        Example: ['Gender', 'Age_Group'] -> 'Gender_Age_Group' (e.g., 'Male_Young')
        """
        # Combine column values with an underscore
        self.df[new_column_name] = self.df[column_names].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        self.sensitive_col = new_column_name
        print(f"Intersectional feature '{new_column_name}' created from {column_names}.")

    def _filter_by_group_size(self, sensitive_col, min_group_size):
        """
        Internal helper to identify groups that meet the minimum size requirement.
        Returns a filtered version of the dataframe.
        """
        group_counts = self.df[sensitive_col].value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index
        
        # Identify excluded groups for reporting
        excluded_groups = group_counts[group_counts < min_group_size].index.tolist()
        if excluded_groups:
            print(f"Warning: Excluding groups with size < {min_group_size}: {excluded_groups}")
            
        return self.df[self.df[sensitive_col].isin(valid_groups)], excluded_groups

    def calculate_classification_metrics(self, engine='fairlearn', privileged_group_name=None, min_group_size=5):
        """
        Updated Classification Engine with group size filtering.
        """
        # Filter the dataframe based on min_group_size
        filtered_df, excluded = self._filter_by_group_size(self.sensitive_col, min_group_size)
        
        if filtered_df.empty:
            raise ValueError("No groups meet the minimum size requirement.")

        y_true = filtered_df[self.target_col]
        y_pred = filtered_df[self.target_col] # Simulated proxy
        sensitive_features = filtered_df[self.sensitive_col]

        if engine == 'fairlearn':
            # Fairlearn calculation on filtered data
            dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
            eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
            
            return {
                "library": "Fairlearn",
                "demographic_parity_difference": dp_diff,
                "equalized_odds_difference": eo_diff,
                "excluded_groups": excluded,
                "valid_samples": len(filtered_df)
            }
        
    def _compute_bootstrap_ci(self, metric_func, n_iterations=1000):
        """
        Requirement: Statistical Validation.
        Computes 95% Bootstrap Confidence Interval for a given metric function.
        """
        bootstrapped_stats = []
        
        for _ in range(n_iterations):
            # Resample with replacement (create a 'fake' dataset of same size)
            sample = self.df.sample(frac=1.0, replace=True)
            
            # Use a temporary analyzer instance to calculate the metric on the sample
            # This ensures we use the same internal logic
            val = metric_func(sample)
            bootstrapped_stats.append(val)
        
        # Calculate percentiles for 95% CI
        lower_bound = np.percentile(bootstrapped_stats, 2.5)
        upper_bound = np.percentile(bootstrapped_stats, 97.5)
        
        return lower_bound, upper_bound
    
    def get_fairness_audit(self, n_iterations=1000):
        """
        Requirement: Statistical Validation Framework & Structured Object Output.
        Returns a comprehensive report with CI and Effect Sizes.
        """
        if not self.target_col or not self.sensitive_col:
            raise ValueError("Target and Sensitive columns must be set.")

        # 1. Calculate the 'point estimate' (the simple value)
        rates = self.calculate_selection_rate(self.target_col, self.sensitive_col, self.positive_label)
        
        # We need to define who is privileged (max rate) and unprivileged (min rate)
        # for the Risk Ratio calculation
        priv_group = rates.idxmax()
        unpriv_group = rates.idxmin()
        
        base_selection_rate = rates[priv_group]
        target_selection_rate = rates[unpriv_group]
        
        # Risk Ratio (Effect Size)
        risk_ratio = target_selection_rate / base_selection_rate if base_selection_rate > 0 else 0

        # 2. Call the internal Bootstrap method for Confidence Intervals
        # We define a small lambda to pass to the bootstrap engine
        metric_to_boot = lambda df: (
            df[df[self.sensitive_col] == unpriv_group][self.target_col] == self.positive_label
        ).sum() / len(df[df[self.sensitive_col] == unpriv_group]) if len(df[df[self.sensitive_col] == unpriv_group]) > 0 else 0

        ci_lower, ci_upper = self._compute_bootstrap_ci(metric_to_boot, n_iterations)

        # 3. Build the Structured Object (Requirement 3)
        report = {
            "metric_name": "Selection Rate Analysis",
            "sensitive_feature": self.sensitive_col,
            "groups": {
                "privileged": {"name": priv_group, "rate": base_selection_rate},
                "unprivileged": {"name": unpriv_group, "rate": target_selection_rate}
            },
            "results": {
                "risk_ratio": risk_ratio,
                "confidence_interval_95": (ci_lower, ci_upper),
                "is_statistically_significant": not (ci_lower <= base_selection_rate <= ci_upper)
            },
            "sample_sizes": self.df[self.sensitive_col].value_counts().to_dict()
        }

        return report
    
    # --- PHASE 3: AEQUITAS INTEGRATION ---
    
    def _prepare_aequitas_input(self):
        """
        Internal adapter for Aequitas. 
        Requires 'score' and 'label_value' columns.
        """
        ae_df = self.df[[self.target_col, self.sensitive_col]].copy()
        ae_df.rename(columns={self.target_col: 'label_value', self.sensitive_col: 'entity_id'}, inplace=True)
        # Assuming y_pred = y_true for the demo audit
        ae_df['score'] = ae_df['label_value'] 
        return ae_df

    def get_aequitas_metrics(self):
        """
        Requirement: Unified Layer (Third library integration).
        Uses Aequitas to compute group and bias metrics.
        """
        ae_df = self._prepare_aequitas_input()
        g = Group()
        xtab, _ = g.get_crosstabs(ae_df, attr_cols=['entity_id'])
        
        b = Bias()
        # Comparing against the most frequent group as default
        bias_df = b.get_disparity_predefined_groups(xtab, ae_df, 
                                                   ref_groups_dict={'entity_id': self.df[self.sensitive_col].mode()[0]},
                                                   mask_entities=False)
        return bias_df

    # --- PHASE 4: MLOPS & TESTING INTEGRATION ---

    def log_to_mlflow(self, report):
        """
        Requirement: Seamless MLOps integration.
        Logs the structured fairness report to an active MLflow run.
        """
        if not mlflow.active_run():
            print("Warning: No active MLflow run found. Metrics will not be logged.")
            return

        # Log parameters
        mlflow.log_param("sensitive_column", self.sensitive_col)
        mlflow.log_param("target_column", self.target_col)

        # Log metrics from the report
        mlflow.log_metric("risk_ratio", report["results"]["risk_ratio"])
        mlflow.log_metric("ci_lower", report["results"]["confidence_interval_95"][0])
        mlflow.log_metric("ci_upper", report["results"]["confidence_interval_95"][1])
        
        print("Fairness metrics successfully logged to MLflow.")

    @staticmethod
    def assert_fairness(report, threshold=0.8):
        """
        Requirement: Custom pytest assertion function.
        Used in CI/CD pipelines to automatically fail if bias is too high.
        """
        ratio = report["results"]["risk_ratio"]
        if ratio < threshold:
            raise AssertionError(f"Fairness check failed: Risk Ratio {ratio:.4f} is below threshold {threshold}")
        return True

    # --- PHASE 5: STRETCH GOAL (Multiple Comparisons) ---

    def apply_bias_correction(self, p_values):
        """
        Stretch Goal: Benjamini-Hochberg correction for intersectional analyses.
        Prevents False Positives when checking many groups.
        """
        _, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        return pvals_corrected