import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame
from aif360.datasets import BinaryLabelDataset

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from aif360.metrics import ClassificationMetric

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

class FairnessAnalyzer:
    # ... (metodi precedenti: __init__, bin_column, set_config, _prepare_aif360_dataset) ...

    def calculate_classification_metrics(self, engine='fairlearn', privileged_group_name=None):
        """
        Requirement: Robust metrics engine for Classification.
        Implements calculations for at least two metrics.
        """
        if not self.target_col or not self.sensitive_col:
            raise ValueError("Configuration missing: call set_config() first.")

        # In a real scenario, y_pred would come from a model. 
        # Here we use the target_col as a proxy for demonstration.
        y_true = self.df[self.target_col]
        y_pred = self.df[self.target_col] 
        sensitive_features = self.df[self.sensitive_col]

        if engine == 'fairlearn':
            dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
            eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
            
            return {
                "library": "Fairlearn",
                "demographic_parity_difference": dp_diff,
                "equalized_odds_difference": eo_diff
            }

        elif engine == 'aif360':
            # AIF360 logic requires defining which group is 'privileged'
            if privileged_group_name is None:
                # Default to the first group found if not specified
                privileged_group_name = self.df[self.sensitive_col].unique()[0]

            aif_ds = self._prepare_aif360_dataset()
            
            # For AIF360, we need to map the privileged_group_name to its encoded value
            privileged_groups = [{self.sensitive_col: 1}] # AIF360 uses 1 for privileged often
            unprivileged_groups = [{self.sensitive_col: 0}]

            # Simplified AIF360 metric extraction
            metric_aif = ClassificationMetric(aif_ds, aif_ds, 
                                               unprivileged_groups=unprivileged_groups, 
                                               privileged_groups=privileged_groups)
            
            return {
                "library": "AIF360",
                "statistical_parity_difference": metric_aif.statistical_parity_difference(),
                "equal_opportunity_difference": metric_aif.equal_opportunity_difference()
            }
    
from sklearn.metrics import mean_absolute_error
from fairlearn.metrics import MetricFrame

class FairnessAnalyzer:
    # ... (metodi precedenti rimangono invariati) ...

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