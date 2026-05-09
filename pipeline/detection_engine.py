"""
bias_detection_engine.py
========================
Unified Detection Engine Module.
   
Automated engine for auditing raw data for bias patterns.
Handles representation, statistical disparities, and proxy detection.

"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import json
from typing import Dict, List, Any, Optional

class BiasDetectionEngine:


    def __init__(self, df: pd.DataFrame, sensitive_col: str, target_col: str):
        self.df = df.copy()
        self.sensitive_col = sensitive_col
        self.target_col = target_col
        self.results = {
            "metadata": {
                "sensitive_column": sensitive_col,
                "target_column": target_col,
                "total_rows": len(df)
            },
            "checks": {}
        }

    def check_representation(self, benchmarks: Dict[str, float]) -> Dict[str, Any]:
        """
        Requirement: Compare demographic distributions against benchmarks.
        Example benchmarks: {"Male": 0.5, "Female": 0.5}
        """
        counts = self.df[self.sensitive_col].value_counts(normalize=True).to_dict()
        disparities = {}
        
        for group, benchmark_ratio in benchmarks.items():
            actual_ratio = counts.get(group, 0)
            # Calculate the ratio relative to benchmark (Representation Ratio)
            disparities[group] = {
                "actual_ratio": round(actual_ratio, 4),
                "benchmark_ratio": benchmark_ratio,
                "diff": round(actual_ratio - benchmark_ratio, 4)
            }
            
        self.results["checks"]["representation_bias"] = disparities
        return disparities

    def check_statistical_disparity(self) -> Dict[str, Any]:
        """
        Requirement: Conduct statistical tests to identify outcome disparities.
        Uses Chi-Squared test for categorical outcomes (e.g., Loan Approval).
        """
        contingency_table = pd.crosstab(self.df[self.sensitive_col], self.df[self.target_col])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        # Calculate selection rates per group
        rates = self.df.groupby(self.sensitive_col)[self.target_col].mean().to_dict()
        
        report = {
            "chi2_stat": round(chi2, 4),
            "p_value": round(p_value, 4),
            "is_statistically_significant": p_value < 0.05,
            "selection_rates": rates
        }
        
        self.results["checks"]["statistical_disparity"] = report
        return report

    def identify_proxies(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Requirement: Identify non-sensitive features correlated with protected attributes.
        Uses basic correlation for numeric features and flags potential proxies.
        """
        proxies = []
        # Temporary numeric encoding for correlation check
        temp_df = self.df.copy()
        temp_df[self.sensitive_col] = temp_df[self.sensitive_col].astype('category').cat.codes
        
        correlations = temp_df.select_dtypes(include=[np.number]).corr()[self.sensitive_col]
        
        for col, score in correlations.items():
            if col != self.sensitive_col and abs(score) >= threshold:
                proxies.append({
                    "feature": col,
                    "correlation_with_sensitive_attr": round(score, 4)
                })
        
        self.results["checks"]["proxy_variables"] = proxies
        return proxies

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Requirement: Output a structured report detailing the findings.
        """
        report_json = json.dumps(self.results, indent=4)
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_json)
            print(f"✓ Bias Audit Report saved to {output_path}")
        return report_json