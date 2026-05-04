import pandas as pd
import numpy as np

class FairnessAnalyzer:
    """
    Measurement Module for Fairness Analysis.
    """
    def __init__(self, file_path):
        """
        Load data
        """
        self.df = pd.read_csv(file_path)
        
        # Removing spaces from column names
        self.df.columns = self.df.columns.str.strip()
        
        print(f"Loaded Dataset: {len(self.df)} rows.")
    
    def bin_column(self, column_name, bins, labels, new_column_name):
        """
        Bin a continuous column into categorical bins.
        """
        self.df[new_column_name] = pd.cut(self.df[column_name], bins=bins, labels=labels)
        print(f"Column '{column_name}' binned into '{new_column_name}' with labels: {labels}")

    def get_group_stats(self, target_col, sensitive_col):
        """
        Number of individuals in each protected group.
        """
        # grouping by the sensitive column and counting the number of entries in each group
        stats = self.df.groupby(sensitive_col).size()
        return stats

