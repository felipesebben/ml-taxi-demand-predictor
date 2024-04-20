from datetime import datetime
from typing import Tuple

import pandas as pd

def train_test_split(
        df: pd.DataFrame,
            cutoff_date: datetime,
            target_column_name: str,
            ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the dataset into training and test datasets.

    Args:

    - `df` (`pd.DataFrame`) - Pandas DataFrame dataset to be splitted.
    - `cutoff_date` - (`datetime`) - Date used to divide the dataset.
    - `target_column_name` - (`str`) - Column containing the target data.

    Returns:

    - `Tuple` - training and test data divided into features and targets (4 objects).
    """
    # Get training data, which is all observations before the cutoff date.
    train_data = df[df["pickup_hour"] < cutoff_date].reset_index(drop=True)
    # Apply the same logic to test data for observations after the cutoff date.
    test_data = df[df["pickup_hour"] >= cutoff_date].reset_index(drop=True)

    # Get training features by removing the target column.
    X_train = train_data.drop(columns=[target_column_name])
    
    # Get training target variable
    y_train = train_data[target_column_name]

    # Apply same logic to test data.
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test