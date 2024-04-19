from typing import Optional, List
from datetime import timedelta

import pandas as pd
import plotly.express as px

def plot_one_sample(
        example_id: int,
        features: pd.DataFrame,
        targets: Optional[pd.Series] = None,
        predictions: Optional[pd.Series] = None,
        display_title: Optional[bool] = True
):
    """
    Plot a sample from the dataset using one row.
    """

    features_ = features.iloc[example_id] # Get row that matches `example_id`

    if targets is not None:
        target_ = targets.iloc[example_id] # Return target if passed as argument
    else:
        target_ = None
    
    ts_columns = [c for c in features.columns if c.startswith("rides_previous_")] # If column name starts with `rides_previous_`, return columm
    ts_values = [features_[c] for c in ts_columns] + [target_]
    ts_dates = pd.date_range(
        features_["pickup_hour"] - timedelta(hours=len(ts_columns)),
        features_["pickup_hour"],
        freq="h"
    )

    # line plot with past values
    title = f"Pick up hour = {features_['pickup_hour']}, location_id={features_['pickup_location_id']}" if display_title else None
    fig = px.line(
        x=ts_dates,
        y=ts_values,
        template="plotly_dark",
        markers=True,
        title=title
    )

    if targets is not None:
        # Green dot for the value we want to predict
        fig.add_scatter(
            x=ts_dates[-1:], y=[target_],
            line_color="green",
            mode="markers", marker_size=10, name="actual value"
        )
    
    if predictions is not None:
        # Red 'X' for the predicted value, if passed
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(
            x=ts_dates[-1:],
            y=[prediction_],
            line_color="red",
            mode="markers", marker_symbol="x", marker_size=15,
            name="prediction"
            )
    return fig


