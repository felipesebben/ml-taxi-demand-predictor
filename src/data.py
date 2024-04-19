# Functions that deal with data transformation steps.
from pathlib import Path
import os
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from pdb import set_trace as stop

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR

def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """
    Constructs the URL where the datafile is.
    Downloads a parquet file with historical taxi rides for given `year` and `month`.

    Args:
    `year` (`int`) - year of URL
    `month` (`int`) - month of URL

    Returns:
    `path` (`str`) - string containing the path to raw data.
    """
    URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f"{URL} is not available")
    

def validate_raw_data(
        rides: pd.DataFrame,
        year: int,
        month: int,
) -> pd.DataFrame:
    """
    Removes rows with `pickup_datetimes` outside their valid range.
    
    Args:
    - `rides` (`pd.DataFrame`) - Dataframe with rides data
    - `year` (`int`) - year of URL
    - `month` (`int`) - month of URL

    Returns:

    - `rides` (`pd.DataFrame`) - Validated Pandas DataFrame
    """
    this_month_start = f"{year}-{month:02d}-01"
    next_month_start = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01"
    rides = rides[rides["pickup_datetime"] >= this_month_start]
    rides = rides[rides["pickup_datetime"] < next_month_start]

    return rides


def fetch_ride_events_from_data_warehouse(
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
    """
    Simulate production data by sampling historical data
    from 52 weeks ago (that is, 1 year)
    
    Args:
    - `from_date` (`datetime`) - initial datetime
    - `to_date` (`datetime`) - ending datetime

    Returns:

    - Prints the datetime periods to be used as reference.
    """
    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days=7*52)
    print(f"Fetching ride events from {from_date} to {to_date}")

    if (from_date_.year == to_date_.year) and (from_date_.month == to_date_.month):
        # Download only one batch of data
        rides = load_raw_data(year=from_date_.year, months=from_date_.month)
        rides = rides[rides["pickup_datetime"] >= from_date_]
        rides = rides[rides["pickup_datetime"] < to_date_]
    
    else:
        # Download 2 batches from webside
        rides = load_raw_data(year=from_date_.year, months=from_date_.month)
        rides = rides[rides["pickup_datetime"] >= from_date_]
        rides_2 = load_raw_data(year=to_date_.year, months=to_date_.month)
        rides_2 = rides_2[rides_2["pickup_datetime"] < to_date_]
        rides = pd.concat([rides, rides_2])

    # Shift the pickup_datetime back 1 year ahead to simulate production data
    # using 7*52-days ago value
    rides["pickup_datetime"] += timedelta(days=7*52)

    rides.sort_values(by=["pickup_location_id", "pickup_datetime"], inplace=True)

    return rides


def load_raw_data(
        year: int,
        months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Loads raw data from local storage or downloads it from the NYC webside,
    and then loads it into a Pandas DataFrame
    
    Args:
    - `year` (`int`) - year of URL
    - `month` (`int`) - month of URL. If `None`, download all months

    Returns:

    - `pd.DataFrame`: DataFrame with the following columns:
        - pickup_datetime: datetime of the pickup
        - pickup_location_id: ID of the pickup location
    """   
    rides = pd.DataFrame()

    if months is None:
        # Download data for the entire year (all months)
        months = list(range(1, 13))
    elif isinstance(months, int):
        # If month is an `int`, download data for the specified month
        months = [months]

    for month in months:

        local_file = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        if not local_file.exists():
            try:
                # Download the file from the NYC website
                print(f"Downloading file {year}-{month:02d}")
                download_one_file_of_raw_data(year, month)
            except:
                print(f"{year}-{month:02d} is not available")
                continue
        else:
            print(f"File {year}-{month:02d} was already in local storage")
        
        # Load the file into Pandas
        rides_one_month = pd.read_parquet(local_file)

        # Rename columns
        rides_one_month = rides_one_month[["tpep_pickup_datetime", "PULocationID"]]
        rides_one_month.rename(columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "PULocationID": "pickup_location_id",
        }, inplace=True)

        # Validate the file
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        # Append to existing data
        rides = pd.concat([rides, rides_one_month])

    if rides.empty:
        # If no data, return an empty DataFrame
        return pd.DataFrame()
    else:
        # Keep only time and origin of the ride
        rides = rides[["pickup_datetime", "pickup_location_id"]]
        return rides


def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add necessary rows to the input `ts_data` to make sure the output has complete list of:
    - pickup_hours
    - pickup_location_id
    
    Args:
    `ts_data` (`pd.DataFrame`) - Pandas DataFrame with time series data to be transformed.

    Returns:
    `output` (`pd.DataFrame`)
    """

    location_ids =  range(1, ts_data["pickup_location_id"].max() + 1)
    full_range = pd.date_range(ts_data["pickup_hour"].min(), 
                               ts_data["pickup_hour"].max(), 
                               freq="h") # Get the min and max pickup_hour observations
    output = pd.DataFrame()

    for location_id in tqdm(location_ids):

        # Keep only rides for this 'location_id'
        ts_data_i = ts_data.loc[ts_data["pickup_location_id"] == location_id, ["pickup_hour", "rides"]]

        if ts_data_i.empty:
            #  If no data for this location, add a dummy row with 0 rides
            ts_data_i = pd.DataFrame.from_dict([
                {"pickup_hour": ts_data["pickup_hour"].max(),
                 "rides": 0}
            ])
        # Quick way to add missing datas with 0 in a Series
        ts_data_i.set_index("pickup_hour", inplace=True) # Set 'pickup_hour' as index
        ts_data_i.index = pd.DatetimeIndex(ts_data_i.index) # Convert index to DatetimeIndex
        ts_data_i = ts_data_i.reindex(full_range, fill_value=0) # Reindex using min and max datetimes as the full range, passing 0 to NaN values

        # Add back `location_id` columns
        ts_data_i["pickup_location_id"] = location_id

        output = pd.concat([output, ts_data_i])

    # Move the `pickup_hour` from the index to a DataFrame column
    output = output.reset_index().rename(columns={"index": "pickup_hour"})

    return output

def transform_raw_data_into_ts_data(
        rides: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforms the raw data into time-series format. 

    Sum the rides per location and pickup hour, grouping them by `pickup_hour` and `pickup_location_id`

    Add rows in which (`locations`, `pickup_hours`) had no observations with a `0` count.
    Args:

    - `rides` (`pd.DataFrame`) - DataFrame to perform the data transformation

    Returns:

    - `pd.DataFrame` - time-series format
    """
    rides["pickup_hour"] = rides["pickup_datetime"].dt.floor("h")
    agg_rides = rides.groupby(["pickup_hour", "pickup_location_id"]).size().reset_index()
    agg_rides.rename(columns={0: "rides"}, inplace=True)

    # Add rows for (locations, pickup_hour) with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots


def transform_ts_data_into_features_and_target(
        ts_data: pd.DataFrame,
        input_seq_len: int,
        step_size: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Slices and transposes data from time series format into a (`features`,`target`)
    format that we can use to train supervised ML models.
    
    Args:
    
    `ts_data` (`pd.DataFrame`) - time series Pandas DataFrame
    
    `input_seq_len` (`int`) - length of the slice to be performed
    
    `step_size` - size to slide the feature window. 

    Returns:
    
    `pd.DataFrame` for features and `pd.DataFrame` for targets
    """    
    assert set(ts_data.columns) == {"pickup_hour", "rides", "pickup_location_id"}  # Validates variables

    location_ids = ts_data["pickup_location_id"].unique() # Store unique ids
    features = pd.DataFrame() # Create empty DataFrame for features
    targets = pd.DataFrame() # Create empty DataFrame for features

    for location_id in tqdm(location_ids):

        # Keep only time series data for the id over which we are iterating
        ts_data_one_location = ts_data.loc[ts_data["pickup_location_id"] == location_id,
                                           ["pickup_hour", "rides"]
                                           ].sort_values(by=["pickup_hour"])
        
        # Pre-compute cutoff indices to split DataFrame rows
        indices = get_cutoff_indices_features_and_target(
            ts_data_one_location,
            input_seq_len,
            step_size
        )

        # Slice and transpose data into NumPy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]["rides"].values
            y[i] = ts_data_one_location.iloc[idx[1] : idx[2]]["rides"].values[0]
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]["pickup_hour"])

        # Convert NumPy array to Pandas DataFrame
        features_one_location = pd.DataFrame(
            x,
            columns=[f"rides_previous_{i+1}_hour" for i in reversed(range(input_seq_len))]
            )
        features_one_location["pickup_hour"] = pickup_hours
        features_one_location["pickup_location_id"] = location_id

        # Convert NumPy target array to Pandas DataFrame
        targets_one_location = pd.DataFrame(y, columns=[f"target_rides_next_hour"])

        # Concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets["target_rides_next_hour"]


def get_cutoff_indices_features_and_target(
        data: pd.DataFrame, 
        input_seq_len: int,
        step_size: int) -> list:
    """
    Slices a time series Pandas DataFrame with a given number of features and step size.

    Args:
    `data` (`pd.DataFrame`) - time series Pandas DataFrame
    `input_seq_len` (`int`) - length of the slice to be performed
    `step_size` - size to slide the feature window

    Returns:
    `list` - List of indices for features and target
    """

    stop_position = len(data) - 1

    # Start the first subsequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx = input_seq_len
    subseq_last_idx = input_seq_len + 1
    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))

        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices