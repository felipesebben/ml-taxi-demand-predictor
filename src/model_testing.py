import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
import geopandas as gpd
import pandas as pd
from src.paths import RAW_DATA_DIR

import lightgbm as lgb

def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rides from:

    - 7 days ago
    - 14 days ago
    - 21 days ago
    - 28 days ago

    Args:

    - `X` (`pd.DataFrame`): Pandas DataFrame with features

    Returns:

    - `pd.DataFrame` with extra column `average_rides_last_4_weeks`
    """
    X["average_rides_last_4_weeks"] = 0.25 * (
        X[f"rides_previous_{7*24}_hour"] + \
        X[f"rides_previous_{2*7*24}_hour"] + \
        X[f"rides_previous_{3*7*24}_hour"] + \
        X[f"rides_previous_{4*7*24}_hour"]   
    )

    return X


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X_ = X.copy()

        # Generate numeric columns from datetime
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek

        return X_.drop(columns=["pickup_hour"])

def create_centroid_coords(
                           file_name: str = "nyc_taxi_zones.zip",
                           crs: str = 2263) -> pd.DataFrame:
    """
    Read in a shapefile of NYC taxi zones. Compute the centroid for each polygon and extract the latitude and longitude. Returns the location id and coordinates.

    Returns:
    - `gdf` (`gpd.GeoDataFrame`) - GeoDataFrame containing the location id and coordinates.    
    """
    try:# Read in the shapefile
        gdf = gpd.read_file(RAW_DATA_DIR / file_name)

        # Reproject to a suitable projection for distance calculations
        gdf["geometry"] = gdf["geometry"].to_crs(crs=crs)

        
        # Compute the centroid of each polygon and convert to lat/lon
        gdf["centroid"] = gdf["geometry"].centroid.to_crs(epsg=4326)
    
        # Extract the latitude and longitude of the centroid
        gdf["latitude"] = gdf["centroid"].y
        gdf["longitude"] = gdf["centroid"].x

        result = gdf[["location_i", "latitude", "longitude"]]
        # Rename location_i to location_id
        result = result.rename(columns={"location_i": "location_id"})

        return result
    except Exception as e:
        print(f"An error occured: {e}")
        return None

def convert_gdf_to_dataframe(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Converts a GeoDataFrame to a DataFrame.

    Args:
    - `gdf` (`gpd.GeoDataFrame`) - GeoDataFrame to be converted.

    Returns:
    - `pd.DataFrame` - DataFrame containing the same data as the input GeoDataFrame.
    """
    return pd.DataFrame(gdf)


def merge_coords_df_with_training_df(training_df: pd.DataFrame,
                                    coord_df: pd.DataFrame, 
                                    training_id: str,
                                    coord_id: str) -> pd.DataFrame:
    """
    Merges the DataFrame containing centroid coordinates with the training DataFrame. Drops the categorical ids.

    Args:
    - `coord_df` (`pd.DataFrame`) - DataFrame containing the coordinates.
    - `training_df` (`pd.DataFrame`) - DataFrame containing the training DataFrame.
    - `coord_id` (`pd.Series`) - Series with the coordinate DataFrame variable to be used on merge.
    - `training_id` - Series with the training DataFrame varibale to be used on merge.


    Returns:
    - `pd.DataFrame` - Merged DataFrame with coordinates.
    """
    df_merged = pd.merge(
        training_df,
        coord_df,
        how="left",
        left_on=training_df[training_id].name, 
        right_on=coord_df[coord_id].name)
    
    df_merged.drop(columns=[training_id, coord_id], inplace=True)
    return df_merged


class CentroidCoordinateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, file_name="nyc_taxi_zones.zip", crs=2263):
        self.file_name = file_name
        self.crs = crs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assuming create_centroid_coords is correctly defined and imported
        coord_df = create_centroid_coords(file_name=self.file_name, crs=self.crs)
        # Assume X is the training DataFrame and it has a 'location_id' that matches 'location_id' in coord_df
        X = merge_coords_df_with_training_df(X, coord_df, training_id='pickup_location_id', coord_id='location_id')
        return X

def get_pipeline(**hyperparams) -> Pipeline:
    # sklearn transform for adding features based on past ride averages
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks, validate=False)
    
    # sklearn transform for adding temporal features
    add_temporal_features = TemporalFeaturesEngineer()

    # New transformer for adding centroid coordinates
    add_centroid_coordinates = CentroidCoordinateTransformer()

    # sklearn pipeline
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        add_centroid_coordinates,
        lgb.LGBMRegressor(**hyperparams)
    )