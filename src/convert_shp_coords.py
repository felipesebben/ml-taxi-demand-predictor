import geopandas as gpd
import pandas as pd
from src.paths import RAW_DATA_DIR

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
        left_on=training_df[training_id].name, 
        right_on=coord_df[coord_id].name)
    
    df_merged.drop(columns=[training_id, coord_id], inplace=True)
    return df_merged