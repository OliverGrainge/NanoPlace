import os
import argparse
from pathlib import Path

import pandas as pd 
import pandas as pd
from pyproj import Transformer
from analyse.dataset import * 
from analyse import DatasetAnalyzer
from .utils import save_config


CITIES = [
    'Bangkok',
    'BuenosAires',
    'LosAngeles',
    'MexicoCity',
    'OSL', # refers to Oslo
    'Rome',
    'Barcelona',
    'Chicago',
    'Madrid',
    'Miami',
    'Phoenix',
    'TRT', # refers to Toronto
    'Boston',
    'Lisbon',
    'Medellin',
    'Minneapolis',
    'PRG', # refers to Prague
    'WashingtonDC',
    'Brussels',
    'London',
    'Melbourne',
    'Osaka',
    'PRS', # refers to Paris
]


def add_utm_coords(df, lat_col="lat", lon_col="lon", epsg=None):
    if epsg is None:
        import math
        lon0 = df[lon_col].iloc[0]
        lat0 = df[lat_col].iloc[0]
        zone = int((lon0 + 180) // 6) + 1
        hemisphere = 32600 if lat0 >= 0 else 32700
        epsg = hemisphere + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    east, north = transformer.transform(df[lon_col].values, df[lat_col].values)
    df = df.copy()
    df["easting"] = east
    df["northing"] = north
    return df

def read_dataframes(dataset_folder):
    df = pd.read_csv(dataset_folder+'Dataframes/'+f'{CITIES[0]}.csv')
    df = df.sample(frac=1)  # shuffle the city dataframe
    for i in range(1, len(CITIES)):
        tmp_df = pd.read_csv(
            dataset_folder+'Dataframes/'+f'{CITIES[i]}.csv')
        prefix = i
        tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
        tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
        df = pd.concat([df, tmp_df], ignore_index=True)
    return df.set_index('place_id')

def get_img_name(row):
    city = row['city_id']
    pl_id = row.name % 10**5 
    pl_id = str(pl_id).zfill(7)
    panoid = row['panoid']
    year = str(row['year']).zfill(4)
    month = str(row['month']).zfill(2)
    northdeg = str(row['northdeg']).zfill(3)
    lat, lon = str(row['lat']), str(row['lon'])
    name = "Images/"+city+"/"+city+'_'+pl_id+'_'+year+'_'+month+'_' + \
        northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
    return name

def add_image_paths(df, dataset_folder): 
    df["image_path"] = df.apply(get_img_name, axis=1)
    #df["image_path"] = df["image_path"].apply(lambda x: os.path.join(dataset_folder, x))
    return df

def compute_defaultconfig(dataset_folder: str) -> pd.DataFrame:
    df = read_dataframes(dataset_folder)
    df = add_image_paths(df, dataset_folder)
    df = df.reset_index(drop=False)  # Create a new numeric index
    df = add_utm_coords(df)
    df = df.drop(columns=["lat", "lon", "year", "month", "northdeg", "city_id", "panoid"])
    df = df.rename(columns={"place_id": "class_id"})
    
    # Store metadata in DataFrame attrs
    df.attrs = {
        "dataset_folder": dataset_folder,
    }
    
    return df
    


def parse_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Compute default configuration for a dataset with spatial matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-folder",
        default="/home/oliver/datasets_drive/vpr_datasets/gsv-cities/",
        type=str,
        help="Path to the dataset folder containing images"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    
    config = compute_defaultconfig(args.dataset_folder)
    config_path = Path(__file__).parent / "configs" / "gsvcities" / "config.parquet"
    save_config(config, config_path)

        # Analyse the dataset 
    analyse_dataset = DatasetAnalyzer(config_path)
    analyse_dataset.compute_descriptors() 
    analyse_dataset.get_class_similarities()
    analyse_dataset.get_intra_class_sim_dist()
    analyse_dataset.plot_class_samples()
    analyse_dataset.plot_class_sim_heatmap()
    analyse_dataset.plot_intra_class_sim_dist()
    analyse_dataset.plot_class_embedding_proj()
    


    

