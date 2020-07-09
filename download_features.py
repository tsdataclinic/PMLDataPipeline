'''This script downloads and processes modeling traning data 
   in to a format that is ready for the modeling script.
   To run use the following command

   python download_features -feature_list=data/features.csv -o output_dir
   
   This will generate EOPatches with each training data point at the center and will
   create a folder for each that contains the EOPatch and some plot's to help contextualize 
   the datapoint.

   The expected format for the input file is a csv with one column per feature

   Lat, Lon, date, label

   Where label is on of "Debris", "Water", "Spume", "Timber", "Pumice", "Seaweed"
'''

import pandas as pd 
import geopanads as gp
import argparse
from plasticfinder.tasks.combined_masks import CombineMask
from plasticfinder.tasks.cloud_classifier import cloud_classifier_task
from plasticfinder.tasks.water_detection import WaterDetector
from plasticfinder.tasks.ndwi import ndwi_task
from plasticfinder.tasks.ndvi import ndvi_task
from plasticfinder.tasks.fdi import CalcFDI
from plasticfinder.tasks.input_tasks import input_task,true_color,add_l2a
from plasticfinder.tasks.local_Norm import LocalNormalization
from eolearn.core import SaveTask, LinearWorkflow, LoadTask
from eolearn.core.constants import OverwritePermission


def load_fetures_from_file(file, buffer_x=500, buffer_y=500):
    '''A function to load in the list of feature targets and generates a buffer around each of the specified extent
    
            Parameters:
                    file (str): Location of the file specifying the training targets
                    buffer_x (float): Amount in meters to buffer the point by. Default is 500m.
                    buffer_y (float): Amount in meters to buffer the point by. Default is 500m.

            Returns:
                    features (GeoDataFrame): A GeoPandas GeoDataFrame  
    ''' 

    features = pd.read_csv('data/features.csv')
    features = gp.GeoDataFrame(features,geometry = features.apply(lambda x: Point(x.Lon,x.Lat),axis=1), crs='EPSG:4326')
    bounds  =  gp.GeoSeries(features.to_crs("EPSG:3857").apply(lambda x : box(x.geometry.x - buffer_x, x.geometry.y - buffer_y, x.geometry.x + buffer_x, x.geometry.y + buffer_y),axis=1), crs='EPSG:3857').to_crs('EPSG:4326')
    features = features.assign(date_start= pd.to_datetime(features.date.str[0:10], format='%Y_%m_%d') - timedelta(days=1), date_end = pd.to_datetime(features.date.str[0:10],format='%Y_%m_%d') +timedelta(days=1))

    # features[features['file']=='S2B_MSI_2019_04_24_07_36_19_T36JUN_L2R'].date_start = datetime.datetime(year = 2019, month=5, day=23)
    # features[features['file']=='S2B_MSI_2019_04_24_07_36_19_T36JUN_L2R'].date_end = datetime.datetime(year = 2019, month=5, day=25)
    return features


def process_feature(feature, feature_index):  
        '''A function to download a given target pixel and it's surroundings as an EOPatch
        
                Parameters:
                        feature (GeoSeries): A row from the GeoDataFrame produced by load_fetures_from_file
                        feature_index (int): The integer used in saving the EOPatch to disk.
                        

                Returns:
                        Nothing  
        ''' 

        save = SaveTask(path=f'{base_dir}/feature_{feature_index}/', overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
        train_test_workflow = LinearWorkflow(input_task,true_color,add_l2a,ndvi,ndwi,add_fdi,cloud_detection,water_detection,combine_mask,save )

        feature_result = train_test_workflow.execute({
            input_task: {
                'bbox':BBox(bounds.iloc[feature_index],bbox_list[0].crs),
                'time_interval': [feature.date_start, feature.date_end]
            },
            combine_mask:{
                'use_water': False #(target.reduced_label != 'Timber')
            },
            add_fdi:{
                'band_layer': USE_BANDS,
                'band_names': band_names
            }
        })
        patch = feature_result.eopatch()
        return patch 


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Script to download target features and their surroundings as EOPatches')
    parser.add_argument('--features',  type=str, help='The location of a csv that holds the data for each feature')
    

    args = parser.parse_args()

    feature_file = args.features
    output_dir = 'data/Training'
    
    features  = load_fetures_from_file(feature_file)
   
    for feature_index in range(0, features.shape[0],1):
        target = features.iloc[feature_index]
        try:
            print("running ", feature_index, ' of ', features.shape[0])
            patch = process_feature(target,feature_index)
            fig,axs = plot_masks_and_vals(patch, features.iloc[feature_index:feature_index+1])
            fig.savefig(f'{base_dir}/feature_{feature_index}/mask_vals.png')
            plt.close(fig)
        except:
            print("Could not download feature ", feature_index)

    for feature_index in range(0,features.shape[0],1):
        target = features.iloc[feature_index]
        try:
            patch = apply_local_norm(target,feature_index, method='median', window_size=40)
        except Exception as e :
            print("Could not process feature ", feature_index)
