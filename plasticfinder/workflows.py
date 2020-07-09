from eolearn.core import  EOTask
from plasticfinder.tasks.combined_masks import CombineMask
from plasticfinder.tasks.cloud_classifier import cloud_classifier_task
from plasticfinder.tasks.water_detection import WaterDetector
from plasticfinder.tasks.ndwi import ndwi_task
from plasticfinder.tasks.ndvi import ndvi_task
from plasticfinder.tasks.fdi import CalcFDI
from plasticfinder.tasks.input_tasks import input_task,true_color,add_l2a
from plasticfinder.tasks.local_Norm import LocalNormalization
from plasticfinder.tasks.detect_plastics import DetectPlastics
from plasticfinder.class_deffs import catMap, colors

import numpy as np
import geopandas as gp
import contextily as cx
import matplotlib.pyplot as plt 
from eolearn.core import SaveTask, LinearWorkflow, LoadTask
from eolearn.core.constants import OverwritePermission
from plasticfinder.viz import plot_ndvi_fid_plots, plot_masks_and_vals
from sentinelhub import UtmZoneSplitter, BBoxSplitter,BBox, CRS
from shapely.geometry import box, Polygon, shape

def get_and_process_patch(bounds,time_range, base_dir,index):
    ''' Defines a workflow that will download and process a specific EOPatch.

        The pipline has the folowing steps 
            - Download data 
            - Calculate NDVI 
            - Calculate NDWI
            - Calculate FDI
            - Add cloud mask
            - Add water mask
            - Combine all masks
            - Perform local noramalization
            - Save the results.

        Parameters:
            - bounds: The bounding box of the EOPatch we wish to process
            - time_range: An array of [start_time,end_time]. Any satelite pass in that range will be procesed.
            - base_dir: the directory to save the patches to 
            - index: An index to label this patch
        
        Returns:
            The EOPatch for this region and time range.
    '''
    save = SaveTask(path=f'{base_dir}/feature_{index}/', overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    
    add_fdi = CalcFDI()
    water_detection= WaterDetector()
    combine_mask= CombineMask()
    local_norm = LocalNormalization()
    
    fetch_workflow = LinearWorkflow(input_task,
                                    true_color,
                                    add_l2a,
                                    ndvi_task(),
                                    ndwi_task(),
                                    add_fdi,
                                    cloud_classifier_task(),
                                    water_detection,
                                    combine_mask,
                                    local_norm,save )

    feature_result = fetch_workflow.execute({
        input_task: {
            'bbox':BBox(bounds, CRS.WGS84),
            'time_interval': time_range
        },
        combine_mask:{
            'use_water': False
        },
        local_norm:{
            'method':'min',
            'window_size': 10,
        }
    })
    patch = feature_result.eopatch()
    return patch 
    
def download_region(base_dir, minx,miny,maxx,maxy,time_range, target_tiles=None):
    ''' Defines a workflow that will download and process all EOPatces in a defined region.

        This workflow will download all EOPatches in a given larger region. 

        The pipline has the folowing steps 
            - Download data 
            - Calculate NDVI 
            - Calculate NDWI
            - Calculate FDI
            - Add cloud mask
            - Add water mask
            - Combine all masks
            - Perform local noramalization
            - Save the results.

        Parameters:
            
            - base_dir: the directory to save the patches to 
            - minx: Min Longitude of the region 
            - miny: Min Latitude of the region 
            - maxx: Max Longitude of the region 
            - maxy: Max Latitude of the region
            - time_range: An array of [start_time,end_time]. Any satelite pass in that range will be procesed.
            - target_tiles: A list of tiles to manually include (not used right now)
       
        Returns:
            Nothing.
    '''
    
    region = box(minx,miny,maxx,maxy)
    bbox_splitter = BBoxSplitter([region],CRS.WGS84, (20, 20) )

    bbox_list = np.array(bbox_splitter.get_bbox_list())
    info_list = np.array(bbox_splitter.get_info_list())

    # Prepare info of selected EOPatches
    geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
    idxs_x = [info['index_x'] for info in info_list]
    idxs_y = [info['index_y'] for info in info_list]
    ids = range(len(info_list))

    gdf = gp.GeoDataFrame({'index': ids,'index_x': idxs_x, 'index_y': idxs_y},
                               crs='EPSG:4326',
                               geometry=geometry)
    
    ax= gdf.to_crs('EPSG:3857').plot(facecolor='w', edgecolor='r', figsize=(20,20),alpha=0.3)

    for idx,row in gdf.to_crs("EPSG:3857").iterrows():
        geo = row.geometry
        xindex = row['index_x']
        yindex = row['index_y']
        index = row['index']
        ax.text(geo.centroid.x, geo.centroid.y, f'{index}', ha='center', va='center')

    cx.add_basemap(ax=ax)
    plt.savefig(f'{base_dir}/region.png')
    
    total = len(target_tiles) if target_tiles else len(bbox_list)

    for index, patch_box in enumerate(bbox_list):
        if(target_tiles and index not in target_tiles):
            continue
        print("Getting patch ", index, ' of ', total)
        try:
            patch = get_and_process_patch(patch_box,time_range,base_dir,index)
            fig,ax = plot_masks_and_vals(patch)
            fig.savefig(f'{base_dir}/feature_{index}/bands.png')
            plt.close(fig)

            fig,ax = plot_ndvi_fid_plots(patch)
            fig.savefig(f'{base_dir}/feature_{index}/ndvi_fdi.png')
            plt.close(fig)
        except:
            print("Failed to process ", index)
                    
def predict_using_model(patch_dir,model_file,method,window_size):
    ''' Defines a workflow that will perform the prediction step on a given EOPatch.

        For a given EOPatch, use the specified model to apply prediction step.
        
        Parameters:
            
            - patch_dir: the directory that contains the patch
            - model_file; the path to the model file.
            - method: The local noramalization method, one of 'min', 'median' or 'mean'. This should be the same as the one used to train the model. 
            - window_size: The window_size used in the local normalization step. Should be the same as that used to train the model. 
            
       
        Returns:
            Nothing. Updates the EOPatch on disk.
    '''
    
    path = patch_dir
    if(type(path)!= str):
        path = str(path)
    save = SaveTask(path=path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    load_task = LoadTask(path=path)
    local_norm = LocalNormalization()

    
    detect_plastics = DetectPlastics(model_file= model_file) 
    workflow = LinearWorkflow(load_task, local_norm, detect_plastics, save)
    workflow.execute({
        local_norm: {
            'method' : method,
            'window_size': window_size
        }
    })
    
def extract_targets(patchDir):
    path = patchDir
    if(type(path)!= str):
        path = str(path)
        
    patch =  LoadTask(path=path).execute()
    box = patch.bbox

    classification = patch.data['CLASSIFICATION'][0,:,:,0]
    print(classification)
    for coord in np.argwhere(classification == catMap['Debris']):
        print(coord)
    