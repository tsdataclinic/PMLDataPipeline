import numpy as np
import matplotlib.pyplot as plt
from plasticfinder.class_deffs import catMap, colors,cols_rgb
from eolearn.core import LoadTask
from pathlib import Path

# def plot_predictions(patch):
    

def plot_masks_and_vals(patch, points=None, scene=0):
    ''' Method that will take a given patch and plot the various data and mask layers it contains

        Parameters:
            - patch: an EOPatch to visualize
            - points: A list of points to overlay on top of the scene
            - scene: The index of the scene within the EOPatch if there are multiple satelite passes.
        Returns
            (fig,axs) : the output figure and the individual plot axis
    '''

    extent = [ patch.bbox.min_x, patch.bbox.max_x, patch.bbox.min_y, patch.bbox.max_y]
    
    ratio = np.abs(patch.bbox.max_x - patch.bbox.min_x) / np.abs(patch.bbox.max_y - patch.bbox.min_y)
    fig, axs = plt.subplots(3,5,figsize=(ratio * 10 * 2, 10*2))
    axs = axs.flatten()
    
    axs[0].set_title("True Color")
    axs[0].imshow(patch.data['TRUE-COLOR-S2-L1C'][scene])

    axs[1].set_title("NDVI")
    axs[1].imshow(patch.data['NDVI'][scene,:,:,0])
    
    axs[2].set_title("FDI")
    axs[2].imshow(patch.data['FDI'][scene,:,:,0])
    
    axs[3].set_title("NDWI")
    axs[3].imshow(patch.data['NDWI'][scene,:,:,0])

    axs[4].set_title("Data Mask")
    axs[4].imshow(patch.mask['IS_DATA'][scene,:,:,0])
    
    axs[5].set_title("Cloud Mask")
    axs[5].imshow(patch.mask['CLM'][scene,:,:,0])
    
    axs[6].set_title("Water Mask")
    axs[6].imshow(patch.mask['WATER_MASK'][scene,:,:,0])
    
    axs[6].set_title("Water Mask")
    axs[6].imshow(patch.mask['WATER_MASK'][scene,:,:,0])
    
    axs[7].set_title("Normed FDI")
    axs[7].imshow(patch.data['NORM_FDI'][scene,:,:,0])
    
    axs[8].set_title("Normed NDVI")
    axs[8].imshow(patch.data['NORM_NDVI'][scene,:,:,0])
    
    axs[9].set_title("AVG NDVI")
    axs[9].imshow(patch.data['MEAN_NDVI'][scene,:,:,0])
    
    axs[10].set_title("AVG FDI")
    axs[10].imshow(patch.data['MEAN_FDI'][scene,:,:,0])
    
    axs[11].set_title("Combined mask")
    axs[11].imshow(patch.mask['FULL_MASK'][scene,:,:,0])
    
    
   
    axs[12].set_title("Simple cutoff")
    axs[12].imshow( (patch.data['NORM_FDI'][scene,:,:,0] > 0.005)  & ( patch.data['NORM_NDVI'][scene,:,:,0] > 0.1) )
    
    if(points):
        axs[13].set_title('Points')
        axs[13].imshow(patch.data['NORM_FDI'][scene,:,:,0], extent = extent )
        points.plot(ax=axs[13], markersize=20, color='red')
    
    if("SCENE_CLASSIFICATION" in patch.data):
        axs[14].set_title("Labels")
        axs[14].imshow(patch.data['SCENE_CLASSIFICATION'][scene,:,:,0])
        
    elif('CLASSIFICATION' in patch.data):
        classifcations = patch.data['CLASSIFICATION'][scene,:,:,0]    
       
        p_grid = np.array([cols[val] for val in classification.flatten()])

        axs[14].set_title("Labels")
        axs[14].imshow(p_grid.reshape(classifcations.shape[0], classifcations.shape[1],3))

    plt.tight_layout()
    return fig,axs
    
def plot_ndvi_fid_plots(patch):
    ''' Method that will take a given patch and plot NDVI and FDI relationships.

        Parameters:
            - patch: an EOPatch to visualize
        Returns
            (fig,axs) : the output figure and the individual plot axis
    '''
        
    fig, axs = plt.subplots(2,3,figsize=(10*3,10*2))
    axs = axs.flatten()
    

    axs[0].scatter(patch.data['NDVI'].flatten(), patch.data['FDI'].flatten(), s=1.0, alpha=1)# c = p_grid)
    axs[0].set_xlabel("NDVI")
    axs[0].set_ylabel("FDI")

    axs[1].scatter(patch.data['NORM_NDVI'].flatten(), patch.data['NORM_FDI'].flatten(), s=2., alpha=0.8) #c=p_grid)
    axs[1].set_xlabel("NORMED_NDVI")
    axs[1].set_ylabel("NORMED_FDI")
    
    axs[2].scatter(patch.data['NORM_NDVI'].flatten(), patch.data['FDI'].flatten(), s=2.0, alpha=0.8)#c = p_grid)
    axs[2].set_xlabel("NORMED_NDVI")
    axs[2].set_ylabel("FDI")
    
    axs[3].scatter(patch.data['MEAN_NDVI'].flatten(), patch.data['NDVI'].flatten(), s=2.0, alpha=0.8)# c=p_grid)

    axs[3].set_xlabel("MEAN_NDVI")
    axs[3].set_ylabel("NDVI")
    plt.tight_layout()
    return fig,axs
    
def plot_classifications(patchDir, features=None):
    ''' Method that will take a given patch plot the results of the model for that patch.
    
        Parameters:
            - patchDir: the directory of the EOPatch to visualize
            - features: Features, could be the training dataset, to overlay on the scatter plots.

        Returns
            Nothing. Will create a file called classifications.png in the EOPatch folder.
    '''
    patch =  LoadTask(path=str(patchDir)).execute()
    classifcations = patch.data['CLASSIFICATION'][0,:,:,0]    
    ndvi = patch.data['NDVI'][0,:,:,0]    
    fdi = patch.data['FDI'][0,:,:,0]
    norm_ndvi = patch.data['NORM_NDVI'][0,:,:,0]
    norm_fdi = patch.data['NORM_FDI'][0,:,:,0]

    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(20,30))
    axs=axs.flatten()
    
    
    fndvi = norm_ndvi.flatten()
    ffdi = norm_fdi.flatten()
    fclassifications = classifcations.flatten()
    fclassifications[ (ffdi < 0.007)]  = 0
    
    p_grid = np.array([cols_rgb[val] for val in fclassifications])

    axs[0].set_title("Labels")
    axs[0].imshow(p_grid.reshape(classifcations.shape[0], classifcations.shape[1],3))

    axs[1].imshow(patch.data['NDVI'][0,:,:,0])
    axs[1].set_title('NDVI')
    axs[2].imshow(patch.data['FDI'][0,:,:,0])
    axs[2].set_title('FDI')
    
    for cat in colors.keys():
        mask = classifcations == cat
        axs[3].scatter(norm_ndvi[mask].flatten(), norm_fdi[mask].flatten(), c=colors[cat], s=0.5,alpha=0.2)
        if(features):
            features.plot.scatter(x='normed_ndvi', y='normed_fdi',ax=axs[3], color=features.label.apply(lambda l: colors[catMap[l]]))
    
    axs[4].imshow(norm_ndvi)
    axs[4].set_title('Normed NDVI')
    
    axs[5].imshow(norm_fdi)
    axs[5].set_title('Normed FDI')
    
    plt.tight_layout()
    plt.savefig(Path(patchDir) / 'classifications.png')
    plt.close(fig)