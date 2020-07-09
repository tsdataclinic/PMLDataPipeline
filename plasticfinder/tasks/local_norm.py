from scipy.ndimage.filters import median_filter, uniform_filter, generic_filter
from skimage.filters.rank import mean, median
from eolearn.core import  EOTask
from eolearn.core import FeatureType
import numpy as np

class LocalNormalization(EOTask):
    '''EOPatch that performs a local noramalization of FDI and NDVI values

       This task will generate a moving average over the EOPatch of NDVI and FDI 
       parameters and subtract these from each pixel to normalize the FDI and NDVI 
       relationship.

       The task expects there to be an NDVI and FDI data layer along with a layer 
       for Sentinel satellite data.

       Appends the following laters to the EOPatch
            NORM_FDI: Normalized FDI values.
            NORM_NDVI: Noralized NDVI values.
            MEAN_FDI:  The windowed average FDI, used mostly for visualization.
            MEAN_NDVI: The windowed average NDIV, used mostly for visualization.
            NORM_BANDS: Each Sentinel band normalized

       Run time arguments:
            - method: the normalization method, one of min,median,mean
            - window_size: the window over which to perform the normalization in pixles
    ''' 
    
    @staticmethod
    def normalize(data, mask, method='mean',axis=[1,4], window_size=2):
        masked_data = np.where(np.invert(mask),data,np.nan)
        
        result = np.zeros(shape=masked_data.shape)
        norm_scene = np.zeros(shape=result.shape)

        for time_bin in range(data.shape[0]):
            for freq_bin in range(data.shape[3]):
                   
                scene  = masked_data[time_bin,:,:,freq_bin]
                if(method == 'mean'):
                    norm = generic_filter(scene, np.nanmean, size=window_size)
                elif(method == 'median'):
                    norm = generic_filter(scene, np.nanmedian, size=window_size)
                elif(method == 'min'):
                    norm = generic_filter(scene,np.nanmin, size=window_size)
                else:
                    raise Exception("Method needs to be either mean, median or min")
                result[time_bin,:,:,freq_bin] = scene - norm
                norm_scene[time_bin,:,:, freq_bin] = norm
    
        return np.array(result), np.array(norm_scene)

    def execute(self,eopatch, method='mean', window_size=100):
        valid_mask = np.copy(eopatch.mask['FULL_MASK'])

        normed_ndvi, m_ndvi = self.normalize(eopatch.data['NDVI'], valid_mask, method=method, window_size=window_size)

        normed_fdi,m_fdi = self.normalize(eopatch.data['FDI'], valid_mask, method=method, window_size=window_size)
        normed_bands = self.normalize(eopatch.data['BANDS-S2-L1C'], eopatch.mask['WATER_MASK'],self.method)
        
        eopatch.add_feature(FeatureType.DATA, 'NORM_FDI', normed_fdi)
        eopatch.add_feature(FeatureType.DATA, 'NORM_NDVI', normed_ndvi)
        eopatch.add_feature(FeatureType.DATA, 'MEAN_FDI', m_fdi.reshape(eopatch.data['FDI'].shape))
        eopatch.add_feature(FeatureType.DATA, 'MEAN_NDVI', m_ndvi.reshape(eopatch.data['NDVI'].shape))
        eopatch.add_feature(FeatureType.DATA, 'NORM_BANDS', nomed_bands.reshape(eopatch.data['NDVI'].shape))

        return eopatch

local_norm = LocalNormalization()