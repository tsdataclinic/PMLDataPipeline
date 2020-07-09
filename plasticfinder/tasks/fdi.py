from eolearn.core import  EOTask
from eolearn.core import FeatureType
import numpy as np

DEFAULT_BAND_NAMES= ['B01', 
                 'B02', 
                 'B03', 
                 'B04', 
                 'B05', 
                 'B06', 
                 'B07', 
                 'B08', 
                 'B08A', 
                 'B09', 
                 'B10', 
                 'B11', 
                 'B12']

class CalcFDI(EOTask):
    ''' EOTask that calculates the floating debris index see https://www.nature.com/articles/s41598-020-62298-z

        Expectes the EOPatch to have either Sentinel L1C or 
        L2A bands.

        Will append the data layer "FDI" to the EOPatch

        Run time parameters:
            - band_layer(str): the name of the data layer to use for raw Sentinel bands
            - band_names(str): the names of each band B01, B02 etc
    '''
    
    @staticmethod
    def FDI(NIR,RE,SWIR):
        factor = 1.636
        return NIR - ( RE + ( SWIR - RE )*factor)
    
        
    def execute(self,
                eopatch,
                band_layer='BANDS-S2-L1C', 
                band_names=DEFAULT_BAND_NAMES
                ):
        bands  = eopatch.data[band_layer]
        
        NIR = bands[:,:,:,band_names.index('B08')]
        RE  = bands[:,:,:,band_names.index('B05')]
        SWIR = bands[:,:,:,band_names.index('B11')]
        
        FDI = self.FDI(NIR,RE,SWIR).reshape([bands.shape[0], bands.shape[1], bands.shape[2], 1])        
        
        eopatch.add_feature(FeatureType.DATA, 'FDI', FDI)
        return eopatch