from eolearn.features import NormalizedDifferenceIndexTask
from eolearn.core import FeatureType

def ndvi_task(band_types = 'BANDS-S2-L1C'):
    ''' EOTask that calculates the NDVI values 

        Expectes the EOPatch to have either Sentinel L1C or 
        L2A bands.

        Will append the data layer "NDVI" to the EOPatch

        Run time parameters:
            - band_types(str): the name of the data layer to use for raw Sentinel bands
    '''
    
    if(band_types == 'BANDS-S2-L1C'):
        band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B08A', 'B09', 'B10', 'B11', 'B12']
    else:
        band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B08A', 'B09', 'B11', 'B12']
        
    ndvi = NormalizedDifferenceIndexTask((FeatureType.DATA, band_types), (FeatureType.DATA, 'NDVI'),
                                     [band_names.index('B08'), band_names.index('B04')])
    return ndvi
