from eolearn.core import  EOTask
import numpy as np
from eolearn.core import FeatureType

class WaterDetector(EOTask):
    """
        Very simple water detector based on NDWI threshold.

        Adds the mask layer "WATER_MASK" to the EOPatch.

        Expects the EOPatch to have an "NDWI" layer.

        Run time arguments:
            - threshold(float): The cutoff threshold for water.
        
    """
    
    @staticmethod
    def detect_water(ndwi,threshold):  
        return ndwi > threshold

    def execute(self, eopatch, threshold=0.5):
        water_masks = np.asarray([self.detect_water(ndwi[...,0], threshold) for ndwi in eopatch.data['NDWI']])
        eopatch.add_feature(FeatureType.MASK, 'WATER_MASK', water_masks.reshape([water_masks.shape[0], water_masks.shape[1], water_masks.shape[2], 1]))
        return eopatch

