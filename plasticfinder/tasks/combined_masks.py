from eolearn.core import EOTask
from eolearn.core import FeatureType

class CombineMask(EOTask):
    ''' Simple task to combine the various masks in to one

        Run time parameters passed on workflow execution: 
        use_water(bool): Include the water mask as part of the full mask. Default is false
    '''
    
    def execute(self,eopatch, use_water=True):
        if(use_water):
            combined = np.logical_or( np.invert(eopatch.mask['WATER_MASK']).astype(bool),eopatch.mask['CLM'].astype(bool) )
        else:
            combined = eopatch.mask['CLM'].astype(bool)
        eopatch.add_feature(FeatureType.MASK, 'FULL_MASK', combined ) #np.invert(eopatch.mask['WATER_MASK']) & eopatch.mask['CLM'] & np.invert(eopatch.mask['IS_DATA']))
        return eopatch
