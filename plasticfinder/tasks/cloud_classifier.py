from eolearn.mask import get_s2_pixel_cloud_detector , AddCloudMaskTask

def cloud_classifier_task():
    '''A convenience function that sets up the cloud detection task. 
    
       Configures an instance of the EOTask s2_pixel_cloud_detector and AddCloudMaskTask
    '''
    
    cloud_classifier = get_s2_pixel_cloud_detector(average_over=2, dilation_size=1, all_bands=False)
    cloud_detection = AddCloudMaskTask(
      cloud_classifier, 
      'BANDS-S2CLOUDLESS', 
      cm_size_y='40m', 
      cm_size_x='40m',
      cmask_feature='CLM', 
      cprobs_feature='CLP' 
    )
    return cloud_detection