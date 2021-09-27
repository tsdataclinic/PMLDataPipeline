'''This script downloads all the tiles for a given scene. 
   It can be run as follows 

   python download_scene.py -scene scenes/ghana.json

   The format of the input specification file is json and should have the following keys 

   - minLat: The minimum latitude of the region you are interested in 
   - maxLat: The maxium latitude of the region you are interested in
   - minLon: The minimum longitude of the region you are interested in 
   - maxLon: The maxium longitude of the region you are interested in
   - timeRange: An array of [start_date, end_date]. All Sentinel passes within this timerange will be downloaded.
   - outputDir: Where the data should be downloaded.

   For an example see the file scenes/gulf.json

'''

from plasticfinder.workflows import download_region
from datetime import datetime
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to download a given region and process it in to EOPatches')
    parser.add_argument('--scene',  type=str, help='Scene specificaiton file')
    args = parser.parse_args()
    try:
        with open(args.scene, 'r') as f:
            scene = json.load(f)
    except:
        raise Exception("Could not read scene file")
    
    timeRange = [
        datetime.strptime(scene['dateRange'][0], "%Y-%m-%d"),
        datetime.strptime(scene['dateRange'][0], "%Y-%m-%d")
    ]

    download_region(scene['outputDir'],scene['minLon'],scene['minLat'],scene['maxLon'],scene['maxLat'],timeRange)

