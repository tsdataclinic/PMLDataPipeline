'''This script applies a trained model to a specific Scene.

    It can be run using 
    
    python predict_scene.py -scene scenes/gulf.json -model models/median_20_both_L1C 

    with the following arguments 
    - scene: The scene definition file
    - model: The folder of the model to run
'''
from plasticfinder.workflows import predict_using_model
from plasticfinder.viz import plot_classifications
from datetime import datetime
from pathlib import Path
import pandas as pd
import json 
import argparse

# features  = pd.read_csv('./data/augmented_features.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to download a given region and process it in to EOPatches')
    parser.add_argument('--scene',  type=str, help='Scene specificaiton file')
    parser.add_argument('--model', type=str, help='Model directory for the model to use.')
    parser.add_argument('--method',  type=str, help='The normalization method used to train the model')
    parser.add_argument('--window',  type=int, help='The window size used to train the model')

    args = parser.parse_args()
    
    try:
        with open(args.scene, 'r') as f:
            scene = json.load(f)
    except:
        raise Exception("Could not read scene file")
    
    model = Path(f'{args.model}/model.joblib')
    method = args.method
    window = arcs.window

    for file in Path(scene['outputDir']).glob('*'):
        print('predicting for ', file )
        predict_using_model(scene['outputDir'],file,model,method,window)
        plot_classifications(file, None)

