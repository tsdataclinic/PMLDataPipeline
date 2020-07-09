from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB 
from matplotlib import pylab
import pandas as pd
import argparse
from eolearn.core import LoadTask
from plasticfinder.tasks.local_norm import LocalNormalization
from plasticfinder.class_defs import catMap
import matplotlib.pyplot as plt
import joblib
import math 

def train_model_and_test(x_train,y_train,x_test,y_test,model=None, model_name=None):
    ''' Will train a model with the specified data and save the results to the specified folder.

        Parameters:
            x_train: A DataFrame or np array containing the training features
            y_train: A DataFrame or np array containing the training labels
            x_test: A DataFrame or np array containing the test features
            y_test: A DataFrame or np array containing the test labels
            model: A scikit learn like model to use, defaults to GaussianNB
            model_name: A name used to save the model in the models folder.

        Returns:
            Nothing but will create a directiory in models/{model_name} which includes a joblib serialized 
            version of the trained model and some plots that show the performance of the model.
     '''

    if(model_name):
        models_dir = Path('models')
        model_dir = models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
    if(model ==None):
        model = GaussianNB()
    
    model.fit(x_train, y_train) 
    fig,axs = plt.subplots(ncols=2, nrows=2, figsize=(15,10))
    axs = axs.flatten()
    
    plot_confusion_matrix(model,x_train, y_train,ax=axs[0], normalize='true')
    axs[0].set_xticklabels(list(cats.keys()))
    axs[0].set_yticklabels(list(cats.keys()))
    axs[0].xaxis.set_tick_params(rotation=45)


    plot_confusion_matrix(model,x_test, y_test,ax=axs[1], normalize='true')
    axs[1].set_xticklabels(list(cats.keys()))
    axs[1].set_yticklabels(list(cats.keys()))
    axs[1].xaxis.set_tick_params(rotation=45)

    plot_confusion_matrix(model,x_train, y_train,ax=axs[2], normalize=None)
    axs[2].set_xticklabels(list(cats.keys()))
    axs[2].set_yticklabels(list(cats.keys()))
    axs[2].xaxis.set_tick_params(rotation=45)


    plot_confusion_matrix(model,x_test, y_test,ax=axs[3], normalize=None)
    axs[3].set_xticklabels(list(cats.keys()))
    axs[3].set_yticklabels(list(cats.keys()))
    axs[3].xaxis.set_tick_params(rotation=45)

    plt.tight_layout()
    
    if(model_name):
        plt.savefig(model_dir / "confusion.png")
        plt.close(fig)
    
        joblib.dump(model,model_dir / "model.joblib")
    
    return model, model.predict(x_train), model.predict(x_test)


def load_and_apply_local_norm(feature_index,method, window_size):

    '''A function to apply the local normalization step to each feature
        
                Parameters:
                        feature (GeoSeries): A row from the GeoDataFrame produced by load_fetures_from_file
                        feature_index (int): The integer used in saving the EOPatch to disk.
                        method: One of 'min', 'median' or 'mean' indicating the type of averaging the window function should use.
                        window_size: The extent in pixles that averaging should carried out over. 

                Returns:
                       EOPatch including the normalized data
    '''
    load_task = LoadTask(path=f'data/Training/feature_{feature_index}/')
    local_norm = LocalNormalization()
    
    workflow = LinearWorkflow(load_task, local_norm)
    patch = workflow.execute({
        local_norm: {
            'method' : method,
            'window_size': window_size
        }
    })
    return patch

def load_features(method,window_size):

    issue_files = []
    train = pd.DataFrame()
    for feature_index in range(0, features.shape[0],1):
        try:
            feature = features.iloc[feature_index]
            patch =load_and_apply_local_norm(feature_index,method, window_size)
            data = patch['data']
            bands_L1C = data['BANDS-S2-L1C']
            bands_L2A = data['BANDS-S2-L2A']
            center_x = math.floor(bands_L1C.shape[1]/2)
            center_y = math.floor(bands_L1C.shape[2]/2)

            ndvi = data['NDVI'][0,center_x,center_y,0]
            fdi = data['FDI'][0,center_x,center_y,0]
            normed_ndvi = data["NORM_NDVI"][0,center_x,center_y,0]
            normed_fdi = data["NORM_FDI"][0,center_x,center_y,0]
            spectra = data['BANDS-S2-L2A'][0,center_x,center_y,:]
            metrics = {
                 'ndvi': ndvi, 
                 'label':feature.reduced_label, 
                 'fdi': fdi, 
                 'normed_ndvi' : normed_ndvi, 
                 'normed_fdi': normed_fdi, 
                 'Lat':feature.Lat, 
                 'Lon': feature.Lon
            }
            band_cols_L1C = dict(zip(
                ['B01_L1C', 'B02_L1C', 'B03_L1C', 'B04_L1C', 'B05_L1C', 'B06_L1C', 'B07_L1C', 'B08_L1C', 'B08A_L1C', 'B09_L1C', 'B10_L1C', 'B11_L1C', 'B12_L1C'],
                bands_L1C[0,center_x,center_y,:]
            ))
            
            band_cols_L2A = dict(zip(
                ['B01_L2A', 'B02_L2A', 'B03_L2A', 'B04_L2A', 'B05_L2A', 'B06_L2A', 'B07_L2A', 'B08_L2A', 'B08A_L2A', 'B09_L2A', 'B10_L2A', 'B11_L2A', 'B12_L2A'],
                bands_L2A[0,center_x,center_y,:]
            ))

            train = train.append( pd.Series(
                {**metrics, **band_cols_L1C, **band_cols_L2A}, name =feature_index))
        except Exception as e :
            print(e)
            issue_files.append(feature_index)
    return (train, issue_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train a model using a specific noramization scheme')
    parser.add_argument('--method',  type=str, help='Normalization method, one of median, mean or min')
    parser.add_argument('--window_size', type=int, help='Normalization window in meters')
    parser.add_argument('--name',  type=str, help='Name for the model')

    args = parser.parse_args()

    window_size_px = math.floor(args.window_size/10.0)

    train, issue_files  = load_features(args.method, window_size_px)
    train = train.assign(label_cat = train.label.apply(lambda x: catMap[x]))
    train = train.dropna(subset=['normed_ndvi'])

    X_train, X_test, Y_train, Y_test = train_test_split(train[['normed_ndvi','normed_fdi', 'B06_L1C', 'B07_L1C', 'B11_L1C']], train['label_cat'], stratify=train['label_cat'])
    model, prediction_train, prediction_test = train_model_and_test(X_train,Y_train, X_test,Y_test, model_name=args.name)