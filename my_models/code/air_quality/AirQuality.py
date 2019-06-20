import numpy as np
import pandas as pd
from matplotlib.path import Path
import mlflow.pyfunc


def _is_in_polygon_2D(x, path_vertices):
    '''Return if a number of vertices are contained in a given path of vertices
    
    Input x is a numpy array of size [n, 2] for n points
    Output is also a numpy array
    '''
    # Note: if we want curved points use this solution:
    # https://matplotlib.org/gallery/shapes_and_collections/path_patch.html#sphx-glr-gallery-shapes-and-collections-path-patch-py
    n = len(path_vertices)

    # repeat the first point in the end and close path
    codes = [Path.MOVETO] + [Path.LINETO]*(n-1) + [Path.CLOSEPOLY]
    path_vertices = path_vertices + [path_vertices[0]]

    path_vertices = np.array(path_vertices, float)
    path = Path(path_vertices, codes)

    return path.contains_points(x)
    
    
class AirQuality(mlflow.pyfunc.PythonModel):
   
    def __init__(self):
        # list of polygon points ordered in circle
        # each element is a tupple of values (temperature, humidity)
        self.vertices_good_air_quality = [(17.5, 75), (22.5, 65), (24, 35), (19, 38)]
        self.vertices_acceptable_air_quality = [(16, 75), (17, 85), (22, 80), (25, 60), (27, 30), (26, 20), (20, 20), (17, 40)]

    def air_quality_good(self, measurements_df):
        return _is_in_polygon_2D(measurements_df, self.vertices_good_air_quality)

    def air_quality_acceptable(self, measurements_df):
        return _is_in_polygon_2D(measurements_df, self.vertices_acceptable_air_quality)

    def predict(self, model_input_df):
        print('Evaluating air quality')
        input_data = model_input_df.loc[:, ['temperature', 'humidity']].values
        is_good = self.air_quality_good(input_data)
        is_acceptable = self.air_quality_acceptable(input_data)
        output_df = pd.DataFrame(columns=['AQ_good', 'AQ_acceptable'], data=np.stack((is_good, is_acceptable), axis=1), index=model_input_df.index)
        return output_df
    