import pathlib
import numpy as np

from utils import *

pred_fpath = pathlib.Path(
    'G:/Cochlea/Manual_Segmentations/latest_inference/pred_train_15.hdf5'
)
data_fpath = pathlib.Path(
    'G:/Cochlea/Manual_Segmentations/fpvct_99_mu_transduced/15.hdf5'
    )

assert data_fpath.is_file()
assert pred_fpath.is_file()


r = Result(data_path=data_fpath, prediction_path=pred_fpath, threshold=0.5)

print(r.get_array_shape(data_fpath, 'raw/raw-0'))

export_path = pathlib.Path(r'C:\Users\Jannik\Desktop\june_results\data\export\test.nrrd')

r.export_prediction(export_path)