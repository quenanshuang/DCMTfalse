1.Data availability

The DRIVE retinal vessel segmentation dataset can be obtained from this link https://drive.grand-challenge.org/ ,and is included in this published article: "Staal, J., Abràmoff, M. D., Niemeijer, M., Viergever, M. A., & Van Ginneken, B. (2004). Ridge-based vessel segmentation in color images of the retina. IEEE transactions on medical imaging, 23(4), 501-509." 

The STARE retinal vessel segmentation dataset can be obtained through this link https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset ,and is included in this published article: " Hoover A D, Kouznetsova V, Goldbaum M. Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response[J]. IEEE Transactions on Medical imaging, 2000, 19(3): 203-210. "



2. Data Preparation

Please make sure the dataset is arranged in the following format:

DATA/
|-- DRIVE
|   |-- TrainDataset
|   |   |-- images
|   |   |-- masks
|   |-- TestDataset
|   |   |-- images
|   |   |-- masks

2.Training

python train.py

3.test

python eval.py
