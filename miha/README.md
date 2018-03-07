In order to reproduce the results for final ensemble execute the following scripts.
Note that the paths are hard coded and you might have to change them in order to run the code on your system.

1. prepare train_test split `genetare_validation_split.py`

2. Train the 3 CNN models by executing scripts `train.py` found in directories `inception_resnet`, `inception_v3`, `xception`
    
3. Extract features from CNN models by executing scrips `extract_features.py` found in directories `inception_resnet`, `inception_v3`, `xception`

4. Train 6 layer2 models `l2_<model>_<10k|12k>.py` in directory `layer2_models`
  
5. Extract features for test set by running `./layer2_models/prepare_layer2_test.py`

6. Generate numpy memory maps that are used for final blend by running `generate_prediciton_memmap_<10k|12k>.py`

Additionally, Dual Path networks were trained and used as part of the final submission. Directory `DPN92` contains the code to train the model up to checkpoint 66. After that the model was retrained by Heng. This part of the code relies on MxNet DPN implementation and trained models found [here](https://github.com/cypw/DPNs). For effiecient data loading and preprocessing Tensorflow records are used. To generate the records run `generate_tf_records.py`. Train the models by running the the scripts: `train_DPN92_part<1-6>.py`.
