* [Introduction](#3d-ucaps-3d-capsules-unet-for-volumetric-image-segmentation)
* [Usage](#usage)
* [Model Zoo](#model-zoo)

# Fine Grained Classification using Mixture of Pretrained model

3D-UCaps is a voxel-based Capsule network for medical image segmentation. Our architecture is based on the symmetry U-net with two parts: the encoder forms by Capsule layers, whereas the decoder contains traditional convolutional layers. 3D-UCaps, therefore inherits the merits from both Capsule networks to preserve the part-to-whole relationship and CNNs to learn translational invariant representation. We conducted experiments on various datasets (including iSeg-2017, LUNA16, Hippocampus, and Cardiac) to demonstrate the superior performance of 3D-UCaps, where our method outperforms the baseline method [SegCaps](https://github.com/lalonderodney/SegCaps) while being more robust against rotational transformation when compared to 3D-Unet.

![alt text](MoPT.png "Mixture of Pretrained model")

Details of the UCaps model architecture and experimental results can be found in our [following paper](https://rdcu.be/cyhMv):


**Please CITE** our paper when UCaps is used to help produce published results or incorporated into other software

## Usage

### Installation
Install dependencies : 
```
pip install requirements.txt
```

### Data preparation
Download datasets:
* FGVC-Aircraft dataset: <https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft?select=fgvc-aircraft-2013b>

We expect the directory structure to be the following:
```
fgvc-aircraft-2013b/
  dfgvc-aircraft-2013b/
    data/ 
      images/
```

**Note**: there are some files in LUNA16 dataset can lead to an error when training so we have removed it:
```
1.3.6.1.4.1.14519.5.2.1.6279.6001.771741891125176943862272696845.mhd
1.3.6.1.4.1.14519.5.2.1.6279.6001.927394449308471452920270961822.mhd
```

### Training

Arguments for training can be divided into 3 groups:

1. Trainer args to initialize [Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api) class from Pytorch Lightning.

   * Important arguments: `gpus`, `accelerator`, `check_val_every_n_epoch`, `max_epochs`.
   * Fix arguments in `train.py`: `benchmark`, `logger`, `callbacks`, `num_sanity_val_steps`, `terminate_on_nan` 
2. Model args depend on which model you use (UCaps, SegCaps or U-net) and defined in `add_model_specific_args` method of that module. 

   * Important arguments: `in_channels`, `out_channels`, `val_frequency`, `val_patch_size`, `sw_batch_size`, `overlap`. The last three args are use in [sliding window inference](https://docs.monai.io/en/latest/inferers.html#sliding-window-inference) method from MONAI library.
3. Args specific for training: `root_dir`, `log_dir`, `dataset`, `fold`, `cache_rate`, `cache_dir`, `model_name`, `train_patch_size`, `num_workers`, `batch_size`, `num_samples`.

   * `cache_rate` and `cache_dir` define whether you want to use [CacheDataset](https://docs.monai.io/en/latest/data.html?highlight=ThreadBuffer#cachedataset) or [PersistentDataset](https://docs.monai.io/en/latest/data.html?highlight=ThreadBuffer#persistentdataset) when loading data.
   * `num_samples` is a arg in [RandCropByPosNegLabel](https://docs.monai.io/en/latest/transforms.html#randcropbyposneglabel) method, the effective batch size is `batch_size` x `num_samples`.

The full list of arguments can be shown through the command:
```
python train.py -h
```

We provide bash script with our config to train UCaps model on all datasets and can be run as follow:
```
bash scripts/train_ucaps_iseg.sh
```

### Validation
Arguments for validation can be divided into 3 groups:

1. Trainer args to initialize [Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api) class. The only argument we need to use here is `gpus`.
2. Args for [sliding window inference](https://docs.monai.io/en/latest/inferers.html#sliding-window-inference) method
3. Args specific for validation `root_dir`, `output_dir`, `save_image`, `model_name`, `dataset`, `fold`, `checkpoint_path`

The full list of arguments can be shown through the command:
```
python evaluate.py -h
```

We provide bash script with our config to validate trained UCaps models on all datasets, you just need to download our models in [Model Zoo](#model-zoo) and put them in `logs` folder. After that, you can run the evaluation script for targeted dataset as follow:
```
bash scripts/evaluate_ucaps_iseg.sh
```

### Rotation experiment
Same with validation but add two more arguments `rotate_angle` (in degree) and `axis` (z/y/x or all) to create test rotated subject.

The full list of arguments can be shown through the command:
```
python evaluate_iseg.py -h
```

We provide bash script with our config to compare between trained UCaps ([download](https://drive.google.com/file/d/1WdyGlopAoI-nDVuqfKOaFOXmQUSRD0FA/view?usp=sharing)) and U-net ([download](https://drive.google.com/file/d/1kR-AkjPp36hOTrZ_4ZdtgIpP4PQouHSB/view?usp=sharing)) on subject 9th of iSeg-2017 dataset, the first arugment is `rotate_angle` and the second argument is `axis`:
```
bash scripts/evaluate_rotation.sh 0 z
```

### Rotation experiment on SkipDenseSeg model
1. Cloning [SkipDenseSeg project](https://github.com/tbuikr/3D-SkipDenseSeg)
2. Replacing their `val.py` with our `val.py`
3. Running `val.py` with args, for example:

```
python val.py --gpu 1 --sw_batch_size 32 --overlap 0.75 --output_dir=/home/ubuntu/
```

## Model Zoo
**About the code** This repository has been refactored to use Pytorch Lightning framework and MONAI library for data preprocessing, data loading, inferencing to ensure the reproducibility and extendability of our work as well as improve efficiency when training. Hence, the results here have been improved a little bit when compared to their counterparts in the paper.


### Result of our proposed method on FGVC-Aircraft dataset:

| Model                |  Accuracy | Pretrained model |
|-------|:------:|:---------:|
| ResNet-50            | 68.92     |[download](https://drive.google.com/file/d/1-4keaQBactTvOFCeoDJyRUdfXmsBk6VN/view?usp=sharing) |
| Inception-V3         | 77.92     |[download](https://drive.google.com/file/d/11nOzMyxTGqpd0UI5urrOxP-KOraeBKbA/view?usp=sharing) |
| InceptionResNet-V2   | 79.54     |[download](https://drive.google.com/file/d/1RwWDYB2QpmokAv8w9vngOcHrTVMCo_kX/view?usp=sharing) |
| MoPT                 | 81.58     |[download](https://drive.google.com/file/d/1-3lbojBiohMjYWPdnkutl8mnWtYvi-0d/view?usp=sharing) |
## Acknowledgement
The implementation of dynamic routing algorithm and capsule layers were based on the Tensorflow build of CapsNet by its authors in this [link](https://github.com/Sarasra/models/tree/master/research/capsules)
