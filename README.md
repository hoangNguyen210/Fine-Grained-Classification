* [Introduction](#intro)
* [Usage](#usage)
* [Model](#model)

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

We expect the images in the following path:
```
fgvc-aircraft-2013b/
  fgvc-aircraft-2013b/
    data/ 
      images/ 
        0034309.jpg
        0034958.jpg
```
We also use the given `train.csv`, `test.csv`, `val.csv`  to extract index and label for the train, test, validation dataset.


## Proposed method
**About the code** Our work has been utilized some pretrained model on keras framework as well as for data augmentation technique. 


### Result of our proposed method on FGVC-Aircraft dataset:

| Model                |  Accuracy | Pretrained model |
|-------|:------:|:---------:|
| ResNet-50            | 68.92     |[download](https://drive.google.com/file/d/1-4keaQBactTvOFCeoDJyRUdfXmsBk6VN/view?usp=sharing) |
| Inception-V3         | 77.92     |[download](https://drive.google.com/file/d/11nOzMyxTGqpd0UI5urrOxP-KOraeBKbA/view?usp=sharing) |
| InceptionResNet-V2   | 79.54     |[download](https://drive.google.com/file/d/1RwWDYB2QpmokAv8w9vngOcHrTVMCo_kX/view?usp=sharing) |
| MoPT                 | 81.58     |[download](https://drive.google.com/file/d/1-3lbojBiohMjYWPdnkutl8mnWtYvi-0d/view?usp=sharing) |


### Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- hoangng210a@gmail.com


