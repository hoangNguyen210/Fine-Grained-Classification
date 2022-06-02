* [Introduction](#intro)
* [Usage](#usage)
* [Conclusion](#conclusion)
* [Contact](#contact)

# MoPT: Fine Grained Classification using Mixture of Pretrained model


## Introduction 

Mixture of Experts is an ensemble method combining weak learners to form a strong learner, hence improving the performance of our models. The crucial component of Mixture of experts is Gating Network, this element will select which experts to use depending on their influences. In this work, we will use Mixture of Experts model to combine each pretrained and form a comprehensive classification system. 


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
**About the code** Our work has been utilized some pretrained model on keras framework as well as for data augmentation technique. To access the code, click to Code column in the table below. 

### Result of our proposed method on FGVC-Aircraft dataset:

| Model                |  Accuracy | Code     |Pretrained model |
|-------|:------:|:---------:|:---------:|
| ResNet-50            | 68.92 |    [ResNet50.ipynb](https://github.com/hoangNguyen210/Fine-Grained-Classification/blob/main/code/ResNet50.ipynb)          |[download](https://drive.google.com/file/d/1-4keaQBactTvOFCeoDJyRUdfXmsBk6VN/view?usp=sharing) |
| Inception-V3         | 77.92 |    [Inception_V3.ipynb](https://github.com/hoangNguyen210/Fine-Grained-Classification/blob/main/code/Inception_V3.ipynb)         |[download](https://drive.google.com/file/d/11nOzMyxTGqpd0UI5urrOxP-KOraeBKbA/view?usp=sharing) |
| InceptionResNet-V2   | 79.54 |    [InceptionResNet_V2.ipynb](https://github.com/hoangNguyen210/Fine-Grained-Classification/blob/main/code/InceptionResNet_V2.ipynb)         |[download](https://drive.google.com/file/d/1RwWDYB2QpmokAv8w9vngOcHrTVMCo_kX/view?usp=sharing) |
| MoPT                 | 81.58 |     [MoPT.ipynb](https://github.com/hoangNguyen210/Fine-Grained-Classification/blob/main/code/MoPT.ipynb)           |[download](https://drive.google.com/file/d/1-3lbojBiohMjYWPdnkutl8mnWtYvi-0d/view?usp=sharing) |


### Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- hoangng210a@gmail.com


