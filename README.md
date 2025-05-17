## P2AT: Pyramid Pooling Axial Transformer for Real-time Semantic Segmentation  [[Arxiv]](https://arxiv.org/abs/2310.15025) | ESWA [ESWA](https://www.sciencedirect.com/science/article/abs/pii/S0957417424014775)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/p2at-pyramid-pooling-axial-transformer-for/real-time-semantic-segmentation-on-camvid)](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-camvid?p=p2at-pyramid-pooling-axial-transformer-for)

You need to download the Cityscapesdatasets. and rename the folder cityscapes, then put the data under data folder.


# Clone this repository

```
git clone https://github.com/mohamedac29/P2AT
cd P2AT
```

### Datasets Preparation

## 1. Cavmvid Dataset
You can download the Camvid dataset from [Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid)

## Citysscapes Dataset
* You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) datasets, unzip them and put the files in the `data` folder with following structure.

```
$SEG_ROOT/data\ 
├── Camvid
│       ├── images
│       ├── labels

│    ├── cityscapes
│        ├── gtFine
│            ├── test
│            ├── train
│            ├── val
│     ── ── leftImg8bit
│             ├── test
│             ├── train
│             └── val
│    ├── list
         ├── Camvid
│          ├── test.lst
│          ├── train.lst
│          ├── trainval.lst
│          └── val.lst
│       ├── cityscapes
│          ├── test.lst
│          ├── train.lst
│          ├── trainval.lst
│          └── val.lst
   
```

### Training

##  Training on Camvid datsaset

* For instance, train the P2AT-S on Camvid dataset with batch size of 8 on 2 GPUs:
````bash
python tools/train.py --cfg configs/camvid/p2at_small_camvid.yaml GPUS (0,1) TRAIN.BATCH_SIZE_PER_GPU 4
````


### Citation

If you find this work useful in your research, please consider citing.

```
@article{elhassan2024p2at,
  title={P2AT: Pyramid pooling axial transformer for real-time semantic segmentation},
  author={Elhassan, Mohammed AM and Zhou, Changjun and Benabid, Amina and Adam, Abuzar BM},
  journal={Expert Systems with Applications},
  volume={255},
  pages={124610},
  year={2024},
  publisher={Elsevier}
}
```

```
@article{elhassan2024csnet,
  title={CSNet: Cross-Stage Subtraction Network for Real-Time Semantic Segmentation in Autonomous Driving},
  author={Elhassan, Mohammed AM and Zhou, Changjun and Zhu, Donglin and Adam, Abuzar BM and Benabid, Amina and Khan, Ali and Mehmood, Atif and Zhang, Jun and Jin, Hu and Jeon, Sang-Woon},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024},
  publisher={IEEE}
}
```