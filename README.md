# jsws
Code for the paper is coming soon

## Environment
clone repo:
```
git clone https://github.com/zengxianyu/jsws.git
git submodule init 
git submodule update
```

prepare environment:
```
conda env create --file=pytorch_environments.yml
```

## Prepare Data
Required training data: 
* PASCAL VOC 2012 segmentation dataset (including 10,582 extra training samples introduced by Hariharan et al [15]). We only use image-level class labels of them. Put the folder ```VOC2012``` in ```data/datasets/segmentation_Dataset/VOCdevkit/```
* DUTS saliency dataset training split. Put the folder ```DUT-train``` in ```data/datasets/saliency_Dataset/```
* (Optional) ECSSD dataset for test and validation on saliency task.  Put the folder ```ECSSD``` in ```data/datasets/saliency_Dataset/```

You can find them on their official sites or use ```download.sh``` to dowload from [an unofficial download place](http://ok.i68h.cn:8000/) that I provide. 

## Train stage 1
train using image-level class labels and saliency ground-truth:

```shell
weak_seg_full_sal_train.py
```

Open http://host ip:8000/savefiles/jlsfcn_dense169.html in a browser for visualizing training process. 

It should be easy to achieve MIOU>54 but you may need to try multiple times to get the score MIOU 57.1 or more than that in Table. 5 of the paper. 

## Train stage 2
train a more complex model using the prediction of the model trained in the stage 1. 

1. make training data

```
weak_seg_full_sal_syn.py
```

2. train

```
self_seg_full_sal_train.py
```

## Test
stage 1 model:
```
weak_seg_full_sal_test.py
```

stage 2 model
```
self_seg_full_sal_test.py
```
By default it calls the function ```test(...)``` to test on segmentation task

Change to call the function ```test_sal(...)``` to test on saliency task

## Saliency results

[download saliency maps](http://ok.i68h.cn:8000/JLWS-sal.zip) on datasets ECSSD, PSACALS, HKU-IS, DUT-OMRON, DUTS-test, SOD; [Google Drive](https://drive.google.com/open?id=1KqO8bhJn2StXGblBL_9V6-yM2CSOBNsz); [One Drive](https://1drv.ms/u/s!AqVkBGUQ01XGjxiqc5pdH20yPXz4?e=WzCpBW)

## Citation
```
@inproceedings{zeng2019joint,
  title={Joint learning of saliency detection and weakly supervised semantic segmentation},
  author={Zeng, Yu and Zhuge, Yunzhi and Lu, Huchuan and Zhang, Lihe},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```
