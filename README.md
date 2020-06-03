# jsws
Code for the paper is coming soon

## Environment
clone repo:
```
git clone https://github.com/zengxianyu/jsws-old.git
git submodule init 
git submodule update
```

prepare environment:
```
conda env create --file=pytorch_environments.yml
```

## Prepare Data
./trainaug/txt
[DUTS-train (Onedrive)](https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/EaJni8OcXzxJi1BDQsjqh4YBFlY_UlMNHvF6TGm43dIDWg?e=AhNHVk)

[VOC2012 (Onedrive)](https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/EVUJBg67ICxHqB_wfehc34gBQKi_RTJgnTCcUPnwxfTSIA?e=ef0AJw)

[SegmentationClassAug (Onedrive)](https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/EXhmcGsGEaBPnhOffoNlh2UBUyZuB7Eck5WUbJ3f3pSSbA?e=vLLc34)

or

./SegmentationClassAug



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

2. train (optional: processing by [densecrf](https://github.com/lucasb-eyer/pydensecrf))

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

[download saliency maps](http://ok.biu886.com:8000/JLWS-sal.zip) on datasets ECSSD, PSACALS, HKU-IS, DUT-OMRON, DUTS-test, SOD; [Google Drive](https://drive.google.com/open?id=1KqO8bhJn2StXGblBL_9V6-yM2CSOBNsz); [One Drive](https://1drv.ms/u/s!AqVkBGUQ01XGjxiqc5pdH20yPXz4?e=WzCpBW)

## Citation
```
@inproceedings{zeng2019joint,
  title={Joint learning of saliency detection and weakly supervised semantic segmentation},
  author={Zeng, Yu and Zhuge, Yunzhi and Lu, Huchuan and Zhang, Lihe},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```
