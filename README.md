# TSAN: A Two-Stage Attentive Network for Single Image Super-Resolution

**Someone tell me after installing prerequisites, it does not work，and the code is different to read. So I will check the environment again and improve the code of the model and configs for an easy read. The modified codes will be made public again.**

pytorch code for papar 'A Two-Stage Attentive Network for Single Image Super-Resolution'.

The code is built on [EDSR (PyTorch)](https://github.com/sanghyun-son/EDSR-PyTorch) and tested on Ubuntu 16.04 environment with Titan 1080Ti/Xp GPUs.

Paper can be download from [TSAN](https://github.com/Jee-King/TSAN/raw/main/SISR_TCSVT.pdf)

## **Contents**
1. Introduction
2. Prerequisites
3. Train
4. Test
5. Performance
6. Citing
7. To-do-list

## Introduction
Recently, deep convolutional neural networks (CNNs) have been widely explored in single image superresolution (SISR) and contribute remarkable progress. However, most of the existing CNNs-based SISR methods do not adequately explore contextual information in the feature extraction stage and pay little attention to the final high-resolution (HR) image reconstruction step, hence hindering the desired SR performance. To address the above two issues, in this paper, we propose a two-stage attentive network (TSAN) for accurate SISR in a coarse-to-fine manner. Specifically, a novel dilated residual block (DRB) is developed as a fundamental unit to extract contextual features efficiently. Based on DRB, we further design a multicontext attentive block (MCAB) to make the network focus on more informative contextual features. Moreover, we present an essential refined attention block (RAB) which could explore useful cues in HR space for reconstructing fine-detailed HR image. Extensive evaluations on four benchmark datasets demonstrate the efficacy of our proposed TSAN in terms of quantitative metrics and visual effects.
![pipeline](https://github.com/Jee-King/TSAN/blob/main/visual_results/pipeline.png)

## Prerequisites
1. Python 3.6
2. PyTorch >= 0.4.0
3. numpy
4. skimage
5. imageio
6. matplotlib
7. tqdm

## Train
### prepare training data
We used DIV2K dataset (1-800) to train our model. Please download it from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).
Extract the file and put it into the _Train/dataset_.
### Training 
Using --ext sep_reset argument on your first running.

You can skip the decoding part and use saved binaries with --ext sep argument in second time.

If you have enough memory, using --ext bin.

```
cd Train/
# TSAN x2  LR: 48 * 48  HR: 96 * 96
python main.py --template TSAN --save TSAN_X2 --scale 2 --reset --save_results --patch_size 96 --ext sep_reset

# TSAN x3  LR: 48 * 48  HR: 144 * 144
python main.py --template TSAN --save TSAN_X3 --scale 3 --reset --save_results --patch_size 144 --ext sep_reset

# TSAN x4  LR: 48 * 48  HR: 192 * 192
python main.py --template TSAN --save TSAN_X4 --scale 4 --reset --save_results --patch_size 192 --ext sep_reset
```

### Test
Test dataset can be download from [here](https://drive.google.com/drive/folders/1xyiuTr6ga6ni-yfTP7kyPHRmfBakWovo).

Using pre-trained model for training, all test datasets must be pretreatment by _Test/Prepare_TestData_HR_LR.m_ and all pre-trained model should be put into _Test/model/_.

```
#TSAN x2
python main.py --data_test MyImage --scale 2 --model TSAN --pre_train ../model/TSAN_x2.pt --test_only --save_results --chop --save "TSAN" --testpath ../LR/LRBI --testset Set5

#TSAN x3
python main.py --data_test MyImage --scale 3 --model TSAN --pre_train ../model/TSAN_x3.pt --test_only --save_results --chop --save "TSAN" --testpath ../LR/LRBI --testset Set5

#TSAN x4
python main.py --data_test MyImage --scale 4 --model TSAN --pre_train ../model/TSAN_x4.pt --test_only --save_results --chop --save "TSAN" --testpath ../LR/LRBI --testset Set5

```
You can  introduce self-ensemble strategy to improve the performance by addding _--self_ensemble_.

More running instructions can be found in _demo.sh_.

### Performance
![x2](https://github.com/Jee-King/TSAN/blob/main/visual_results/x2.png)
![x3](https://github.com/Jee-King/TSAN/blob/main/visual_results/x3.png)
![x4](https://github.com/Jee-King/TSAN/blob/main/visual_results/x4.png)

This implementation is for non-commercial research use only.

### Citing
If you do publish a paper where this Work helped your research, Please cite the following papers in your publications.

```
@article{zhang2021tsan,
 title={A Two-Stage Attentive Network for Single Image Super-Resolution},
 author={Zhang, Jiqing and Long, Chengjiang and Wang, Yuxin and Piao, Haiyin and Mei, Haiyang and Yang, Xin and Yin, Baocai},
 journal={IEEE Transactions on Circuits and Systems for Video Technology},
 year={2021},
 publisher={IEEE}
}
```
