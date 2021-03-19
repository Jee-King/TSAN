# TSAN: A Two-Stage Attentive Network for Single Image Super-Resolution

pytorch code for papar 'A Two-Stage Attentive Network for Single Image Super-Resolution'.

The code is built on EDSR (PyTorch) and tested on Ubuntu 16.04 environment (Python3.6, PyTorch > 0.4.0) with Titan X/1080Ti/Xp GPUs.

## **Contents**
1. Introduction
2. Train
3. Test
4. Results
5. To-do-list

## Introduction
Recently, deep convolutional neural networks (CNNs) have been widely explored in single image superresolution (SISR) and contribute remarkable progress. However, most of the existing CNNs-based SISR methods do not adequately explore contextual information in the feature extraction stage and pay little attention to the final high-resolution (HR) image reconstruction step, hence hindering the desired SR performance. To address the above two issues, in this paper, we propose a two-stage attentive network (TSAN) for accurate SISR in a coarse-to-fine manner. Specifically, a novel dilated residual block (DRB) is developed as a fundamental unit to extract contextual features efficiently. Based on DRB, we further design a multicontext attentive block (MCAB) to make the network focus on more informative contextual features. Moreover, we present an essential refined attention block (RAB) which could explore useful cues in HR space for reconstructing fine-detailed HR image. Extensive evaluations on four benchmark datasets demonstrate the efficacy of our proposed TSAN in terms of quantitative metrics and visual effects.


## Train
