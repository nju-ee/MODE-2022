# MODE: Multi-view Omnidirectional Depth Estimation with 360$^\circ$ Cameras
This repository contains the source code for our paper:

MODE: Multi-view Omnidirectional Depth Estimation with 360$^\circ$ Cameras. ECCV 2022

Coming soon!
## Dataset Deep360
Deep360 is a large synthetic outdoor dataset for multi-view omnidirectional depth estimation. It contains 2100 frames for training, 300 frames for validation and 600 frames for testing. Panoramas, ground truth disparity and depth maps are presented to train and evaluate omnidirectional depth estimation algorithms. This dataset also contains "soiled version" of panoramas which are soiled or affected by three common outdoor factors: mud spots, water drops and glare.

You can download Deep360 through this [link]() as a single zip file.

The MD5 check codes of this zip file is: 
```

```
The file structure of Deep360 is as follow:
```
Deep360
├── README.txt
├── ep1_500frames
    │ ├── training
    │   ├── rgb
    │   ├── rgb_soiled
    │   ├── disp
    │   ├── depth
    │ ├── validation
    │ ├── testing
├── ep2_500frames
├── ep3_500frames
├── ep4_500frames
├── ep5_500frames
├── ep6_500frames
```
Please download and unzip the file, and follow the README.txt in the dataset folder and the dataloader in this repository to use this dataset.
## Introduction
MODE is a two-stage omnidirectional depth estimation framework with multi-view 360◦ cameras. The framework first estimates the depth maps from different camera pairs via omnidirectional stereo matching and then fuses the depth maps to achieve robustness against mud spots, water drops on camera lenses, and glare caused by intense light.

![The pipeline of the proposed two-stage MODE](./net_arch.png)
## Requirements
+ gcc/g++ <=7.5.0 (to compile the sphere convolution operator)
+ PyTorch >=1.5.0
+ tensorboardX
+ cv2
+ numpy
+ PIL
+ numba
+ prettytable (to show the error metrics)
+ tqdm (to visualize the progress bar)
## Usage
* **First you need to compile the Spherical Convolution operator with following bash command:**
```
cd models/basic/spherical_conv && bash ./build.sh && cd ../../../
```
* **Training the disparity estimation stage(stereo matching)**

you can train the model with the same protocol in this paper (w.r.t load the pretrained stack hourglass part of PSMNet) using this command:
```
python train_disparity.py --dataset_root [path to Deep360 folder] --checkpoint_disp [path to pretrained PSMNet ckpt] --loadSHGonly --parallel
```
or train it from the random initialization:
```
python train_disparity --dataset_root [path to Deep360 folder] --parallel
```

* **Testing the disparity estimation stage(stereo matching)**

please run 
```
python test_disparity.py --dataset_root [path to Deep360 folder] --checkpoint_disp [path to trained ckpt of disparity stage] --parallel --save_output_path [path to save output]
```
for testing.

* **Saving outputs of stereo matching stage**

We suggest storge the disparity maps and confidence maps of stereo matching stage before the fusion stage to save the training time. 
please run 
```
python save_output_disparity_stage.py --checkpoint_disp [path to trained ckpt of disparity stage] --datapath [path to Deep360 folder] --outpath [path to save predicted disparity and confidence maps]
```
* **Training the fusion stage**

please run following command to train the fusion model:
```
python train_fusion.py --datapath-dataset [path to Deep360 folder] --datapath-input [path to saved outputs of disparity stage]
```
* **Testing the fusion stage**

please run following command to test the fusion model:
```
python test_fusion.py --datapath-dataset [path to Deep360 folder] --datapath-input [path to saved outputs of disparity stage] --outpath [path to save the fusion results]
```


*For all the command above, add ```--soiled``` for the soiled version of Deep360.*
## Pretrained Models
Our pre-trained models can be found:

[ModeDisparity](https://drive.google.com/file/d/123HOYyc6d9KRKV2sFVOhN0c0iEOofaz_/view?usp=sharing)

[ModeFusion](https://drive.google.com/file/d/1d-X0ygrhvYT3Wgwt3dbN2oD9oRkUcjVL/view?usp=sharing)

[ModeFusion_soiled](https://drive.google.com/file/d/10JCwp_RaW1113lb4ZfPBaovxBuoipSbF/view?usp=sharing)

## Acknowledgements
The sperical convolution in this paper refers to Coors et al.[1].

The code of stack hourglass in stereo matching network is adapted from [PSMNet](https://github.com/JiaRenChang/PSMNet) [2].

## Citation
If you use the Deep360 dataset, or find this project and paper helpful in your research, welcome to cite the paper.
(The citaion could be slightly different when this paper is published formally. We will update it in time.)
```
@inproceedings{Li_Jin2022MODE,
  author       = "Li, Ming and Jin, Xueqian and Hu, Xuejiao and Dai, Jingzhao and Du, Sidan and Li, Yang",
  title        = "MODE: Multi-view Omnidirectional Depth Estimation with 360$^\circ$ Cameras",
  booktitle    = "European Conference on Computer Vision (ECCV)",
  month        = "October",
  year         = "2022"
}
```

## References
[1] Coors, B., Condurache, A.P., Geiger, A.: Spherenet: Learning spherical representations for detection and classification in omnidirectional images. In: Ferrari, V.,Hebert, M., Sminchisescu, C., Weiss, Y. (eds.) Computer Vision – ECCV 2018. pp.525–541. Springer International Publishing, Cham(2018)

[2] Chang, J., Chen, Y.: Pyramid stereo matching network. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5410–5418 (2018). https://doi.org/10.1109/CVPR.2018.0