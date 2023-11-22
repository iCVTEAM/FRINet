# FRINet






> Frequency Representation Integration Network for Camouflaged Object Detection, MM2023,            
> *ACM MM 2023 ([MM'23](https://dl.acm.org/doi/abs/10.1145/3581783.3611773))*

## Abstract
Recent camouflaged object detection (COD) approaches have been proposed to accurately segment objects blended into surroundings. The most challenging and critical issue in COD is to find out the lines of demarcation between objects and background in the camouflage environment. Because of the similarity between the target object and the background, these lines are difficult to be found accurately. However, these are easy to be observed in different frequency components of the image. To this end, in this paper we rethink COD from the perspective of frequency components and propose a Frequency Representation Integration Network to mine informative cues from them. Specifically, we obtain high-frequency components from the original image by Laplacian pyramid-like decomposition, and then respectively send the image to a transformer-based encoder and frequency components to a tailored CNN-based Residual Frequency Array Encoder. Besides, we utilize the multi-head self-attention in transformer encoder to capture low-frequency signals, which can effectively parse the overall contextual information of camouflage scenes. We also design a Frequency Representation Reasoning Module, which progressively eliminates discrepancies between differentiated frequency representations and integrates them by modeling their point-wise relations. Moreover, to further bridge different frequency representations, we introduce the image reconstruction task to implicitly guide their integration. Sufficient experiments on three widely-used COD benchmark datasets demonstrate that our method surpasses existing state-of-the-art methods by a large margin.


## Usage
### Requirements
* Python 3.8
* Pytorch 1.7.1
* OpenCV
* Numpy
* Apex
* Timm

### Directory
The directory should be like this:

````
-- src 
-- model (saved model)
-- pre (pretrained model)
-- result (maps)
-- data (train dataset and test dataset)
   |-- TrainDataset
   |   |-- image
   |   |-- mask
   |-- TestDataset
   |   |--NC4K
   |   |   |--image
   |   |   |--mask
   |   |--CAMO
   |   |--COD10K
   ...
   
````

### Train
```
cd src
./train.sh
```
* We implement our method by PyTorch and conduct experiments on 1 NVIDIA 3090 GPU.
* We adopt pre-trained [DeiT](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth) and [PvT](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth) as backbone networks, which are saved in PRE folder.
* We train our method on 2 backbone settings : ViT and Pvt.
* After training, the trained models will be saved in MODEL folder.

### Test

```
cd src
python test.py
```
* After testing, maps will be saved in RESULT folder




## Results

ViT and PvT_v2: [Google Drive](https://drive.google.com/file/d/1JvfVeKE_VGsF0XgldKOswdCNWhMBxWnO/view?usp=drive_link)


## Citation
```
@inproceedings{10.1145/3581783.3611773,
author = {Xie, Chenxi and Xia, Changqun and Yu, Tianshu and Li, Jia},
title = {Frequency Representation Integration for Camouflaged Object Detection},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3581783.3611773},
doi = {10.1145/3581783.3611773},
pages = {1789–1797},
numpages = {9},
series = {MM '23}
}
```
