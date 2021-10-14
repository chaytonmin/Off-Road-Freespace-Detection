# ORFD: A Dataset and Benchmark for Off-Road Freespace Detection


## Introduction
Freespace detection is very important for carrying out trajectory planning and decision-making in autonomous navigation. A great deal of research has demonstrated that freespace detection can be solved with deep learning methods. However, these efforts were focused in the urban road environments and few deep learning-based methods were specifically designed for off-road freespace detection due to the lack of off-road benchmarks. In this paper, we present the ORFD dataset, which to our knowledge is the first off-road freespace detection dataset. The dataset was collected in different scenarios (woodland, farmland, grassland, and countryside), different weather conditions (sunny, rainy, foggy, and snowy) and different light conditions (day, evening and night), which totally contains 12,198 lidar and RGB image pairs with the traversable area, non-traversable area and unreachable area annotated in detail. We propose a novel network named OFF-Net, which unifies Transformer to aggregate local and global information, to meet the requirement of large receptive fields for freespace detection task. We also propose the cross-attention for dynamically fusing LiDAR and RGB image information for accurate off-road freespace detection. 

<p align="center">
<img src="doc/demo1.gif" width="100%"/>demo 1
</p>
<p align="center">
<img src="doc/demo2.gif" width="100%"/>demo 2
</p>
<p align="center">
<img src="doc/demo3.gif" width="100%"/>demo 3
</p>


## Requirements

- python 3.6

- pytorch 1.4+

- other requirements: `pip install -r requirements.txt`

## Pretrained models

The pretrained models of our OFF-Net trained on ORFD dataset can be download [here](https://drive.google.com/drive/folders/1lnm2M1HEkVs9W3-FSEX3ddE9GYz4rqCU). 

## Prepare data

The proposed off-road freespace detection dataset ORFD can be found [here](https://pan.baidu.com/s/13lQD8qBmsdSfpOgW18fA-g) (code: kbpl). Extract and organize as follows:

```
|-- datasets
 |  |-- ORFD
 |  |  |-- training
 |  |  |  |-- sequence   |-- calib
 |  |  |  |-- sequence   |-- sparse_depth
 |  |  |  |-- sequence   |-- dense_depth
 |  |  |  |-- sequence   |-- lidar_data
 |  |  |  |-- sequence   |-- image_data
 |  |  |  |-- sequence   |-- gt_image
 ......
 |  |  |-- validation
 ......
 |  |  |-- testing
 ......
```

## Usage

### Demo ###
```
bash ./scripts/demo.sh
```
### Training
```
bash ./scripts/train.sh
```
### Testing

```
bash ./scripts/test.sh
```

## Acknowledgement

This repository is heavily based on [SNE-RoadSeg](https://github.com/hlwang1124/SNE-RoadSeg) [1], and  [SegFormer](https://github.com/NVlabs/SegFormer) [2].

## References

[1] Fan, Rui, et al. "Sne-roadseg: Incorporating surface normal information into semantic segmentation for accurate freespace detection." *European Conference on Computer Vision*. Springer, Cham, 2020. 

[2] Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." *arXiv preprint arXiv:2105.15203* (2021).
