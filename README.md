<p align="right">English | <a href="./README_CN.md">简体中文</a></p>  


<p align="center">
  <img src="docs/figs/logo.png" align="center" width="25%">
  
  <h3 align="center"><strong>Calib3D: Calibrating Model Preferences for Reliable 3D Scene Understanding</strong></h3>

  <p align="center">
      <a href="https://ldkong.com/" target='_blank'>Lingdong Kong</a><sup>1,2,*</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://xiangxu-0103.github.io/" target='_blank'>Xiang Xu</a><sup>3,*</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://cen-jun.com/" target='_blank'>Jun Cen</a><sup>4</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com/citations?user=QDXADSEAAAAJ" target='_blank'>Wenwei Zhang</a><sup>1</sup>
      <br>
      <a href="https://scholar.google.com/citations?user=lSDISOcAAAAJ" target='_blank'>Liang Pan</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com/citations?user=eGD0b7IAAAAJ" target='_blank'>Kai Chen</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://liuziwei7.github.io/" target='_blank'>Ziwei Liu</a><sup>5</sup>
  <br />
  <sup>1</sup>Shanghai AI Laboratory&nbsp;&nbsp;&nbsp;
  <sup>2</sup>National University of Singapore&nbsp;&nbsp;&nbsp;
  <sup>3</sup>Nanjing University of Aeronautics and Astronautics&nbsp;&nbsp;&nbsp;
  <sup>4</sup>The Hong Kong University of Science and Technology&nbsp;&nbsp;&nbsp;
  <sup>5</sup>S-Lab, Nanyang Technological University
  </p>

</p>

<p align="center">
  <a href="https://arxiv.org/abs/2403.17010" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-slategray">
  </a>
  
  <a href="https://ldkong.com/Calib3D" target='_blank'>
    <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-lightblue">
  </a>
  
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-%F0%9F%8E%AC-pink">
  </a>
  
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/%E4%B8%AD%E8%AF%91%E7%89%88-%F0%9F%90%BC-red">
  </a>
  
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=ldkong1205.Calib3D&left_color=gray&right_color=firebrick">
  </a>
</p>


## About

`Calib3D` is a comprehensive benchmark targeted at probing the uncertainties of 3D scene understanding models in real-world conditions. It encompasses a systematic study of state-of-the-art models across diverse 3D datasets, laying a solid foundation for the future development of reliable 3D scene understanding systems.

- :construction: **Aleatoric Uncertainty in 3D:** We examine how intrinsic factors, such as sensor measurement noises and point cloud density variations, contribute to data uncertainty in 3D scene understanding. Such uncertainty cannot be reduced even with more data or improved models, necessitating efforts that effectively interpret and quantify this inherent variability.
- :bus: **Epistemic Uncertainty in 3D:** Different from the rather unified network structures in 2D, 3D scene understanding models shed a wider array of structures due to the complex nature of 3D data processing. Our investigation extends to the model uncertainty associated with the diverse 3D architectures, highlighting the importance of addressing knowledge gaps in model training and data representation.


### Motivation

| <img src="docs/figs/teaser.png" align="center" width="98%"> |
| :-: |
| Well-calibrated 3D scene understanding models are anticipated to deliver ***low uncertainties*** when predictions are ***accurate*** and ***high uncertainties*** when predictions are ***inaccurate***. Existing 3D models (UnCal) struggled to provide proper uncertainty estimates. The plots shown are the point-wise expected calibration error (ECE) rates. The colormap goes from *dark* to *light* denoting *low* and *high* error rates, respectively. |

Visit our [project page](https://ldkong.com/Calib3D) to explore more examples. :blue_car:



## Updates

- \[2024.03\] - Our [paper](https://arxiv.org/abs/2403.17010) is available on arXiv. The code has been made publicly accessible. :rocket:



## Outline

- [Installation](#gear-installation)
- [Data Preparation](#hotsprings-data-preparation)
- [Getting Started](#rocket-getting-started)
- [Model Zoo](#dragon-model-zoo)
- [Calib3D Benchmark](#triangular_ruler-calib3d-benchmark)
- [TODO List](#memo-todo-list)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)



## :gear: Installation

For details related to installation and environment setups, kindly refer to [INSTALL.md](docs/INSTALL.md).



## :hotsprings: Data Preparation

| [**nuScenes**](https://www.nuscenes.org/nuscenes) | [**SemanticKITTI**](http://semantic-kitti.org/) | [**Waymo Open**](https://waymo.com/open) | [**SemanticSTF**](https://github.com/xiaoaoran/SemanticSTF) | [**SemanticPOSS**](http://www.poss.pku.edu.cn/semanticposs.html) |
| :-: | :-: | :-: | :-: | :-: |
| <img width="125" src="docs/figs/nuscenes.png"> | <img width="125" src="docs/figs/semantickitti.png"> | <img width="125" src="docs/figs/waymo-open.png"> | <img width="125" src="docs/figs/semanticposs.png"> | <img width="125" src="docs/figs/semanticstf.png"> |
| [**ScribbleKITTI**](https://github.com/ouenal/scribblekitti) | [**Synth4D**](https://github.com/saltoricristiano/gipso-sfouda) | [**S3DIS**](http://buildingparser.stanford.edu/dataset.html) | [**nuScenes-C**](https://github.com/ldkong1205/Robo3D) | [**SemanticKITTI-C**](https://github.com/ldkong1205/Robo3D) |
| <img width="125" src="docs/figs/scribblekitti.png"> | <img width="125" src="docs/figs/synth4d.png"> | <img width="125" src="docs/figs/s3dis.png"> | <img width="125" src="docs/figs/nuscenes-c.png"> | <img width="125" src="docs/figs/semantickitti-c.png"> |

Kindly refer to [DATA_PREPARE.md](docs/document/DATA_PREPARE.md) for the details to prepare these datasets.



## :rocket: Getting Started

To learn more usage about this codebase, kindly refer to [GET_STARTED.md](docs/GET_STARTED.md).



## :dragon: Model Zoo

<details open>
<summary>&nbsp<b>Range View</b></summary>

> - [x] **[RangeNet<sup>++</sup>](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf), IROS 2019.** <sup>[**`[Code]`**](https://github.com/PRBonn/lidar-bonnetal)</sup>
> - [x] **[SalsaNext](https://arxiv.org/abs/2003.03653), ISVC 2020.** <sup>[**`[Code]`**](https://github.com/TiagoCortinhal/SalsaNext)</sup>
> - [x] **[FIDNet](https://arxiv.org/abs/2109.03787), IROS 2021.** <sup>[**`[Code]`**](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI)</sup>
> - [x] **[CENet](https://arxiv.org/abs/2207.12691), ICME 2022.** <sup>[**`[Code]`**](https://github.com/huixiancheng/CENet)</sup>
> - [x] **[RangeViT](https://arxiv.org/abs/2301.10222), CVPR 2023.** <sup>[**`[Code]`**](https://github.com/valeoai/rangevit)</sup>
> - [x] **[RangeFormer](https://arxiv.org/abs/2303.05367), ICCV 2023.** 
> - [x] **[FRNet](https://arxiv.org/abs/2312.04484), arXiv 2023.** <sup>[**`[Code]`**](https://github.com/Xiangxu-0103/FRNet)</sup>

</details>

<details open>
<summary>&nbsp<b>Bird's Eye View</b></summary>

> - [x] **[PolarNet](https://arxiv.org/abs/2003.14032), CVPR 2020.** <sup>[**`[Code]`**](https://github.com/edwardzhou130/PolarSeg)</sup>

</details>

<details open>
<summary>&nbsp<b>Sparse Voxel</b></summary>

> - [x] **[MinkUNet<sub>18</sub>](https://arxiv.org/abs/1904.08755), CVPR 2019.** <sup>[**`[Code]`**](https://github.com/NVIDIA/MinkowskiEngine)</sup>
> - [x] **[MinkUNet<sub>34</sub>](https://arxiv.org/abs/1904.08755), CVPR 2019.** <sup>[**`[Code]`**](https://github.com/NVIDIA/MinkowskiEngine)</sup>
> - [x] **[Cylinder3D](https://arxiv.org/abs/2011.10033), CVPR 2021.** <sup>[**`[Code]`**](https://github.com/xinge008/Cylinder3D)</sup>
> - [x] **[SpUNet<sub>18</sub>](https://github.com/traveller59/spconv), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/traveller59/spconv)</sup>
> - [x] **[SpUNet<sub>34</sub>](https://github.com/traveller59/spconv), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/traveller59/spconv)</sup>

</details>

<details open>
<summary>&nbsp<b>Multi-View Fusion</b></summary>

> - [x] **[SPVCNN<sub>18</sub>](https://arxiv.org/abs/2007.16100), ECCV 2020.** <sup>[**`[Code]`**](https://github.com/mit-han-lab/spvnas)</sup>
> - [x] **[SPVCNN<sub>34</sub>](https://arxiv.org/abs/2007.16100), ECCV 2020.** <sup>[**`[Code]`**](https://github.com/mit-han-lab/spvnas)</sup>
> - [x] **[RPVNet](https://arxiv.org/abs/2103.12978), ICCV 2021.**
> - [x] **[2DPASS](https://arxiv.org/abs/2207.04397), ECCV 2022.** <sup>[**`[Code]`**](https://github.com/yanx27/2DPASS)</sup>
> - [x] **[CPGNet](https://arxiv.org/abs/2204.09914), ICRA 2022.** <sup>[**`[Code]`**](https://github.com/GangZhang842/CPGNet)</sup>
> - [x] **[GFNet](https://arxiv.org/abs/2207.02605), TMLR 2022.** <sup>[**`[Code]`**](https://github.com/haibo-qiu/GFNet)</sup>
> - [x] **[UniSeg](https://arxiv.org/abs/2309.05573), ICCV 2023.**

</details>

<details open>
<summary>&nbsp<b>Raw Point</b></summary>

> - [x] **[PointNet<sup>++</sup>](https://arxiv.org/abs/1706.02413), NeurIPS 2017.** <sup>[**`[Code]`**](https://github.com/erikwijmans/Pointnet2_PyTorch)</sup>
> - [x] **[DGCNN](https://arxiv.org/abs/1801.07829), TOG 2019.** <sup>[**`[Code]`**](https://github.com/WangYueFt/dgcnn)</sup>
> - [x] **[KPConv](https://arxiv.org/abs/1904.08889), ICCV 2019.** <sup>[**`[Code]`**](https://github.com/HuguesTHOMAS/KPConv)</sup>
> - [x] **[PAConv](https://arxiv.org/abs/2103.14635), CVPR 2021.** <sup>[**`[Code]`**](https://github.com/CVMI-Lab/PAConv)</sup>
> - [x] **[PIDS<sub>1.25x</sub>](https://arxiv.org/abs/2211.15759), WACV 2023.** <sup>[**`[Code]`**](https://github.com/lordzth666/WACV23_PIDS-Joint-Point-Interaction-Dimension-Search-for-3D-Point-Cloud)</sup>
> - [x] **[PIDS<sub>2.0x</sub>](https://arxiv.org/abs/2211.15759), WACV 2023.** <sup>[**`[Code]`**](https://github.com/lordzth666/WACV23_PIDS-Joint-Point-Interaction-Dimension-Search-for-3D-Point-Cloud)</sup>
> - [x] **[WaffleIron](http://arxiv.org/abs/2301.10100), ICCV 2023.** <sup>[**`[Code]`**](https://github.com/valeoai/WaffleIron)</sup>

</details>

<details open>
<summary>&nbsp<b>3D Data Augmentation</b></summary>

> - [x] **[PolarMix](https://arxiv.org/abs/2208.00223), NeurIPS 2022.** <sup>[**`[Code]`**](https://github.com/xiaoaoran/polarmix)</sup>
> - [x] **[LaserMix](https://arxiv.org/abs/2207.00026), CVPR 2023.** <sup>[**`[Code]`**](https://github.com/ldkong1205/LaserMix)</sup>
> - [x] **[FrustumMix](https://arxiv.org/abs/2312.04484), arXiv 2023.** <sup>[**`[Code]`**](https://github.com/Xiangxu-0103/FRNet)</sup>

</details>

<details open>
<summary>&nbsp<b>SparseConv Backend</b></summary>

> - [x] **[MinkowskiEngine](https://arxiv.org/abs/1904.08755), CVPR 2019.** <sup>[**`[Code]`**](https://github.com/NVIDIA/MinkowskiEngine)</sup>
> - [x] **[SpConv](https://github.com/traveller59/spconv), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/traveller59/spconv)</sup>
> - [x] **[TorchSparse](https://arxiv.org/abs/2204.10319), MLSys 2022.** <sup>[**`[Code]`**](https://github.com/mit-han-lab/torchsparse)</sup>
> - [ ] **[TorchSparse<sup>++</sup>](https://www.dropbox.com/scl/fi/obdku0kqxjlkvuom2opk4/paper.pdf?rlkey=0zmy8eq9fzllgkx54zsvwsecf&dl=0), MICRO 2023.** <sup>[**`[Code]`**](https://github.com/mit-han-lab/torchsparse)</sup>

</details>


## :triangular_ruler: Calib3D Benchmark

### Reliability Diagram

| <img src="docs/figs/diagram.png" align="center" width="99%"> |
| :-: |
| The reliability diagrams of visualized calibration gaps on the val set of SemanticKITTI. UnCal, TempS, MetaC, and DeptS denote the uncalibrated, temperature, meta, and our depth-aware scaling calibrations, respectively. |



### In-Domain 3D Uncertainty

<details open>
<summary>&nbsp<b>nuScenes & SemanticKITTI</b></summary>

<table>
  <tr>
    <th rowspan="2"><strong>Method</strong></th>
    <th rowspan="2"><strong>Modal</strong></th>
    <th colspan="6"><strong>nuScenes</strong></th>
    <th colspan="6"><strong>SemanticKITTI</strong></th>
  </tr>
  <tr>
    <td><strong>UnCal</strong></td>
    <td><strong>TempS</strong></td>
    <td><strong>LogiS</strong></td>
    <td><strong>DiriS</strong></td>
    <td><strong>MetaC</strong></td>
    <td><strong>DeptS</strong></td>
    <td><strong>UnCal</strong></td>
    <td><strong>TempS</strong></td>
    <td><strong>LogiS</strong></td>
    <td><strong>DiriS</strong></td>
    <td><strong>MetaC</strong></td>
    <td><strong>DeptS</strong></td>
  </tr>
  <tr>
    <td>RangeNet++</td>
    <td>Range</span></td>
    <td>4.57%</td>
    <td>2.74%</td>
    <td>2.79%</td>
    <td>2.73%</td>
    <td>2.78%</td>
    <td>2.61%</td>
    <td>4.01%</td>
    <td>3.12%</td>
    <td>3.16%</td>
    <td>3.59%</td>
    <td>2.38%</td>
    <td>2.33%</td>
  </tr>
  <tr>
    <td>SalsaNext</td>
    <td>Range</td>
    <td>3.27%</td>
    <td>2.59%</td>
    <td>2.58%</td>
    <td>2.57%</td>
    <td>2.52%</td>
    <td>2.42%</td>
    <td>5.37%</td>
    <td>4.29%</td>
    <td>4.31%</td>
    <td>4.11%</td>
    <td>3.35%</td>
    <td>3.19%</td>
</tr>
<tr>
    <td>FIDNet</td>
    <td>Range</td>
    <td>4.89%</td>
    <td>3.35%</td>
    <td>2.89%</td>
    <td>2.61%</td>
    <td>4.55%</td>
    <td>4.33%</td>
    <td>5.89%</td>
    <td>4.04%</td>
    <td>4.15%</td>
    <td>3.82%</td>
    <td>3.25%</td>
    <td>3.14%</td>
</tr>
<tr>
    <td>CENet</td>
    <td>Range</td>
    <td>4.44%</td>
    <td>2.47%</td>
    <td>2.53%</td>
    <td>2.58%</td>
    <td>2.70%</td>
    <td>2.44%</td>
    <td>5.95%</td>
    <td>3.93%</td>
    <td>3.79%</td>
    <td>4.28%</td>
    <td>3.31%</td>
    <td>3.09%</td>
</tr>
<tr>
    <td>RangeViT</td>
    <td>Range</td>
    <td>2.52%</td>
    <td>2.50%</td>
    <td>2.57%</td>
    <td>2.56%</td>
    <td>2.46%</td>
    <td>2.38%</td>
    <td>5.47%</td>
    <td>3.16%</td>
    <td>4.84%</td>
    <td>8.80%</td>
    <td>3.14%</td>
    <td>3.07%</td>
</tr>
<tr>
    <td>RangeFormer</td>
    <td>Range</td>
    <td>2.44%</td>
    <td>2.40%</td>
    <td>2.41%</td>
    <td>2.44%</td>
    <td>2.27%</td>
    <td>2.15%</td>
    <td>3.99%</td>
    <td>3.67%</td>
    <td>3.70%</td>
    <td>3.69%</td>
    <td>3.55%</td>
    <td>3.30%</td>
</tr>
<tr>
    <td>FRNet</td>
    <td>Range</td>
    <td>2.27%</td>
    <td>2.24%</td>
    <td>2.22%</td>
    <td>2.28%</td>
    <td>2.22%</td>
    <td>2.17%</td>
    <td>3.46%</td>
    <td>3.53%</td>
    <td>3.54%</td>
    <td>3.49%</td>
    <td>2.83%</td>
    <td>2.75%</td>
</tr>
<tr>
    <td>PolarNet</td>
    <td>BEV</td>
    <td>4.21%</td>
    <td>2.47%</td>
    <td>2.54%</td>
    <td>2.59%</td>
    <td>2.56%</td>
    <td>2.45%</td>
    <td>2.78%</td>
    <td>3.54%</td>
    <td>3.71%</td>
    <td>3.70%</td>
    <td>2.67%</td>
    <td>2.59%</td>
</tr>
<tr>
    <td>MinkUNet<sub>18</sub></td>
    <td>Voxel</td>
    <td>2.45%</td>
    <td>2.34%</td>
    <td>2.34%</td>
    <td>2.42%</td>
    <td>2.29%</td>
    <td>2.23%</td>
    <td>3.04%</td>
    <td>3.01%</td>
    <td>3.08%</td>
    <td>3.30%</td>
    <td>2.69%</td>
    <td>2.63%</td>
</tr>
<tr>
    <td>MinkUNet<sub>34</sub></td>
    <td>Voxel</td>
    <td>2.50%</td>
    <td>2.38%</td>
    <td>2.38%</td>
    <td>2.53%</td>
    <td>2.32%</td>
    <td>2.24%</td>
    <td>4.11%</td>
    <td>3.59%</td>
    <td>3.62%</td>
    <td>3.63%</td>
    <td>2.81%</td>
    <td>2.73%</td>
</tr>
<tr>
    <td>Cylinder3D</td>
    <td>Voxel</td>
    <td>3.19%</td>
    <td>2.58%</td>
    <td>2.62%</td>
    <td>2.58%</td>
    <td>2.39%</td>
    <td>2.29%</td>
    <td>5.49%</td>
    <td>4.36%</td>
    <td>4.48%</td>
    <td>4.42%</td>
    <td>3.40%</td>
    <td>3.09%</td>
</tr>
<tr>
    <td>SpUNet<sub>18</sub></td>
    <td>Voxel</td>
    <td>2.58%</td>
    <td>2.41%</td>
    <td>2.46%</td>
    <td>2.59%</td>
    <td>2.36%</td>
    <td>2.25%</td>
    <td>3.77%</td>
    <td>3.47%</td>
    <td>3.44%</td>
    <td>3.61%</td>
    <td>3.37%</td>
    <td>3.21%</td>
</tr>
<tr>
    <td>SpUNet<sub>34</sub></td>
    <td>Voxel</td>
    <td>2.60%</td>
    <td>2.52%</td>
    <td>2.47%</td>
    <td>2.66%</td>
    <td>2.41%</td>
    <td>2.29%</td>
    <td>4.41%</td>
    <td>4.33%</td>
    <td>4.34%</td>
    <td>4.39%</td>
    <td>4.20%</td>
    <td>4.11%</td>
</tr>
<tr>
    <td>RPVNet</td>
    <td>Fusion</td>
    <td>2.81%</td>
    <td>2.70%</td>
    <td>2.73%</td>
    <td>2.79%</td>
    <td>2.68%</td>
    <td>2.60%</td>
    <td>4.67%</td>
    <td>4.12%</td>
    <td>4.23%</td>
    <td>4.26%</td>
    <td>4.02%</td>
    <td>3.75%</td>
</tr>
<tr>
    <td>2DPASS</td>
    <td>Fusion</td>
    <td>2.74%</td>
    <td>2.53%</td>
    <td>2.51%</td>
    <td>2.51%</td>
    <td>2.62%</td>
    <td>2.46%</td>
    <td>2.32%</td>
    <td>2.35%</td>
    <td>2.45%</td>
    <td>2.30%</td>
    <td>2.73%</td>
    <td>2.27%</td>
</tr>
<tr>
    <td>SPVCNN<sub>18</sub></td>
    <td>Fusion</td>
    <td>2.57%</td>
    <td>2.44%</td>
    <td>2.49%</td>
    <td>2.54%</td>
    <td>2.40%</td>
    <td>2.31%</td>
    <td>3.46%</td>
    <td>2.90%</td>
    <td>3.07%</td>
    <td>3.41%</td>
    <td>2.36%</td>
    <td>2.32%</td>
</tr>
<tr>
    <td>SPVCNN<sub>34</sub></td>
    <td>Fusion</td>
    <td>2.61%</td>
    <td>2.49%</td>
    <td>2.54%</td>
    <td>2.61%</td>
    <td>2.37%</td>
    <td>2.28%</td>
    <td>3.61%</td>
    <td>3.03%</td>
    <td>3.07%</td>
    <td>3.10%</td>
    <td>2.99%</td>
    <td>2.86%</td>
</tr>
<tr>
    <td>CPGNet</td>
    <td>Fusion</td>
    <td>3.33%</td>
    <td>3.11%</td>
    <td>3.17%</td>
    <td>3.15%</td>
    <td>3.07%</td>
    <td>2.98%</td>
    <td>3.93%</td>
    <td>3.81%</td>
    <td>3.83%</td>
    <td>3.78%</td>
    <td>3.70%</td>
    <td>3.59%</td>
</tr>
<tr>
    <td>GFNet</td>
    <td>Fusion</td>
    <td>2.88%</td>
    <td>2.71%</td>
    <td>2.70%</td>
    <td>2.73%</td>
    <td>2.55%</td>
    <td>2.41%</td>
    <td>3.07%</td>
    <td>3.01%</td>
    <td>2.99%</td>
    <td>3.05%</td>
    <td>2.88%</td>
    <td>2.73%</td>
</tr>
<tr>
    <td>UniSeg</td>
    <td>Fusion</td>
    <td>2.76%</td>
    <td>2.61%</td>
    <td>2.63%</td>
    <td>2.65%</td>
    <td>2.45%</td>
    <td>2.37%</td>
    <td>3.93%</td>
    <td>3.73%</td>
    <td>3.78%</td>
    <td>3.67%</td>
    <td>3.51%</td>
    <td>3.43%</td>
</tr>
<tr>
    <td>KPConv</td>
    <td>Point</td>
    <td>3.37%</td>
    <td>3.27%</td>
    <td>3.34%</td>
    <td>3.32%</td>
    <td>3.28%</td>
    <td>3.20%</td>
    <td>4.97%</td>
    <td>4.88%</td>
    <td>4.90%</td>
    <td>4.91%</td>
    <td>4.78%</td>
    <td>4.68%</td>
</tr>
<tr>
    <td>PIDS<sub>1.25x</sub></td>
    <td>Point</td>
    <td>3.46%</td>
    <td>3.40%</td>
    <td>3.43%</td>
    <td>3.41%</td>
    <td>3.37%</td>
    <td>3.28%</td>
    <td>4.77%</td>
    <td>4.65%</td>
    <td>4.66%</td>
    <td>4.64%</td>
    <td>4.57%</td>
    <td>4.49%</td>
</tr>
<tr>
    <td>PIDS<sub>2.0x</sub></td>
    <td>Point</td>
    <td>3.53%</td>
    <td>3.47%</td>
    <td>3.49%</td>
    <td>3.51%</td>
    <td>3.34%</td>
    <td>3.27%</td>
    <td>4.91%</td>
    <td>4.83%</td>
    <td>4.72%</td>
    <td>4.89%</td>
    <td>4.66%</td>
    <td>4.47%</td>
</tr>
<tr>
    <td>PTv2</td>
    <td>Point</td>
    <td>2.42%</td>
    <td>2.34%</td>
    <td>2.46%</td>
    <td>2.55%</td>
    <td>2.48%</td>
    <td>2.19%</td>
    <td>4.95%</td>
    <td>4.78%</td>
    <td>4.71%</td>
    <td>4.94%</td>
    <td>4.69%</td>
    <td>4.62%</td>
</tr>
<tr>
    <td>WaffleIron</td>
    <td>Point</td>
    <td>4.01%</td>
    <td>2.65%</td>
    <td>3.06%</td>
    <td>2.59%</td>
    <td>2.54%</td>
    <td>2.46%</td>
    <td>3.91%</td>
    <td>2.57%</td>
    <td>2.86%</td>
    <td>2.67%</td>
    <td>2.58%</td>
    <td>2.51%</td>
</tr>

</table>  

</details>



<details open>
<summary>&nbsp<b>Other Datasets</b></summary>

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Type</th>
    <th>Method</th>
    <th>Modal</th>
    <th>UnCal</th>
    <th>TempS</th>
    <th>LogiS</th>
    <th>DiriS</th>
    <th>MetaC</th>
    <th>DeptS</th>
    <th>mIoU</th>
  </tr>
</thead>
<tbody>
  <!-- Waymo Open Dataset -->
  <tr>
    <td rowspan="3">Waymo Open</td>
    <td rowspan="3">High-Res</td>
    <td>PolarNet</td>
    <td>BEV</td>
    <td>3.92%</td>
    <td>1.93%</td>
    <td>1.90%</td>
    <td>1.91%</td>
    <td>2.39%</td>
    <td>1.84%</td>
    <td>58.33%</td>
  </tr>
  <tr>
    <td>MinkUNet</td>
    <td>Voxel</td>
    <td>1.70%</td>
    <td>1.70%</td>
    <td>1.74%</td>
    <td>1.76%</td>
    <td>1.69%</td>
    <td>1.59%</td>
    <td>68.67%</td>
  </tr>
  <tr>
    <td>SPVCNN</td>
    <td>Fusion</td>
    <td>1.81%</td>
    <td>1.79%</td>
    <td>1.80%</td>
    <td>1.88%</td>
    <td>1.74%</td>
    <td>1.69%</td>
    <td>68.86%</td>
  </tr>
  <!-- SemanticPOSS Dataset -->
  <tr>
    <td rowspan="3">SemanticPOSS</td>
    <td rowspan="3">Dynamic</td>
    <td>PolarNet</td>
    <td>BEV</td>
    <td>4.24%</td>
    <td>8.09%</td>
    <td>7.81%</td>
    <td>8.30%</td>
    <td>5.35%</td>
    <td>4.11%</td>
    <td>52.11%</td>
  </tr>
  <tr>
    <td>MinkUNet</td>
    <td>Voxel</td>
    <td>7.22%</td>
    <td>7.44%</td>
    <td>7.36%</td>
    <td>7.62%</td>
    <td>5.66%</td>
    <td>5.48%</td>
    <td>56.32%</td>
  </tr>
  <tr>
    <td>SPVCNN</td>
    <td>Fusion</td>
    <td>8.80%</td>
    <td>6.53%</td>
    <td>6.91%</td>
    <td>7.41%</td>
    <td>4.61%</td>
    <td>3.98%</td>
    <td>53.51%</td>
  </tr>
  <!-- SemanticSTF Dataset -->
  <tr>
    <td rowspan="3">SemanticSTF</td>
    <td rowspan="3">Weather</td>
    <td>PolarNet</td>
    <td>BEV</td>
    <td>5.76%</td>
    <td>4.94%</td>
    <td>4.49%</td>
    <td>4.53%</td>
    <td>4.17%</td>
    <td>4.12%</td>
    <td>51.26%</td>
  </tr>
  <tr>
    <td>MinkUNet</td>
    <td>Voxel</td>
    <td>5.29%</td>
    <td>5.21%</td>
    <td>4.96%</td>
    <td>5.10%</td>
    <td>4.78%</td>
    <td>4.72%</td>
    <td>50.22%</td>
  </tr>
  <tr>
    <td>SPVCNN</td>
    <td>Fusion</td>
    <td>5.85%</td>
    <td>5.53%</td>
    <td>5.16%</td>
    <td>5.05%</td>
    <td>5.12%</td>
    <td>4.97%</td>
    <td>51.73%</td>
  </tr>
  <!-- ScribbleKITTI Dataset -->
  <tr>
    <td rowspan="3">ScribbleKITTI</td>
    <td rowspan="3">Scribble</td>
    <td>PolarNet</td>
    <td>BEV</td>
    <td>4.65%</td>
    <td>4.59%</td>
    <td>4.56%</td>
    <td>4.55%</td>
    <td>3.25%</td>
    <td>3.09%</td>
    <td>55.22%</td>
  </tr>
  <tr>
    <td>MinkUNet</td>
    <td>Voxel</td>
    <td>7.97%</td>
    <td>7.13%</td>
    <td>7.29%</td>
    <td>7.21%</td>
    <td>5.93%</td>
    <td>5.74%</td>
    <td>59.87%</td>
  </tr>
  <tr>
    <td>SPVCNN</td>
    <td>Fusion</td>
    <td>7.04%</td>
    <td>6.63%</td>
    <td>6.93%</td>
    <td>6.66%</td>
    <td>5.34%</td>
    <td>5.13%</td>
    <td>60.22%</td>
  </tr>
  <!-- Synth4D Dataset -->
  <tr>
    <td rowspan="3">Synth4D</td>
    <td rowspan="3">Synthetic</td>
    <td>PolarNet</td>
    <td>BEV</td>
    <td>1.68%</td>
    <td>0.93%</td>
    <td>0.75%</td>
    <td>0.72%</td>
    <td>1.54%</td>
    <td>0.69%</td>
    <td>85.63%</td>
  </tr>
  <tr>
    <td>MinkUNet</td>
    <td>Voxel</td>
    <td>2.43%</td>
    <td>2.72%</td>
    <td>2.43%</td>
    <td>2.05%</td>
    <td>4.01%</td>
    <td>2.39%</td>
    <td>69.11%</td>
  </tr>
  <tr>
    <td>SPVCNN</td>
    <td>Fusion</td>
    <td>2.21%</td>
    <td>2.35%</td>
    <td>1.86%</td>
    <td>1.70%</td>
    <td>3.44%</td>
    <td>1.67%</td>
    <td>69.68%</td>
  </tr>
  <!-- S3DIS Dataset -->
  <tr>
    <td rowspan="3">S3DIS</td>
    <td rowspan="3">Indoor</td>
    <td>PointNet++</td>
    <td>Point</td>
    <td>9.13%</td>
    <td>8.36%</td>
    <td>7.83%</td>
    <td>8.20%</td>
    <td>6.93%</td>
    <td>6.79%</td>
    <td>56.96%</td>
  </tr>
  <tr>
    <td>DGCNN</td>
    <td>Point</td>
    <td>6.00%</td>
    <td>6.23%</td>
    <td>6.35%</td>
    <td>7.12%</td>
    <td>5.47%</td>
    <td>5.39%</td>
    <td>54.50%</td>
  </tr>
  <tr>
    <td>PAConv</td>
    <td>Point</td>
    <td>8.38%</td>
    <td>5.87%</td>
    <td>6.03%</td>
    <td>5.98%</td>
    <td>4.67%</td>
    <td>4.57%</td>
    <td>66.60%</td>
  </tr>
    
</tbody>
</table>

</details>



### Domain-Shift 3D Uncertainty

<details open>
<summary>&nbsp<b>nuScenes-C & SemanticKITTI-C</b></summary>

<table>
  <thead>
    <tr>
      <th rowspan="2">Type</th>
      <th colspan="6">nuScenes-C</th>
      <th colspan="6">SemanticKITTI-C</th>
    </tr>
    <tr>
      <th>UnCal</th>
      <th>TempS</th>
      <th>LogiS</th>
      <th>DiriS</th>
      <th>MetaC</th>
      <th>DeptS</th>
      <th>UnCal</th>
      <th>TempS</th>
      <th>LogiS</th>
      <th>DiriS</th>
      <th>MetaC</th>
      <th>DeptS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Clean</td>
      <td>2.45%</td>
      <td>2.34%</td>
      <td>2.34%</td>
      <td>2.42%</td>
      <td>2.29%</td>
      <td>2.23%</td>
      <td>3.04%</td>
      <td>3.01%</td>
      <td>3.08%</td>
      <td>3.30%</td>
      <td>2.69%</td>
      <td>2.63%</td>
    </tr>
    <tr>
      <td>Fog</td>
      <td>5.52%</td>
      <td>5.42%</td>
      <td>5.49%</td>
      <td>5.43%</td>
      <td>4.77%</td>
      <td>4.72%</td>
      <td>12.66%</td>
      <td>12.55%</td>
      <td>12.67%</td>
      <td>12.48%</td>
      <td>11.08%</td>
      <td>10.94%</td>
    </tr>
    <tr>
      <td>Wet Ground</td>
      <td>2.63%</td>
      <td>2.54%</td>
      <td>2.54%</td>
      <td>2.64%</td>
      <td>2.55%</td>
      <td>2.52%</td>
      <td>3.55%</td>
      <td>3.46%</td>
      <td>3.54%</td>
      <td>3.72%</td>
      <td>3.33%</td>
      <td>3.28%</td>
    </tr>
    <tr>
      <td>Snow</td>
      <td>13.79%</td>
      <td>13.32%</td>
      <td>13.53%</td>
      <td>13.59%</td>
      <td>11.37%</td>
      <td>11.31%</td>
      <td>7.10%</td>
      <td>6.96%</td>
      <td>6.95%</td>
      <td>7.26%</td>
      <td>5.99%</td>
      <td>5.63%</td>
    </tr>
    <tr>
      <td>Motion Blur</td>
      <td>9.54%</td>
      <td>9.29%</td>
      <td>9.37%</td>
      <td>9.01%</td>
      <td>8.32%</td>
      <td>8.29%</td>
      <td>11.31%</td>
      <td>11.16%</td>
      <td>11.24%</td>
      <td>12.13%</td>
      <td>9.00%</td>
      <td>8.97%</td>
    </tr>
    <tr>
      <td>Beam Missing</td>
      <td>2.58%</td>
      <td>2.48%</td>
      <td>2.49%</td>
      <td>2.57%</td>
      <td>2.53%</td>
      <td>2.47%</td>
      <td>2.87%</td>
      <td>2.83%</td>
      <td>2.84%</td>
      <td>2.98%</td>
      <td>2.83%</td>
      <td>2.79%</td>
    </tr>
    <tr>
      <td>Crosstalk</td>
      <td>13.64%</td>
      <td>13.00%</td>
      <td>12.97%</td>
      <td>13.44%</td>
      <td>9.98%</td>
      <td>9.73%</td>
      <td>4.93%</td>
      <td>4.83%</td>
      <td>4.86%</td>
      <td>4.81%</td>
      <td>3.54%</td>
      <td>3.48%</td>
    </tr>
    <tr>
      <td>Incomplete Echo</td>
      <td>2.44%</td>
      <td>2.33%</td>
      <td>2.33%</td>
      <td>2.42%</td>
      <td>2.32%</td>
      <td>2.21%</td>
      <td>3.21%</td>
      <td>3.19%</td>
      <td>3.25%</td>
      <td>3.48%</td>
      <td>2.84%</td>
      <td>2.19%</td>
    </tr>
    <tr>
      <td>Cross Sensor</td>
      <td>4.25%</td>
      <td>4.15%</td>
      <td>4.20%</td>
      <td>4.28%</td>
      <td>4.06%</td>
      <td>3.20%</td>
      <td>3.15%</td>
      <td>3.13%</td>
      <td>3.18%</td>
      <td>3.43%</td>
      <td>3.17%</td>
      <td>2.96%</td>
    </tr>
    <tr>
      <td>Average</td>
      <td>6.78%</td>
      <td>6.57%</td>
      <td>6.62%</td>
      <td>6.67%</td>
      <td>5.74%</td>
      <td>5.56%</td>
      <td>6.10%</td>
      <td>6.01%</td>
      <td>6.07%</td>
      <td>6.29%</td>
      <td>5.22%</td>
      <td>5.03%</td>
    </tr>
  </tbody>
</table>

</details>


### Model Configurations

| <img src="docs/figs/miou.png" align="center" width="260"> | <img src="docs/figs/backend.png" align="center" width="260"> | <img src="docs/figs/modality.png" align="center" width="260"> |
| :-: | :-: | :-: |
| ECE vs. mIoU | SparseConv Backend | LiDAR Modality |



## :memo: TODO List
- [x] Initial release. 🚀
- [x] Add 3D calibration benchmarks.
- [x] Add 3D calibration algorithms.
- [x] Add acknowledgments.
- [x] Add citations.
- [ ] Add more 3D scene understanding models.



## Citation
If you find this work helpful for your research, please kindly consider citing our papers:

```bibtex
@article{kong2024calib3d,
    author = {Lingdong Kong and Xiang Xu and Jun Cen and Wenwei Zhang and Liang Pan and Kai Chen and Ziwei Liu},
    title = {Calib3D: Calibrating Model Preferences for Reliable 3D Scene Understanding},
    journal = {arXiv preprint arXiv:2403.17010},
    year = {2024},
}
```
```bibtex
@misc{mmdet3d,
    title = {MMDetection3D: OpenMMLab Next-Generation Platform for General 3D Object Detection},
    author = {MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year = {2020}
}
```



## License

This work is under the <a rel="license" href="https://www.apache.org/licenses/LICENSE-2.0">Apache License Version 2.0</a>, while some specific implementations in this codebase might be with other licenses. Kindly refer to [LICENSE.md](docs/LICENSE.md) for a more careful check, if you are using our code for commercial matters.



## Acknowledgements

This work is developed based on the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) codebase.

> <img src="https://github.com/open-mmlab/mmdetection3d/blob/main/resources/mmdet3d-logo.png" width="31%"/><br>
> MMDetection3D is an open-source toolbox based on PyTorch, towards the next-generation platform for general 3D perception. It is a part of the OpenMMLab project developed by MMLab.

Part of the benchmarked models are from the [OpenPCSeg](https://github.com/PJLab-ADG/OpenPCSeg) and [Pointcept](https://github.com/Pointcept/Pointcept) codebases.

We acknowledge the use of the following public resources, during the course of this work: <sup>1</sup>[nuScenes](https://www.nuscenes.org/nuscenes), <sup>2</sup>[SemanticKITTI](http://semantic-kitti.org/), <sup>3</sup>[Waymo Open](https://waymo.com/open), <sup>4</sup>[SemanticPOSS](http://www.poss.pku.edu.cn/semanticposs.html), <sup>5</sup>[Synth4D](https://github.com/saltoricristiano/gipso-sfouda), <sup>6</sup>[SemanticSTF](https://github.com/xiaoaoran/SemanticSTF), <sup>7</sup>[ScribbleKITTI](https://github.com/ouenal/scribblekitti), <sup>8</sup>[S3DIS](http://buildingparser.stanford.edu/dataset.html), <sup>9</sup>[Robo3D](https://github.com/ldkong1205/Robo3D), <sup>10</sup>[lidar-bonnetal](https://github.com/PRBonn/lidar-bonnetal), <sup>11</sup>[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), <sup>12</sup>[SPConv](https://github.com/traveller59/spconv), <sup>13</sup>[TorchSparse](https://github.com/mit-han-lab/torchsparse), <sup>14</sup>[WaffleIron](https://github.com/valeoai/WaffleIron), <sup>15</sup>[PolarMix](https://github.com/xiaoaoran/polarmix), <sup>16</sup>[LaserMix](https://github.com/ldkong1205/LaserMix), <sup>17</sup>[FRNet](https://github.com/Xiangxu-0103/FRNet), and <sup>18</sup>[Open3D-ML](https://github.com/isl-org/Open3D-ML).

We thank the exceptional contributions from the above open-source repositories! :heart:

