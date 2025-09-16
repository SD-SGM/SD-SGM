<div align="center">

# Reliable LiDAR Loop Detection through Structural Descriptors and Semantic Graph Matching

</div>

# 1. Prerequisites

We tested our code on **ubuntu 20.04** and **ubuntu 22.04**. 

- **[Eigen](https://gitlab.com/libeigen/eigen/-/releases) (3.3.7)**
- **[PCL](https://github.com/PointCloudLibrary/pcl/releases) (1.12)**
- **[Ceres-solver](https://github.com/ceres-solver/ceres-solver/tags) (2.1.0)**

You need to install these libraries from official guidance.'

# 2. Data

## KITTI

You can download the point cloud dataset from the KITTI official [website](https://www.cvlibs.net/datasets/kitti/). In our experiments, we use the labels from the SegNet4D. For the convenience, you can download from [here](https://1drv.ms/u/c/807229e8eebd9eb1/Edbvmep12YdGvGoyMPiKEVAB4VyR7-brHEzbJ_zFS3QeOQ).

Loop pairs: we use the distance-based criteria from the [SSC](https://github.com/lilin-hitcrt/SSC). You also can download from our [link](https://1drv.ms/u/c/807229e8eebd9eb1/Edbvmep12YdGvGoyMPiKEVAB4VyR7-brHEzbJ_zFS3QeOQ).

## 3. Usage

### 3.1 Install

```bash
git clone https://github.com/SD-SGM/SD-SGM.git
mkdir build
cd build
cmake ..
make -j8
```

### 3.2 Loop Closure Detection

- **KITTI dataset (distance-based)** 

Modify `config/config_kitti_graph.yaml`

```yaml
eval_seq:
  cloud_path: "xx/kitti/sequences/08/velodyne/" # your LiDAR scans
  label_path: "xx/SegNet4D_predicitions/kitti/sequences/08/predictions/" # semantic predictions from our link
  pairs_file: "../loop_data/pairs/pairs_kitti/neg_100/08.txt" # loop pairs
  out_file: "../out/kitti/08.txt"  # output file for evaluating
```

Then, you can run the `.bin` file following this:

```bash
cd /SD-SGM/bin
./eval_lcd_seq
```

you can find the output file in the `SD-SGM/out/`. for evaluating, you can run:

```bash
cd /SD-SGM/scripts
python pr_curve.py
```




# **License**

This project is free software made available under the MIT License. For details see the LICENSE file.
