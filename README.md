# EE5346 项目
# Loop Closure Verification with Radar Depth and ICP-based Descriptor Filtering

- 运行三个不同的文件
    - **运行Baseline方法** ：直接运行verification_Baseline.py,可以获得由Baseline方法得出的PR曲线。
    - **运行基于Baseline改进的利用雷达深度信息的方法** ：直接运行verification_PRcurve.py,可以获得利用雷达深度信息的方法计算得出的PR曲线。
    - **运行最终预测的结果** ：直接运行verification_Predict.py，可以获得
  `robotcar_qAutumn_dbNight_val_final.txt`，`robotcar_qAutumn_dbSunCloud_val_final.txt`预测的结果。**预测的结果需要用VScode打开，Windows的文本编辑器打开是乱码** 。

## 各个文件的放置位置

因为文件代码里面写的读取文件的路径是相对路径，如果想要运行这三个文件，**必须按照固定的位置放置。** 下面是本项目的全部文件放置方式：

`注意：`github直接下载的文件EE5346_2023_project-main为一级目录（根目录），**需要将`Autumn_val`、`Night_val`、`Suncloud_val`这个三个数据集放在根目录下**。
#### 一级目录（项目的根目录）
- ├── Autumn_mini_query
- ├── Autumn_val
- ├── Kudamm_diff_final.txt
- ├── Kudamm_easy_final.txt
- ├── Kudamm_mini_query
- ├── Kudamm_mini_ref
- ├── Night_mini_ref
- ├── Night_val
- ├── README.md
- ├── robotcar-dataset-sdk
- ├── robotcar_qAutumn_dbNight_diff_final.txt
- ├── robotcar_qAutumn_dbNight_easy_final.txt
- ├── robotcar_qAutumn_dbNight_val_final.txt
- ├── robotcar_qAutumn_dbNight_val_result.txt
- ├── robotcar_qAutumn_dbSunCloud_diff_final.txt
- ├── robotcar_qAutumn_dbSunCloud_easy_final.txt
- ├── robotcar_qAutumn_dbSunCloud_val_final.txt
- ├── robotcar_qAutumn_dbSunCloud_val_result.txt
- ├── Suncloud_mini_ref
- ├── Suncloud_val
- ├── verification_Baseline.py
#### 二级目录（一级目录中的robotcar-dataset-sdk）
- ├── extrinsics
- ├── LICENSE
- ├── matlab
- ├── models
- ├── python
- ├── README.md
- └── tags
#### 三级目录（二级目录中的python）
- ├── build_pointcloud.py
- ├── image.py
- ├── __init__.py
- ├── interpolate_poses.py
- ├── play_images.py
- ├── play_radar.py
- ├── play_road_boundary.py
- ├── play_velodyne.py
- ├── pointcloud_world.txt
- ├── project_laser_into_camera1.py
- ├── project_laser_into_camera.py
- ├── project.sh
- ├── __pycache__
- ├── radar.py
- ├── README.md
- ├── requirements.txt
- ├── road_boundary.py
- ├── transform.py
- ├── velodyne.py
- ├── verification_PRcurve.py
- ├── verification_Predict.py




## 运行Baseline方法

- 将verification_Baseline.py文件放在**一级目录**下,可以获得由Baseline方法得出的PR曲线。**（用vscode按F5运行）**
- 画出的PR曲线为：`Kudamm_easy_final.txt`, `Kudamm_diff_final.txt`, `robotcar_qAutumn_dbNight_easy_final.txt`, `robotcar_qAutumn_dbNight_diff_final.txt`, `robotcar_qAutumn_dbSunCloud_easy_final.txt`, `robotcar_qAutumn_dbSunCloud_diff_final.txt`这四个数据集的曲线。

## 运行基于Baseline改进的利用雷达深度信息的方法

- 将verification_PRcurve.py文件放在**三级目录**下,可以获得由利用雷达深度信息的方法计算得出的PR曲线。**（用vscode按F5运行）**
- 画出的PR曲线为： `robotcar_qAutumn_dbNight_easy_final.txt`, `robotcar_qAutumn_dbNight_diff_final.txt`, `robotcar_qAutumn_dbSunCloud_easy_final.txt`, `robotcar_qAutumn_dbSunCloud_diff_final.txt`这四个数据集的曲线。

## 运行最终预测的结果

- 将verification_Predict.py文件放在**三级目录**下，可以获得
  `robotcar_qAutumn_dbNight_val_final.txt`，`robotcar_qAutumn_dbSunCloud_val_final.txt`预测的结果。**（用vscode按F5运行）**

