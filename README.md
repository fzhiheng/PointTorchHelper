# PointTorchHelper
A training framework for point cloud deep learning

## 框架思路
参考mmdetection

将训练模块解耦为数据集、模型两大部分，其中数据集使用pipelines作为dataset构造的整体流程。

所有的类使用注册器进行注册，后续自定义新的类，只需要在构造的时候注册一下，并在对应的包下面import一下就完成了注册（因为注册发生在import过程中）

## 数据集pipelines
pipelines会将常见的点云数据预处理以及增强方式加入其中，数据集会将自动驾驶经典数据集比如KITTI、Nuscene等数据集加入其中，
数据集的不同任务会新开几个仓库处理，包括但不限于：里程计与配准、场景流、语义分割、目标检测与追踪、重建。


## 模型
经典点云模型以及期刊会议中最新的模型