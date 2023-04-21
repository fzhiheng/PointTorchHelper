# -- coding: utf-8 --
import math
from typing import Container, List

import numpy as np

from .base import BaseRandomSample, BaseRangeLimit
from ..builder import PIPELINES

# @Time : 2023/4/21 22:08
# @Author : Zhiheng Feng
# @File : pointcloud.py
# @Software : PyCharm
# -- coding: utf-8 --





@PIPELINES.register
class RangeLimit(BaseRangeLimit):
    def __init__(
        self,
        *,
        x_range=(-30, 30),
        y_range=(-1, 1.4),
        z_range=(0, 35),
        lidar_keys: Container = ("lidar1", "lidar2", "point_clouds"),
    ):
        super().__init__(x_range=x_range, y_range=y_range, z_range=z_range)
        self.lidar_keys = lidar_keys

    def __call__(self, data: dict[str, dict]):
        for key in self.lidar_keys:
            pc = data.get(key, None)
            if pc is None:
                continue
            if isinstance(pc, np.ndarray):
                new_pc = self.limit_range(pc)
                data[key] = new_pc
            if isinstance(pc, List):
                new_pc = [self.limit_range(p) for p in pc]
                data[key] = new_pc
        return data


@PIPELINES.register
class RandomSample(BaseRandomSample):
    def __init__(self, *, num_points, allow_less_points, lidar_keys: Container = ("lidar1", "lidar2", "point_clouds")):
        super().__init__(num_points, allow_less_points)
        self.lidar_keys = lidar_keys

    def __call__(self, data: dict):
        for key in self.lidar_keys:
            pc = data.get(key, None)
            if pc is None:
                continue
            if isinstance(pc, np.ndarray):
                new_pc = self.random_sample(pc)
                data[key] = new_pc
            if isinstance(pc, List):
                new_pc = [self.random_sample(p) for p in pc]
                data[key] = new_pc
        return data


@PIPELINES.register
class ShakeAug(object):
    def __init__(
        self,
        *,
        x_clip,
        y_clip,
        z_clip,
    ):
        super().__init__()
        self.aug_max = aug_matrix(x_clip, y_clip, z_clip)

    def __call__(self, data: dict) -> dict:
        lidar1 = data["lidar1"]
        T_gt_lidar = data["T_gt_lidar"]
        lidar1 = left_hand_mul_trans(lidar1, self.aug_max)
        T_gt_lidar = T_gt_lidar @ np.linalg.inv(self.aug_max)
        data["lidar1"] = lidar1
        data["T_gt_lidar"] = T_gt_lidar
        return data


@PIPELINES.register
class CoordinateSysTrans(object):
    def __init__(self, trans_matrix, lidar_keys: Container = ("lidar1", "lidar2", "point_clouds")) -> None:
        super().__init__()
        self.matrix = trans_matrix

    def __call__(self, data: dict) -> dict:
        data_tmp = dict()
        for k, v in data.items():
            if v.get("type", "other") == "point":
                v["data"] = left_hand_mul_trans(v["data"], self.matrix)
            data_tmp[k] = v
        return data_tmp


@PIPELINES.register
class MatToEuler(object):
    def __init__(self, cy_thresh=None, seq="zyx", pop_flag=True):
        self.cy_thresh = cy_thresh
        self.seq = seq
        self.pop_flag = pop_flag

    def __call__(self, data: dict) -> dict:
        if self.pop_flag:
            T_gt = data.pop("R")
        else:
            T_gt = data["R"]
        M = np.asarray(T_gt)
        if self.cy_thresh is None:
            self.cy_thresh = np.finfo(M.dtype).eps * 4

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33 * r33 + r23 * r23)
        if self.seq == "zyx":
            if cy > self.cy_thresh:  # cos(y) not close to zero, standard form
                z = math.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
                y = math.atan2(r13, cy)  # atan2(sin(y), cy)
                x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
            else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
                # so r21 -> sin(z), r22 -> cos(z) and
                z = math.atan2(r21, r22)
                y = math.atan2(r13, cy)  # atan2(sin(y), cy)
                x = 0.0
        elif self.seq == "xyz":
            if cy > self.cy_thresh:
                y = math.atan2(-r31, cy)
                x = math.atan2(r32, r33)
                z = math.atan2(r21, r11)
            else:
                z = 0.0
                if r31 < 0:
                    y = np.pi / 2
                    x = math.atan2(r12, r13)
                else:
                    y = -np.pi / 2
        else:
            raise Exception("Sequence not recognized")
        data["euler"] = {"x": x, "y": y, "z": z}
        return data


@PIPELINES.register
class EulerToQuat(object):
    def __init__(self, isRadian=True, pop_flag=True):
        self.isRadian = isRadian
        self.pop_flag = pop_flag

    def __call__(self, data: dict):
        if self.pop_flag:
            euler = data.pop("euler")
        else:
            euler = data["euler"]
        x = euler["x"]
        y = euler["y"]
        z = euler["z"]
        if not self.isRadian:
            z = ((np.pi) / 180.0) * z
            y = ((np.pi) / 180.0) * y
            x = ((np.pi) / 180.0) * x
        z = z / 2.0
        y = y / 2.0
        x = x / 2.0
        cz = math.cos(z)
        sz = math.sin(z)
        cy = math.cos(y)
        sy = math.sin(y)
        cx = math.cos(x)
        sx = math.sin(x)
        data["quat"] = np.array(
            [cx * cy * cz - sx * sy * sz, cx * sy * sz + cy * cz * sx, cx * cz * sy - sx * cy * sz, cx * cy * sz + sx * cz * sy]
        )
        return data


@PIPELINES.register
class QuatToMat(object):
    def __init__(self, pop_flag):
        self.pop_flag = pop_flag

    def __call__(self, data: dict):
        if self.pop_flag:
            q = data.pop("quat")
        else:
            q = data["quat"]
        w, x, y, z = q
        Nq = w * w + x * x + y * y + z * z
        if Nq < 1e-8:
            return np.eye(3)
        s = 2.0 / Nq
        X = x * s
        Y = y * s
        Z = z * s
        wX = w * X
        wY = w * Y
        wZ = w * Z
        xX = x * X
        xY = x * Y
        xZ = x * Z
        yY = y * Y
        yZ = y * Z
        zZ = z * Z
        data["mat"] = np.array(
            [[1.0 - (yY + zZ), xY - wZ, xZ + wY], [xY + wZ, 1.0 - (xX + zZ), yZ - wX], [xZ - wY, yZ + wX, 1.0 - (xX + yY)]]
        )


def left_hand_mul_trans(pc: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    C = pc.shape[-1]
    if C == 3:
        pc = np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=-1)
        pc = pc @ matrix.T
        pc = pc[:, :3]
    else:
        reflect = pc[..., -1]
        pc = pc @ matrix.T
        pc[..., -1] = reflect
    return pc


def aug_matrix(x_clip, y_clip, z_clip):
    """

    :return: 返回一个[4,4]随机增强矩阵 np.float64,还有对应的q和t
    """

    anglex = np.clip(0.01 * np.random.randn(), -x_clip, x_clip).astype(np.float32) * np.pi / 4.0
    angley = np.clip(0.05 * np.random.randn(), -y_clip, y_clip).astype(np.float32) * np.pi / 4.0
    anglez = np.clip(0.01 * np.random.randn(), -z_clip, z_clip).astype(np.float32) * np.pi / 4.0

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

    # 3*3
    R_trans = Rx.dot(Ry).dot(Rz)

    xx = np.clip(0.1 * np.random.randn(), -0.2, 0.2).astype(np.float32)  # 就是一个数,维度是0
    yy = np.clip(0.05 * np.random.randn(), -0.15, 0.15).astype(np.float32)
    zz = np.clip(0.5 * np.random.randn(), -1, 1).astype(np.float32)

    add_3 = np.array([[xx], [yy], [zz]])
    T_trans = np.concatenate([R_trans, add_3], axis=-1)
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    filler = np.expand_dims(filler, axis=0)  ##1*4
    T_trans = np.concatenate([T_trans, filler], axis=0)  # 4*4

    return T_trans
