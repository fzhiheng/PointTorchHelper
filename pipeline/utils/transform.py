# -- coding: utf-8 --
import os
import numpy as np

# @Time : 2023/3/17 15:27
# @Author : Zhiheng Feng
# @File : transform
# @Software : VsCode

"""
用于不同坐标系之间坐标的变换,不同坐标系下变换矩阵的变换
"""

# Transition matrix and transformation matrix


def coor_in_B_from_A(src_points: np.ndarray, transition_matrix: np.ndarray):
    """使用变换矩阵,将A坐标系下的坐标变换到B坐标系下,A,B是两组基

    Args:
        src_points (np.ndarry): (*,n,3) A坐标系下的坐标
        transform_matrix (np.ndarry): (4,4) B坐标系到A坐标系的过渡矩阵,A中坐标到B中坐标的变换矩阵
        (A=B@T,T为B到A的过渡矩阵,B坐标系下的坐标O,A坐标下的坐标M,则O=TM)

    Returns:
        np.ndarry: B坐标系下的坐标
    """

    shape = src_points.shape
    pad_shape = list(shape[:-1])
    pad = np.ones(pad_shape.append(1))
    pad_points = np.concatenate([src_points, pad], axis=-1)  # (*,n,4)
    new_points = pad_points @ transition_matrix.T  # (*,n,4)
    return new_points[..., :3]


def transform_between_B_from_A(src_matrixs: np.ndarray, transition_matrix: np.ndarray):
    """使用变换矩阵,将A坐标系的帧间变换,变换为B坐标系的帧间变换矩阵

    Args:
        src_matrixs (np.ndarry): (n,4,4) 两个A坐标系之间的变换矩阵
        transition_A_to_B (np.ndarry): (4,4)  B坐标系到A坐标系的过渡矩阵,A中坐标到B中坐标的变换矩阵
        (A=BT,T为B到A的过渡矩阵,B坐标系下的坐标O,A坐标下的坐标M,则O=TM)

    Returns:
        np.ndarry: 对应的两个N坐标系之间的变换矩阵
    """
    transform_matrix = np.linalg.inv(transition_matrix)
    target_matrix = transition_matrix @ src_matrixs @ transform_matrix
    return target_matrix


def transform_to_transition(src_matrixs: np.ndarray):
    """将两个坐标系下坐标之间的变换矩阵转换成两个坐标系之间的过渡矩阵

    Args:
        src_matrixs (np.ndarry): (*,4,4) 坐标之间的变换矩阵
    """
    target_matrix = np.linalg.inv(src_matrixs)
    return target_matrix


def matrix_pad(src_matrixs: np.ndarray):
    """将矩阵补全为4*4的形式

    Args:
        src_matrixs (np.ndarry): (*,3,4) or (*,12)
    """
    shape = src_matrixs.shape
    if shape[-1] != 12:
        if shape[-2:] != (3, 4):
            raise ValueError(f"expert src_matrixs shape is (*,3,4) or (*,12), but get {shape}")
    
    src_matrixs = np.reshape(src_matrixs, (-1, 3, 4))
    N = src_matrixs.shape[0]
    padding = np.array([0, 0, 0, 1]).reshape(1, 1, 4)
    paddings = np.repeat(padding, N, axis=0)
    target_matrixs = np.concatenate([src_matrixs, paddings], axis=-2)  # (n,4,4)
    return target_matrixs


