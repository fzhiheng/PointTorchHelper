import os
import sys
from typing import List, Dict

def mkdirs_dir_trees(root, tree_info: Dict[str,Dict]):
    root = os.path.abspath(root)
    for dir_name, sub_tree in tree_info.items():
        dir_path = os.path.join(root, dir_name)
        if not sub_tree:
            mkdirs_dir_trees(dir_path, sub_tree)
        else:
            os.makedirs(dir_path, exist_ok=True)

def makedirs_ok(_a:str, *paths, exist_ok=True):
    path = os.path.join(_a, *paths)
    os.makedirs(path, exist_ok=exist_ok)
    return path
