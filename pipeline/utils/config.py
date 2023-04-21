import os
import sys
import types
from importlib import import_module

from addict import Dict


def config_from_py(path: str) -> Dict:
    """读取一个py文件

    Args:
        path (str): _description_

    Raises:
        IOError: _description_

    Returns:
        Dict: _description_
    """
    
    config_path = os.path.abspath(os.path.expanduser(path))
    assert os.path.isfile(config_path)
    if config_path.endswith(".py"):
        module_name = os.path.basename(config_path)[:-3]
        config_dir = os.path.dirname(config_path)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith("__")
            and not isinstance(value, types.ModuleType)
            and not isinstance(value, types.FunctionType)
        }
        return Dict(cfg_dict)
    else:
        raise IOError("not .py file!")
    

def config_from_file(file_path):
    config_path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.isfile(config_path):
        raise FileExistsError(f"{config_path} do not exist!")
    if config_path.endswith(".py"):
        cfg_dict = config_from_py(config_path)
    else:
        raise IOError("just support .py file now!")
    return cfg_dict


