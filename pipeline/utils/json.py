# -- coding: utf-8 --
import json


def json2dict(path: str) -> dict:
    with open(path, "r", encoding="utf8") as f:
        content = json.load(f)
    return content