import os
import uuid

import requests

from datetime import datetime


def save_result_to_file(msg_str, model_folder="AHRec"):
    """
    保存结果字符串到指定文件夹，文件名包含时间和短 UUID。

    :param msg_str: 要保存的字符串内容
    :param folder: 保存的文件夹路径（默认是 'results'）
    :return: 完整文件路径
    """
    # 创建文件夹（如不存在）
    root_file = "log_wechat_msg"
    target_file = os.path.join(root_file, model_folder)
    os.makedirs(target_file, exist_ok=True)

    # 生成文件名：时间戳 + 5位UUID
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:5]
    filename = f"{model_folder}-{timestamp}-{short_uuid}.log"

    # 完整路径
    full_path = os.path.join(target_file, filename)

    # 保存文件
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(msg_str)

    return full_path


def dict_to_table_str(d, indent=0):
    """
    将嵌套字典格式化为文本表格字符串。
    如果是顶层调用，在最上方加上当前时间。
    当遇到 'timestamp' 属性时，不再递归其值。

    :param d: dict
    :param indent: 缩进层级
    :param is_root: 是否是顶层调用（自动设置，不用手动传）
    :return: str
    """
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict, got {type(d)}")

    lines = []

    space = '  ' * indent
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{space}{key}:")
            lines.append(dict_to_table_str(value, indent + 1))
        else:
            lines.append(f"{space}{key}: {value}")

    return '\n'.join(lines)


def dict_extract_keys_recursive(d, wanted_keys):
    """
    递归地从嵌套 dict 中查找指定 key 并按顺序格式化为表格字符串。

    :param d: 嵌套字典
    :param wanted_keys: 想要输出的字段名列表（只按 key 匹配，无需路径）
    :return: str 表格
    """

    def find_key_recursive(d, target_key):
        """递归查找 key 值"""
        if isinstance(d, dict):
            for k, v in d.items():
                if k == target_key:
                    return v
                elif isinstance(v, dict):
                    result = find_key_recursive(v, target_key)
                    if result is not None:
                        return result
        return None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[{now}]"]
    for key in wanted_keys:
        value = find_key_recursive(d, key)
        lines.append(f"{key}: {value}")
    return '\n'.join(lines)
