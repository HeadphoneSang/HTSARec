import importlib.util
import os


def load_class_from_file(file_path, class_name):
    """从指定的 Python 文件加载指定类"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} 文件不存在")

    # 获取模块名（不含路径和扩展名）
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # 构造模块 spec
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"无法加载模块 {module_name}")

    # 加载模块
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取类
    cls = getattr(module, class_name, None)
    if cls is None:
         raise ImportError(f"在 {file_path} 中未找到类 {class_name}")

    return cls
