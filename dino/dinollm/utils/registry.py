# 导入类型提示需要的工具（不用管原理，照抄就行）
from typing import Callable, Generic, Iterable, List, TypeVar

# 定义一个“通用类型标记” T
# 意思：这个注册器可以存【任意类型】（类、函数、对象都能存）
T = TypeVar("T")


# ==============================================
# 注册器类（核心：一个自动管理插件的花名册）
# 作用：把 名字 ↔ 类/函数 对应起来，方便查找、调用
# ==============================================
class Registry(Generic[T]):

    # 初始化：创建一个空花名册
    # 参数 type：给这个花名册起个说明（比如“优化器”“注意力”“模型”）
    def __init__(self, type: str):
        self._registry = {}  # 真正存数据的字典：{名字: 对象}
        self._type = type    # 花名册用途（报错时会显示）

    # ==========================================
    # 装饰器：注册一个东西到花名册
    # 使用方法：@注册器.register("名字")
    # ==========================================
    def register(self, name: str) -> Callable[[T], None]:
        # 如果名字已经被注册过了 → 报错，防止重复
        if name in self._registry:
            raise KeyError(f"{self._type} '{name}' 已经被注册过了！")

        # 定义一个装饰器（用来接收要注册的类/函数）
        def decorator(item: T) -> None:
            # 把 item（类/函数）存进字典
            # key = 你起的名字
            # value = 类/函数本身
            self._registry[name] = item

        # 返回装饰器
        return decorator

    # ==========================================
    # 重载 [] 符号
    # 作用：让你可以用 花名册["名字"] 取东西
    # ==========================================
    def __getitem__(self, name: str) -> T:
        # 如果名字不存在 → 报错
        if name not in self._registry:
            raise KeyError(f"不支持的{self._type}类型：{name}")

        # 从字典里取出对应的类/函数
        return self._registry[name]

    # ==========================================
    # 获取所有支持的名字列表
    # 例如：["sgd", "adam", "adamw"]
    # ==========================================
    def supported_names(self) -> List[str]:
        return list(self._registry.keys())

    # ==========================================
    # 检查名字是否合法
    # 不支持就直接报错，防止传错参数
    # ==========================================
    def assert_supported(self, names: str | Iterable[str]) -> None:
        # 如果传入的是字符串，转成列表，统一处理
        if isinstance(names, str):
            names = [names]

        # 遍历检查每一个名字
        for name in names:
            if name not in self._registry:
                from argparse import ArgumentTypeError
                # 报错：不支持 xxx，支持的列表是 xxx
                raise ArgumentTypeError(
                    f"不支持的{self._type}：{name}。\n"
                    f"支持的选项：{self.supported_names()}"
                )