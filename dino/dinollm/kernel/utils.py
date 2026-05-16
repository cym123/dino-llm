from __future__ import annotations

# 处理文件路径，找cuda/c++源码文件夹
import pathlib
# 导入类型注解工具：类型别名、元组、列表、命名元组、联合类型
from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias, Union

# 仅在代码编辑器/类型检查时生效，运行时不导入
if TYPE_CHECKING:
    from tvm_ffi import Module  # 编译后的C++/CUDA模块对象

# --------------------------
# 1 固定路径配置：定位底层cuda/c++源码目录
# --------------------------
# __file__ 当前这个utils.py文件
# .parent 当前文件所在文件夹
# 再拼上 csrc 子文件夹 → 所有内核代码都放在这里
KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"

# 默认头文件搜索路径：编译器找 .h 头文件时会去这里找
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]

# 默认C++编译参数
# -std=c++20 用C++20标准
# -O3 最高级别代码优化
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]

# 默认CUDA编译参数
# --expt-relaxed-constexpr 放开CUDA constexpr语法限制，方便写模板内核
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]

# 默认链接器参数，目前为空
DEFAULT_LDFLAGS = []

# 定义类型别名：C++模板只支持这三种类型
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]


# --------------------------
# 2 自定义字符串列表类：专门用来拼接C++模板参数
# 继承原生list，只是多了转字符串的功能
# --------------------------
class CppArgList(list[str]):
    # 把列表里的参数拼成 "a,b,c" 这种C++模板格式
    def __str__(self) -> str:
        return ", ".join(self)


# --------------------------
# 3 内核配置结构体：描述CUDA核的硬件配置
# NamedTuple 只读、轻量、存配置参数
# --------------------------
class KernelConfig(NamedTuple):
    num_threads: int    # 每个block开多少线程，常见128/256
    max_occupancy: int  # 控制GPU多流水并发数，调性能用
    use_pdl: bool       # 是否开启PDL（新一代NVIDIA显卡硬件加速指令）

    # 把配置自动转成C++模板参数字符串
    # 例如：128,1,false
    @property
    def template_args(self) -> str:
        # bool转小写字符串true/false，给C++模板用
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


# --------------------------
# 4 内部工具函数：生成模块唯一名称
# 防止编译出来的模块重名冲突
# 统一前缀 dinollm__ 再拼接参数
# --------------------------
def _make_name(*args: str) -> str:
    return "dinollm__" + "_".join(str(arg) for arg in args)


# --------------------------
# 5 内部工具函数：生成FFI包装代码
# 作用：把C++/CUDA函数暴露给Python调用
# tvm_ffi 固定语法，导出函数符号
# tup 是二元组：(对外函数名, 内核真实函数名)
# --------------------------
def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    # 生成一行固定导出宏，让Python能找到C++函数入口
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


# --------------------------
# 6 类型转换工具：Python值 → C++模板字符串
# 给load_jit/load_aot拼模板参数用
# --------------------------
def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    # 内部小函数：单个参数转C++能识别的字符串
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        # 布尔值 → 小写true/false
        if isinstance(arg, bool):
            return "true" if arg else "false"
        # 数字直接转字符串
        if isinstance(arg, (int, float)):
            return str(arg)
        # 不支持的类型直接抛错
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")
    # 批量转换，返回自定义的CppArgList
    return CppArgList(_convert(arg) for arg in args)


# =====================================================================
# 7 核心函数一：load_aot 加载【提前编译好】的C++/CUDA模块
# AOT = Ahead Of Time 提前编译，启动直接加载，不现场编译
# 适用：pynccl、radix、test_tensor 这种通用底层模块
# =====================================================================
def load_aot(
    *args: str,                                   # 用来生成唯一模块名的自定义标识
    cpp_files: List[str] | None = None,            # 要编译的cpp文件名列表
    cuda_files: List[str] | None = None,           # 要编译的cu文件名列表
    extra_cflags: List[str] | None = None,         # 额外C++编译参数
    extra_cuda_cflags: List[str] | None = None,    # 额外CUDA编译参数
    extra_ldflags: List[str] | None = None,       # 额外链接参数
    extra_include_paths: List[str] | None = None,  # 额外头文件路径
    build_directory: str | None = None,            # 编译产物输出目录
) -> Module:
    # 导入tvm_ffi的加载接口，运行时才导入，避免循环依赖
    from tvm_ffi.cpp import load

    # 空值兜底，避免None报错
    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # 拼接完整绝对路径：csrc/src/下面的文件
    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    # 调用底层load，编译并返回可被Python调用的Module对象
    return load(
        _make_name(*args),                              # 唯一模块名
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        # 默认编译参数 + 额外参数合并
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )


# =====================================================================
# 8 核心函数二：load_jit 【运行时现场动态编译】CUDA/C++内核
# JIT = Just In Time 即时编译
# 适用：index.cu、store.cu 这种要根据参数模板实例化的内核
# 特点：同一份cu源码，编译出不同模板参数的版本，按需生成
# =====================================================================
def load_jit(
    *args: str,                                   # 生成唯一模块名标识
    cpp_files: List[str] | None = None,            # 要包含的cpp文件
    cuda_files: List[str] | None = None,           # 要包含的cu文件
    cpp_wrappers: List[Tuple[str, str]] | None = None,  # cpp函数导出包装
    cuda_wrappers: List[Tuple[str, str]] | None = None, # cuda函数导出包装
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    from tvm_ffi.cpp import load_inline

    # 空值兜底
    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # --------------------------
    # 处理cpp源码：拼成#include 字符串
    # 源码放在 csrc/jit/ 下
    # --------------------------
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    # 追加FFI导出包装代码，把函数暴露给Python
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # --------------------------
    # 处理cuda源码：同上
    # --------------------------
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    # 动态内联源码、现场编译、返回可调用模块
    return load_inline(
        _make_name(*args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )
