from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

setup(
    name="cpools",
    ext_modules=[
        CppExtension(
            "top_pool",
            ["src/top_pool.cpp"],
            extra_link_args=[f"-Wl,-rpath,{torch_lib_path}"],
        ),
        CppExtension(
            "bottom_pool",
            ["src/bottom_pool.cpp"],
            extra_link_args=[f"-Wl,-rpath,{torch_lib_path}"],
        ),
        CppExtension(
            "left_pool",
            ["src/left_pool.cpp"],
            extra_link_args=[f"-Wl,-rpath,{torch_lib_path}"],
        ),
        CppExtension(
            "right_pool",
            ["src/right_pool.cpp"],
            extra_link_args=[f"-Wl,-rpath,{torch_lib_path}"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
