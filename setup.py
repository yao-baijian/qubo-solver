"""Setup for qubo-solver — standalone solver library for QUBO problems."""

from setuptools import setup, find_packages

setup(
    name="qubo-solver",
    version="0.1.0",
    description="QUBO solvers: FEM (mean-field), SBM (Simulated Bifurcation), QIS3, DIGCIM",
    author="Yao Baijian",
    author_email="yao-baijian@users.noreply.github.com",
    url="https://github.com/yao-baijian/qubo-solver",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "torch>=2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
