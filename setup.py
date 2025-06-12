from setuptools import setup, find_packages

setup(
    name="my_deeprl_network",
    version="0.1",
    packages=find_packages(include=["agents", "agents.*"]),
)
