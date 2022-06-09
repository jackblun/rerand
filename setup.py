# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open("README.MD") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="rerand",
    version="0.1.0",
    description="Tools for rerandomisation in randomised experiments.",
    long_description=readme,
    author="Jackblundell",
    author_email="jackblun@gmail.com",
    url="https://github.com/jackblun/rerand",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
)
