import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="audioperm",
    version="0.0.5",
    description="Audioperm, a python library for generating different permutations of audible segments from audio files.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/zabir-nabil/audioperm",
    author="Zabir Al Nazi",
    author_email="zabiralnazi@yahoo.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["feedparser", "html2text", "numpy", "librosa>=0.8.1", "pydub", "PySoundFile"],
)
