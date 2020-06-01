
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="avair.ai",
    version="0.0.1",
    description="Intelligent birdcam",
    author="Jori van Lier",
    long_description=long_description,
    author_email="jori@jvlanalytics.nl",
    packages=["aviar"],
    install_requires=[
        "numpy>=1.18.0",
        "matplotlib>=3.2.0",
        "torch==1.4.0",
        "torchvision==0.5.0",
        "fastai==1.0.61"
    ],
    extras_require={
        "test": {
            "flake8",
            "pep8-naming",
            "pytest"
        },
    },
    scripts=["daq_infer.py"]
)
