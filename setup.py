
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
        "click>=7.0.0",
    ],
    extras_require={
        "test": {
            "flake8",
            "pep8-naming",
            "pytest"
        },
    },
    scripts=["daq_infer"]
)
