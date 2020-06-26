from setuptools import setup, find_packages

# TODO: add cudatoolkit 
# conda install -c anaconda cudatoolkit

setup(
    name="lfadslite",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "future",
        "tensorflow-gpu==1.14.0",
        "tensorflow-probability==0.7.0",
        "tensorboard==1.14.0",
        "numpy==1.16.6",
        "pandas==0.24.2",
        "matplotlib==2.2.4",
        "scikit-learn==0.20.3",
        "h5py==2.9.0",
        "pymongo==3.8.0",
    ],
    environment_editor="Feng Zhu",
    description="Set up evnironment to run LFADSLITE",
)
