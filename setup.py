from setuptools import setup, find_packages

# TODO: add cudatoolkit 
# conda install -c anaconda cudatoolkit

setup(
    name="lfadslite",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "future",
        "tensorflow-gpu==1.15.2",
        "tensorflow-probability>=0.7.0",
        "tensorboard>=1.14.0",
        "numpy>=1.16.6",
        "pandas>=0.24.2",
        "matplotlib>=2.2.4",
        "scikit-learn>=0.20.3",
        "h5py>=2.9.0",
        "gast==0.2.2"
    ],
    environment_editor="Lahiru Wimalasena",
    description="Set up evnironment to run lfads_tf1 in python 3.6.9 with Tensorflow 1.15.2",
)
