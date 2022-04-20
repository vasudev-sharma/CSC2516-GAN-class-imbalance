from setuptools import setup, find_packages

setup(
    name="APA",
    version="0.1.0",
    description="Adaptive Pseudo Augmentation for GAN Training with Limited Data",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "torch",
    ],
    include_package_data=True,
)