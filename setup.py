from setuptools import find_packages, setup

setup(
    name='polus',
    packages=find_packages(include=['poluslib']),
    version='0.1.0',
    description='A tensorflow based framework for training deep learning models',
    author='Tiago Almeida and Rui Antunes',
    author_email='tiagomeloalmeida@ua.pt',
    license='MIT',
    install_requires=["tensorflow>=2.6.0","transformers","tensorflow-addons","wandb"],
    test_suite='tests',
)