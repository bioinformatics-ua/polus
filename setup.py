from setuptools import find_packages, setup

setup(
    name='polus',
    packages=find_packages(include=['polus*']),
    version='0.1.5',
    description='A tensorflow based framework for training deep learning models',
    author='Tiago Almeida and Rui Antunes',
    author_email='tiagomeloalmeida@ua.pt',
    license='MIT',
    install_requires=["tensorflow>=2.6.0","transformers","tensorflow-addons","wandb","torch"],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)