from setuptools import find_packages, setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
import polus
install_reqs = parse_requirements("requirements.txt")

setup(
    name='polus',
    packages=find_packages(include=['polus*']),
    version=polus.__version__,
    description='A tensorflow based framework for training deep learning models',
    author='Tiago Almeida and Rui Antunes',
    author_email='tiagomeloalmeida@ua.pt',
    license='MIT',
    install_requires=[str(ir.req) for ir in install_reqs],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)