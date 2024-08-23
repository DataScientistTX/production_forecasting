from setuptools import find_packages, setup

setup(
    name='oil_production_analysis',
    packages=find_packages(),
    version='0.1.0',
    description='A project for analyzing and forecasting oil production',
    author='Your Name',
    license='MIT',
    install_requires=[
        'pandas>=2.2.1',
        'numpy>=1.26.4',
        'matplotlib>=3.8.3',
        'seaborn>=0.13.2',
        'scikit-learn>=1.4.1',
        'plotly>=5.20.0',
        'torch>=2.2.2',
        'chronos @ git+https://github.com/amazon-science/chronos-forecasting.git@96cedec3fa9795c9bd58650080643e2b68bd3a6e'
    ],
    python_requires='>=3.8',
)