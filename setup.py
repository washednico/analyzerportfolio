from setuptools import setup, find_packages

setup(
    name='portfolioanalyzer',
    version='0.1.0',
    author='Tobiols Hedge Fund',
    author_email='xxxxx',
    description='A Python package for stock portfolio analysis and optimization.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='xxxxxxxxxx',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'yfinance',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'xxxxxxxxxx',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)