from setuptools import setup, find_packages

setup(
    name='analyzerportfolio',
    version='0.1.92',
    author='Nicola Fochi, Leonardo Mario di Gennaro',
    author_email='portfolioanalyzer-devs@proton.me',
    description='A Python package for stock portfolio analysis and optimization.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    url = 'https://github.com/washednico/analyzerportfolio',
    packages=find_packages(include=['analyzerportfolio', 'analyzerportfolio.*']),
    install_requires=[
        'openai>=1.43.0',
        'pandas>=1.5.1',
        'yfinance>=0.2.32',
        'numpy==1.26.4',
        'plotly>=5.18.0',
        'arch>=7.0.0',
        'scipy==1.14.0',
        "statsmodels==0.14.1",
        "nbformat>=4.2.0"
    ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.6',
)