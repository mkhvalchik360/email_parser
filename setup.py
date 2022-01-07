from setuptools import find_packages, setup
from glob import glob
import os


setup(name='email_parser',
      packages=find_packages(include=['email_parser']),
      version='0.0.1',
      description='Email parser',
      author='JB Polle',
      license='MIT',
      install_requires=['langid==1.1.6',
                        'numpy>=1.19.5',
                        'pandas>=1.2.3',
                        'regex',
                        'scikit-learn==0.24.1',
                        'sentence-transformers==1.0.4',
                        'tensorflow==2.6.0',
                        'tensorflow-hub>=0.12.0',
                        'tensorflow-text==2.6.0',
                        'tokenizers==0.10.1',
                        'torch>=1.8.0',
                        'umap-learn==0.5.1',
                        'dateparser==1.0.0',
                        'transformers>=4.3',
                        'gradio>=2.7'])
