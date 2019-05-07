'''
Created on 08.02.2016

@author: tklingenberg
'''

from setuptools import setup, find_packages

setup(
    # Application name:
    name='memobuilder',

    # Version number (initial):
    version='0.2.0',

    # Short desription of the project:
    description='Environment for the development of simulation metamodels',

    # Application author details:
    author='Thole Klingenberg',
    autor_email='thole.klingenberg@offis.de',

    # Packages
    packages=find_packages(),

    include_package_data=True,

    classifiers=[
        'Private :: Do Not Upload',
    ],

    install_requires=[
        'mosaik',
        'mosaik-api>=2.0',
        'pydoe',
        'matplotlib',
        'sklearn',
        'scikit-neuralnetwork',
        'pandas',
        'sphinx',
        'sphinxcontrib-plantuml',
        'sphinx_rtd_theme',
    ],
)
