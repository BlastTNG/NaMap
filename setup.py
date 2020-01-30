from setuptools import setup

setup(
    name='namap',
    version='1.0',
    description='NaMap: a naive mapmaker for the BLASTPol and BLAST-TNG experiments',
    author='Gabriele Coppi',
    author_email='gcoppi@sas.upenn.edu',
    packages=['namap'], 
    install_requires=['numpy', 'scipy', 'astropy', 'photutils'], #external packages as dependencies
)