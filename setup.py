# -*- coding: utf-8 -*-


from setuptools import setup


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='GpGAN',
    version='1.0',
    description='Graphical perception based GAN',
    long_description=readme,
    author='Daniel Haehn, Aswin Vasudevan',
    author_email='aswinv@bu.edu',
    url='',
    license=license
)
