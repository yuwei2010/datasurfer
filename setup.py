#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as f:
    requirements = [] #f.read().splitlines()L


test_requirements = ['pytest>=3', ]

setup(
    author="Wei Yu",
    author_email='wei.yu2@de.bosch.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A Python package for data processing",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='datasurfer',
    name='datasurfer',
    packages=find_packages(include=['datasurfer', 'datasurfer.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/yuw1si/datasurfer',
    version='1.0.7',
    zip_safe=False,
)
