#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import glob
version = None
package_name = "mlmisc"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open('{}/version.py'.format(package_name)) as version_file:
    exec(version_file.read())

requirements = [
    'PyYaml',
    "imgaug",
    "pyconfigmanager",
]
dependency_links = [
    """git+https://github.com/miacro/{}.git@master#egg={}-9999""".format(
        "pyconfigmanager", "pyconfigmanager")
]
dependency_links = []


def get_scripts():
    result = []
    for item in glob.glob("{}/bin/*".format(package_name)):
        if os.access(item, os.X_OK) and os.path.isfile(item):
            result.append(item)
    return result


long_description = ''
setup(
    name=package_name,
    version=version,
    description=package_name,
    long_description=long_description,
    url='',
    author='miacro',
    author_email='fqguozhou@gmail.com',
    maintainer='miacro',
    maintainer_email='fqguozhou@gmail.com',
    packages=find_packages(exclude=['test.*', 'test', 'tests.*', 'tests']),
    install_requires=requirements,
    classifiers=[
        'Environment :: Other Environment',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
    ],
    scripts=get_scripts(),
    ext_modules=[],
    package_data={
        package_name: ["*.yaml", "*.json", "**/*.yaml", "**/*.json"]
    },
    exclude_package_data={package_name: ["test*", "tests*"]},
    dependency_links=dependency_links,
)
