import os
from setuptools import find_packages, setup #type: ignore
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(filepath:str)->List[str]:
    with open(filepath, 'r') as file:
        requirements = file.readlines()
        reqs = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in reqs:
            reqs.remove(HYPHEN_E_DOT)
    return reqs


setup(
    name='Recommendation System: Movies',
    version='0.0.1',
    author='Akeem I. Lagundoye',
    author_email='akeemifedayolag@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
