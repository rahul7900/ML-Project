from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''this function will return the list of requirements'''
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        ''' Addition of -e . whill connect the setup.py file to requirements but 
        we dont need to read it over here'''

    return requirements



setup (
name = 'ML Project',
version = '0.0.1',
author = 'Rahul',
author_email = 'rahul.singh.7920000@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')
)