#!/usr/bin/env python

__author__ = "Anubhav Jain"
__copyright__ = "Copyright 2013, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Anubhav Jain"
__email__ = "ajain@lbl.gov"
__date__ = "Mar 15, 2013"

import os, sys

from mpworks import __version__
from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        readme_text = open(os.path.join(module_dir, 'README.rst'), encoding="UTF-16").read()
    else:
        readme_text = open(os.path.join(module_dir, 'README.rst')).read()
    setup(name='MPWorks',
          version=__version__,
          description='Materials Project codes',
          long_description=readme_text,
          url='https://github.com/materialsproject/MPWorks',
          author='Anubhav Jain',
          author_email='anubhavster@gmail.com',
          license='modified BSD',
          packages=find_packages(),
          zip_safe=False,
          install_requires=["pymatgen>=4.0", "FireWorks>=0.9", "custodian>=0.7"],
          classifiers=["Programming Language :: Python :: 2.7", "Development Status :: 2 - Pre-Alpha",
                       "Intended Audience :: Science/Research", "Intended Audience :: System Administrators",
                       "Intended Audience :: Information Technology",
                       "Operating System :: OS Independent", "Topic :: Other/Nonlisted Topic",
                       "Topic :: Scientific/Engineering"],
          test_suite='nose.collector',
          tests_require=['nose'],
          scripts=[os.path.join(os.path.join(module_dir, "scripts", f)) for f in
                   os.listdir(os.path.join(module_dir, "scripts"))])
