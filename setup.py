# To increment version
# Check you have ~/.pypirc filled in
# git tag x.y.z
# git push && git push --tags
# rm -rf dist; python setup.py sdist bdist_wheel
# auditwheel repair dist/*.whl -w dist/ (Linux)
# TEST: twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*

from setuptools import setup, find_packages
import numpy
from setuptools.extension import Extension

__version__ = "2.0.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.readlines()

with open("requirements_test.txt", "r") as fh:
    test_requirements = fh.readlines()

entry_points = {
    'console_scripts' :
        ['turboSETI = turbo_seti.find_doppler.seti_event:main',
         'find_event = turbo_seti.find_event.find_event:main',
         'find_scan_sets = turbo_seti.find_event.find_scan_sets:main',
         'plot_event = turbo_seti.find_event.plot_event:main',
     ]
}

package_data={
    'turbo_seti': ['drift_indexes/*.txt', 'find_doppler/kernels/**/*.cu'],
}

setup(
    name="turbo_seti",
    version=__version__,
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    install_requires=install_requires,
    tests_require=test_requirements,
    entry_points=entry_points,
    author="Emilio Enriquez",
    author_email="e.enriquez@berkeley.edu",
    description="Analysis tool for the search of narrow band drifting signals in filterbank data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT License",
    keywords="astronomy",
    url="https://github.com/UCBerkeleySETI/turbo_seti",
    zip_safe=False,
    options={"bdist_wheel": {"universal": "1"}},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        ]
)
