# To increment version
# Check you have ~/.pypirc filled in
# git tag x.y.z
# git push && git push --tags
# rm -rf dist; python setup.py sdist bdist_wheel
# auditwheel repair dist/*.whl -w dist/ (Linux)
# TEST: twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*

from setuptools import setup, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
from setuptools.extension import Extension

__version__ = "1.2.2"

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

extensions = [Extension(
        name="turbo_seti.find_doppler.taylor_tree",
        sources=["turbo_seti/find_doppler/taylor_tree.pyx"],
        include_dirs=[numpy.get_include()],
        )
    ]
cmdclass = {'build_ext': build_ext}


# Need to copy over index files, generate filenames
idxs = [2,3,4,5,6,7,8,9,10,11]
drift_idxs = ['drift_indexes/drift_indexes_array_%i.txt' % ii for ii in idxs]
package_data={
    'turbo_seti': drift_idxs,
}

setup(
    name="turbo_seti",
    version=__version__,
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    cmdclass=cmdclass,
    ext_modules=cythonize(extensions),
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
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        ]
)
