from setuptools import setup, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
from setuptools.extension import Extension

__version__ = "1.0.0"

install_requires = [
    'astropy',
    'cython',
    'numpy',
    'blimpy',
    'pandas'
]

entry_points = {
    'console_scripts' :
        ['turboSETI = turbo_seti.findoppler.seti_event:main',
         'find_event = turbo_seti.findevent.find_event:main',
         'find_scan_sets = turbo_seti.findevent.find_scan_sets:main',
         'plot_event = turbo_seti.findevent.plot_event:main',
     ]
}

extensions = [Extension(
        name="turbo_seti.findoppler.taylor_tree",
        sources=["turbo_seti/findoppler/taylor_tree.pyx"],
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
    cmdclass=cmdclass,
    ext_modules = cythonize(extensions),
    install_requires=install_requires,
    entry_points=entry_points,
    author="Emilio Enriquez",
    author_email="e.enriquez@berkeley.edu",
    description="Analysis tool for the search of narrow band drifting signals in filterbank data",
    license="MIT License",
    keywords="astronomy",
    url="https://github.com/UCBerkeleySETI/turbo_seti",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        ]
)
