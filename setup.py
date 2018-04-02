from setuptools import setup, find_packages

__version__ = "0.7.1"


install_requires = [
    'astropy',
    'cython',
    'numpy',
    'blimpy',
]

entry_points = {
    'console_scripts' :
        ['turboSETI = turbo_seti.findoppler.seti_event:main',
         'find_candidates = turbo_seti.findoppler.find_candidates:main',
         'plot_candidates = turbo_seti.findoppler.plot_candidates:main',
     ]
}

setup(
    name="turbo_seti",
    version=__version__,
    packages=find_packages(),
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
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering",
        ]
)
