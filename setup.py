from setuptools import setup, find_packages

__version__ = "0.0.1"


install_requires = [
    'astropy',
    'cython',
    'numpy',
    'blimpy',
]

scripts = [
    'scripts/find_candidates.py',
    'scripts/plot_candidates.py',
    'scripts/seti_event.py',
    'scripts/taylor_tree.pyx',
]

setup(
    name="turbo_seti",
    version=__version__,
    packages=find_packages(),
    scripts=scripts,
    install_requires=install_requires,
    author="Emilio Enriquez",
    author_email="e.enriquez@berkeley.edu",
    description="Analysis tool for the search of narrow band drifting signals in filterbank data",
    license="PLEASECHANGE",
    keywords="astronomy",
    url="https://github.com/UCBerkeleySETI/turbo_seti",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering",
        ]
)
