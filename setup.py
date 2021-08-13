from setuptools import setup, find_packages

__version__ = "2.1.11"

with open("turbo_seti/find_doppler/turbo_seti_version.py", "w") as fh:
    fh.write("TURBO_SETI_VERSION = '{}'\n".format(__version__))

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.readlines()

with open("requirements_test.txt", "r") as fh:
    test_requirements = fh.readlines()

entry_points = {
    "console_scripts" : [
        "turboSETI = turbo_seti.find_doppler.seti_event:main",
        "plotSETI = turbo_seti.find_event.run_pipelines:main" ]
}

package_data = {"turbo_seti": ["drift_indexes/*.txt", "find_doppler/kernels/**/*.cu"]}

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
    long_description_content_type="text/markdown",
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
    ],
)
