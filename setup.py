import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="davils",
    version="1.0.0",
    author="Davide Raviolo",
    description="Davide's Utilities for Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DavideRepo/davils",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'h5py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)