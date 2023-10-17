import setuptools

setuptools.setup(
    name="davils",
    version="1.0.0",
    author="Davide Raviolo",
    description="Davide's Utilities for Python.",
    url="https://github.com/DavideRepo/davils",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)