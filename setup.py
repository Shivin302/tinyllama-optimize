from setuptools import setup, find_packages

setup(
    name="tinyllama",
    version="0.1",
    packages=find_packages(include=['tinyllama']),
    package_dir={'': '.'},  # This tells setuptools to look for packages in the root directory
    include_package_data=True,  # This includes non-Python files specified in MANIFEST.in
)