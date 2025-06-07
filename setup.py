
from setuptools import setup, find_packages

setup(
    name="mintflow",
    version="0.0.1",
    packages=find_packages(),

    ## Include data files
    package_data={
        "mypackage": ["config.ini", "data/*.txt"],
    }
    ## Metadata
    # author={name = "Amir Hossein Hosseini Akbarnejad"},
    # {name = "Sebastian Birk"},
    # author_email="your.email@example.com",
    # description="A simple Python package with additional files",

)
