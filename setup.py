import re

from setuptools import find_packages
from setuptools import setup


def get_version():
    filename = "unet/__init__.py"
    with open(filename) as f:
        match = re.search(
            r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0]
    return version


def get_install_requires():
    return [
        "focal-loss",
        "matplotlib",
        "mlflow",
        "tensorflow==2.9.1",
        "typeguard",
    ]


def get_long_description():
    return "This should contain a long description"


def main():
    version = get_version()

    setup(
        name="oct_unet",
        version=version,
        packages=find_packages(),
        description="U-net model",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="BioTeam, Inc.",
        author_email="bruno@bioteam.net",
        url="https://www.bioteam.net",
        install_requires=get_install_requires(),
        license="GPLv3",
        keywords="Image Segmentation, Machine Learning",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Researchers",
            "Natural Language :: English",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.10",
        ],
        package_data={},
        entry_points={},
        data_files=[],
    )


if __name__ == "__main__":
    main()
