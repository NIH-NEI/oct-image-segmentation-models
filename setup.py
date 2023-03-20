import re

from setuptools import find_packages
from setuptools import setup


def get_version():
    filename = "oct_image_segmentation_models/__init__.py"
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
        "focal-loss==0.0.7",
        "matplotlib==3.5.2",
        "mlflow==2.0.1",
        "scikit-image==0.19.3",
        "scikit-learn==1.1.3",
        "tensorflow==2.9.0",
        "typeguard==2.13.3",
        "Surface-Distance-Based-Measures==0.1",
    ]


def get_long_description():
    return "This should contain a long description"


def main():
    version = get_version()

    setup(
        name="oct_image_segmentation_models",
        version=version,
        packages=find_packages(),
        description="OCT Image Segmentation Models",
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
