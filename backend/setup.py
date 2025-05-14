from setuptools import find_packages, setup


def parse_requirements(filename: str) -> list[str]:
    """
    Parse a requirements file into a list of dependencies.

    Parameters
    ----------
    filename : str
        The path to the requirements file.

    Returns
    -------
    list[str]
        A list of dependencies.
    """
    with open(filename, encoding="utf-8") as file:
        lines = file.readlines()

    requirements = []
    for line in lines:
        # Strip whitespace and ignore comments or empty lines
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

    return requirements


# Read dependencies from requirements.txt
requirements = parse_requirements("requirements.txt")

setup(
    name="mnemorai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    description="mnemorai short description",
    url="https://github.com/StephanAkkerman/mnemorai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "mnemorai-main=mnemorai.main:main",
        ],
    },
    python_requires=">=3.9,<3.13",
)
