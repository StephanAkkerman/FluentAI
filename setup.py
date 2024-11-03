from setuptools import find_packages, setup


def parse_requirements(filename):
    """
    Parse a requirements file into a list of dependencies.

    Args:
        filename (str): Path to the requirements file.

    Returns:
        list: A list of dependency strings.
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    requirements = []
    for line in lines:
        # Strip whitespace and ignore comments or empty lines
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

    return requirements


# Read dependencies from requirements.txt
requirements = parse_requirements("requirements/requirements.txt")

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fluentai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    description="FluentAI short description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StephanAkkerman/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "fluentai-main=fluentai.main:main",  # Adjust as needed
        ],
    },
    python_requires=">=3.10",
    extras_require={
        "dev": parse_requirements("requirements/dev-requirements.txt"),
    },
)
