# FluentAI: Learn languages in a flash

![FluentAI Banner](img/banner-withbg.jpg)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Supported versions">
  <img src="https://img.shields.io/github/license/StephanAkkerman/FluentAI.svg?color=brightgreen" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

---

> [!CAUTION]
> This project is currently under development, please see the issues to see everything that still needs to be done before this is ready to use.

## Introduction

FluentAI is inspired by the method detailed in the paper [SmartPhone: Exploring Keyword Mnemonic with Auto-generated Verbal and Visual Cues by Jaewook Lee and Andrew Lan](https://arxiv.org/pdf/2305.10436.pdf). The aim is to recreate their approach using accessible, open-source models.
The pipeline they propose, as shown below, serves as the blueprint for our project. It illustrates the process of automating language learning, blending cutting-edge AI techniques with proven language learning methodology.

![image](https://github.com/StephanAkkerman/FluentAI/assets/45365128/c9ca3190-b136-453d-91cd-f785eac11fa3)

You can find our docs [here](https://github.com/StephanAkkerman/FluentAI/wiki) and the list of supported languages [here](https://github.com/StephanAkkerman/FluentAI/wiki/Supported-languages).

## Table of Contents 🗂

- [Prerequisites](#prerequisites-)
- [Installation](#installation-)
    - [Using `pip`](#using-pip)
    - [Building from Source](#building-from-source)
    - [GPU Support](#gpu-support)
- [Usage](#usage-)
- [Citation](#citation-)
- [Contributing](#contributing-)
- [License](#license-)



## Prerequisites 📋

Before starting, make sure you have the following requirements:

- [Anki](https://apps.ankiweb.net/) installed on your device.
- [Anki-Connect](https://foosoft.net/projects/anki-connect/) This add-on allows you to add cards to Anki from the command line.
- [Python](https://www.python.org/downloads/) 3.10 installed on your device.
- [React](https://react.dev) installed on your device (optional).

## Installation ⚙️

### Using `pip`

We have bundled all required dependencies into a package for easy installation. To get started, simply run one of the following commands:

```bash
pip install .
```

or install directly from the repository:

```bash
pip install git+https://github.com/StephanAkkerman/FluentAI.git
```

### Building from Source

If you prefer to build from source, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/StephanAkkerman/FluentAI.git
   ```

2. Navigate to the project directory:

   ```bash
   cd FluentAI
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Install with GPU Support (Recommended)

If you would like to run the code on a GPU, you can install the `torch` package with CUDA support.
After installing the required dependencies, run the following command:

```bash
pip install -r requirements/gpu-requirements.txt
```

## Usage ⌨️

Using FluentAI is simple, after following the steps in [pre-requisites](#pre-requisites) you can start the back-end by executing the following command.
```bash
python fluentai/services/card_gen/api.py
``` 

### Running the Front-End (Optional)

If you would like to build the front-end yourself you can do so by executing these commands.

1. Navigate to the `frontend` directory:

   ```bash
   cd fluentai/frontend
   ```

2. Install the necessary frontend dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm start
   ```

This will launch the front-end and connect it to the API for a seamless user experience.

## Citation ✍️

If you use this project in your research, please cite as follows:

```bibtex
@misc{FluentAI,
  author  = {Stephan Akkerman, Winston Lam, Tim Koornstra},
  title   = {FluentAI},
  year    = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/FluentAI}}
}
```

## Contributing 🛠

Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.
If you would like to make code contributions yourself, please read [CONTRIBUTING.MD](CONTRIBUTING.md).\
![https://github.com/StephanAkkerman/FluentAI/graphs/contributors](https://contributors-img.firebaseapp.com/image?repo=StephanAkkerman/FluentAI)

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
