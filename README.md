# ML_Project_Upgrade

This project is an improvement of a previous project done for the Machine Learning course of University of Pisa (Computer Science Master Degree). The focus of the upgrade is on improving the **Multi-Layer Perceptron (MLP)** architecture, which was the weakest part of the original project.

The project leverages the **Eigen** library for efficient linear algebra operations, which is crucial for the performance and training of the MLP.

## Requirements

Before running the project, make sure you have the following dependencies installed:

1. **Eigen Library**: The project uses the Eigen library for linear algebra operations. Eigen is header-only, meaning it does not need to be compiled separately. Simply download it from the official [Eigen repository](https://eigen.tuxfamily.org/dox/), or install it via your system's package manager:
     ```bash
     sudo apt-get install libeigen3-dev
     ```

2. **Make**: The project uses Makefiles to compile the code. Make sure `make` is installed on your system:
     ```bash
     sudo apt-get install build-essential
     ```

## Building the Project

Follow these steps to build the project:

1. **Clone the Repository**:
    If you haven't already cloned the repository, run:
    ```bash
    git clone https://github.com/JacopoRaffi/ML_Project_Upgrade.git
    ```

2. **Compile with make**
    Navigate into the project directory:
    ```bash
    cd ML_Project_Upgrade
    ```

    Execute the make command:
    ```bash
    make
    ```
