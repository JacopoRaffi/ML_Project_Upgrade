# MLP Training Example with Loss Curve Plotting

This example demonstrates how to perform a simple training of a Multi-Layer Perceptron (MLP) model and visualize the training loss curves using gnuplot.

## Requirements

Before running the example, make sure you meet the following requirements:

1. **Object Files Compiled**: 
   You must have already compiled the object files as mentioned in the main README. These files are required for the proper functioning of the example. Please follow the compilation instructions in the main README if you haven't done so yet.

2. **gnuplot**:
   You will need **gnuplot** installed to visualize the loss curves. gnuplot will plot the training loss during the MLP model's training process. You can install gnuplot using:
     ```bash
     sudo apt-get install gnuplot
     ```

## Running the Example

### Step 1: Navigate to the Example Directory

Open a terminal and change to the directory where the example code is located. You can do this by running:

```bash
cd path/to/ML_Project_Upgrade/examples
```

### Step 2: Execute the script
```bash
./script_training.sh
```

If everything is correct, a plot will be shown displaying both the training loss and test loss over time.
