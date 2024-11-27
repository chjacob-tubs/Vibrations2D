# Vibrations2D

Vibrations2D is a Python code for vibrational 2D spectroscopy.

## Requirements

Vibrations2D is an independent code that for running needs only Python standard
packages, extended with NumPy and SciPy.

## Installation

After clone or download the 2d-ir-spectroscopy repository:
 
Use the pip package manager:

    vibrations2d% pip install .

Or use the conda environment manager:

Installation of requirements:

     vibrations2d% conda env create -f environment.yml
     vibrations2d% conda activate Vib2DCondaENV
     (Vib2DCondaENV)vibrations2d% conda develop src/

Within the conda environment the pip installation can of course also be performed. 

Verify the installation with pytest (install pytest before): 

     vibrations2d/src/Vibrations2D/test% pytest -v

If all tests pass then the installation has been successful.

## Usage and Documentation

See `script` directory and the jupyterlab file `How_to_2DIR_spectra.ipynb` 
for some examples of typical runs.

For further documentation see the corresponding papers 
and the docstrings in the code.

Vibrations can be run using Python's interpreter, or interactively with
some of the interactive Python consoles.

### Any suggestions and improvements are welcome.
