# NLU project 2 - Story Cloze Task
## Group 9: ChiNet

### Getting Started
These instructions will get you a copy of the project up and running on your local machine for testing purposes.

#### Environment Setup

In order to reproduce our environment with all the packages and versions required:
1. Download the file spec-file.txt from polybox [ https://polybox.ethz.ch/index.php/s/m0xJnbNl5iFEPCk]
2. Run the following command from the terminal or an Anaconda Prompt:
```
conda create --name myenv --file spec-file.txt
```
This will create a conda environment and install all packages needed.

3. Activate the environment:
On Windows, in your Anaconda Prompt, run: 
```
activate myenv
```
On macOS and Linux, in your Terminal Window, run:
```
source activate myenv
```
4. Open a python shell and write the following commands:
```
import nltk
nltk.download('punkt')
nltk.download('names')
```
#### Code Files Setup

1. The unzipped folder 'ChiNet' contains src folder where the code main.py is

#### Data Files Setup

All data files are available in a zipped folder on polybox: 

1. Download the zipped folder 'datasets' [https://polybox.ethz.ch/index.php/s/i2az9ljuZt3eGx5] and unzip it.
This folder contains:
* The vocab folder created after preprossing.
* The word2vec embedding folder
* The training, validation and test sets
2. Download the zipped folder 'outputs' [https://polybox.ethz.ch/index.php/s/3FOMzOGPZdoAU9k] and unzip it
This folder contains model weights after training.
3. Move both unzipped folders to the folder 'ChiNet' 

#### Running the Code

Run test_main.py available in ChiNet\src





