# ChiNet
NLU project 2 - Story Cloze Task

Source paper: [Conditional Generative Adversarial Networks for Commonsense Machine Comprehension (Chinese et al, 2017)](https://www.ijcai.org/proceedings/2017/0576.pdf)

## Tasks
[Put your name next to tasks you are currently working on and remove tasks once you have pushed to repo]

- **Joseph**:Implement word preprocessing
>Except maybe \<unk\>, no tokens should be necessary. All sentences should be processed such that strings representing words are replaced by their index in the vocabulary. Use vocabulary size 20k?

- **Xander** Fix word2vec embedding loading
>Currently gan.py takes an embedding weight matrix in the same way it takes a data batch, a better solution is likely possible.

- Test and debug discriminator

- **Xander** Implement attention

- **Xander** Implement generator

- Train model (DIFFICULT!)

- Implement result writer

- Write report

## Deadlines

Friday 25: Implement disciminator

Friday 1: Implement attention and generator

Tuesday 5: Have model trained and results ready

Thursday 7: Finish report

Friday 8: Hand in project




## Structure

* `datasets/` - all data sources required for training/validation/testing.
* `outputs/` - any output for a model will be placed here, including logs, summaries, checkpoints, and Kaggle submission `.csv` files.
* `src/` - all source code.
    * `core/` - base classes
    * `datasources/` - routines for reading and preprocessing entries for training and testing
    * `models/` - neural network definitions
    * `util/` - utility methods
    * `main.py` - training script

## Creating your own model
### Model definition
To create your own neural network, do the following:
1. Make a copy of `src/models/example.py`. For the purpose of this documentation, let's call the new file `newmodel.py` and the class within `NewModel`.
2. Now edit `src/models/__init__.py` and insert the new model by making it look like:
```
from .example import ExampleNet
from .newmodel import NewModel
__all__ = ('ExampleNet', 'NewModel')
```
3. Lastly, make a copy or edit `src/main.py` such that it imports and uses class `NewModel` instead of `ExampleNet`.

### Training the model
If your training script is called `main.py`, simply `cd` into the `src/` directory and run
```
python3 main.py
```

_[The skeleton of this project has been done by Seonwook Park and has been adapted by Nil Adell for this project]_
