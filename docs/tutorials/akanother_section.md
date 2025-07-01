# Command-line interface (CLI)
**MintFlow CLI is under construction.**

The python API (i.e. the API used in the tutorial notebooks) is not suitable if the dataset contains a huge number of tissue sections and cells.
As an example, the python api may keep the predictions for different tissue sections on memory,
while for huge datasets the evaluation has to be done section by section, writing the prediction for each section on disk, removing it from memory and moving on to the next tissue section.
Or as another example, the python api by default loads all tissue sections into memory, while one may one to perform the analysis post-training and on a single tissue section and with less memory and computation requirements.
MintFlow's command-line interface (CLI) is implemeneted for such use cases for huge datasets.

