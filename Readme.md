# The repo has 3 python scripts:

 -  models.py defines the LSTM models used. TorchLSTM uses PyTorch's built in LSTM module, while
    CustomLSTM builds one from scratch.
 -  training.py defines functions for training and evaluating DNNs. Since training will mostly not
    be needed, this can be ignored. 
 -  utils.py defines some utilities, specifically for getting the character set from a dataset, and
    for converting a trained TorchLSTM into an equivalent CustomLSTM model. Again, not really
    needed.

# The Dataset:

The data is a bunch of text, stored at `data/input.txt`. Note that this is directly fed into the nn
without any preprocessing like removing newlines etc. I have only used the first ~20KB of this data
for training. `data\meta_20000.txt` is a file which contains a list of characters appearing, and
fixes a map from the characters to indices of the input and output vectors. This file is an `str`
representation of a python `dict`, and can be parsed by just using python `eval`.

# The Models:

The weights and biases of the models are stored as `models\*.pth` file. The last part of the
filename gives the model class used (`torch` for `TorchLSTM` and `custom` for `CustomLSTM`) and the
hyperparameters passed during construction, seperated by `_`. So, for example, if the model class
was created with `TorchLSTM(500, 3)`, the filename will be of form `*_torch_500_3.pth`.

To extract the weights and biases from the file, first create the corresponding model object (for eg
by calling `TorchLSTM(500, 3)` and then use the `load_model()` method of the object with the `.pth`
file path. This will give you a pytorch model with the weights and biases loaded from the file. Now,
the weight matrices and bias vectors can be extracted by accessing the parameters within the model.

The model classes use submodules (the `TorchLSTM` has an `LSTM`, and the `CustomLSTM` uses a bunch of
`Linear` modules). These are where the weights and biases will be stored. So, for example, to access
the weights and biases used to calculate `i` inside the `CustomLSTM`, use `i_gate.weight` and
`i_gate.bias`. Once `load_model()` is called, these parameters of these submodules will get
populated with the correct weights and biases.
