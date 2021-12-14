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

Each class in `models.py` specifies an implementation/variation of the LSTM. Each of these classes
follow some conventions, and these are specified below. New classes can be added to `models.py`, and
the training of these will work directly as long as the new classes also follow the conventions
given below.

## Inheritance:

Each class should inherit from `torch.nn.Module`. The constructor should call the `torch.nn.Module`
constructor before doing anything else.

## Constructor:

Every model class's constructor should follow one convention: the first argument should be
`char_size` and should correspond to the number of characters in the dataset. Then, the input and
output of the LSTM should be vectors of this dimension. Other than that, the class can have any
number of extra arguments with any type, depending on architecture and implementation details.

## Forward function:

The forward function should take in a sequence integer indices as input for the LSTM, and provides a
sequence of vectors as output. 

Each index in the input sequence correspond to a character, and the metadata file used during
training has a dict mapping characters to indices and vice-versa (see below). The input is passed as
tensor is of shape `(N,)` and type `int`, where `N` is the length of the sequence. 

The output is a sequence of vectors where each vector represents a probability distribution on the
characters. The higher the value at the index `i` of the vector, the higher is probability of
picking the character corresponding to index `i`. The output is packed into a `float` tensor of
shape `(N, k)`, where `N` is the length of the sequence and `k` is the number of characters. 

**Note** that the output is not expected to be a normalized probability distribution, and the
components of the vectors need not add up to `1`. The only requirement is that they be nonegative.
Softmax is applied on the output of the LSTM externally within training and when generating text.

## The Internal State and Gradient Information:

The LSTM class should carry vectors inside it for storing the internal state. However, due to
limitations in pytorch backprop, these vectors do not need to carry gradient information. For the
classes I have implemented, I use the `.detach()` method to drop the gradient information between
`forward()` calls. This also means that the backpropagation through time only occurs within the
sequence passed to the `forward()` function, and does not occur across two `forward()` calls.
Removing the `.detatch()` will cause an error to occur, and all workarounds slow down training
significantly.

## Saving and Loading Weights:

Each model class should have a `save_model()` function that takes a file path and saves the weights
there, and a `load_model()` function to copy the weights stored in a passed path into the weights of
the model.

## Conventions Used for the Given Weights Files:

The weights and biases of the models are stored as `models\*.pth` files. The last part of the
filename gives the model class used (`torch` for `TorchLSTM` and `custom` for `CustomLSTM`) and the
arguments passed during construction, separated by `_`. So, for example, if the model class
was created with `TorchLSTM(500, 3)`, the filename will be of form `*_torch_500_3.pth`.

## Extracting Weights:

To extract the weights and biases from the file, first create the corresponding model object (for eg
by calling `TorchLSTM(500, 3)` and then use the `load_model()` method of the object with the `.pth`
file path. This will give you a pytorch model with the weights and biases loaded from the file. Now,
the weight matrices and bias vectors can be extracted by accessing the parameters within the model.

Some of the model classes use submodules (the `TorchLSTM` has an `LSTM`, and the `CustomLSTM` uses a
bunch of `Linear` modules). These are where the weights and biases will be stored. So, for example,
to access the weights and biases used to calculate `i` inside the `CustomLSTM`, use `i_gate.weight`
and `i_gate.bias`. Once `load_model()` is called, these parameters of these submodules will get
populated with the correct weights and biases.

# Training:

The training is done via the `training.py` script. The script basically takes a model specified by
the user, and trains it according to the parameters specified by the user. The training algorithm is
standard Truncated Backprop through Time, with slowly decaying learning rate. The script takes many
command line arguments in standard unix-style fashion. A summary of these can be obtained by running
the script with `--help` as the only argument, and a detailed description is given below:

## Model Class and Arguments:

The script allows you to chose a model from `models.py` and initialize it via two command line
arguments:

 -  `--model-name` is the name of the model class to use. This can be any class from `model.py`,
including classes that you add in.

 -  `--model-args` are a comma separated list of arguments to be passed to the model class to initialize
it. These should match with the extra arguments that come after `char_size`. Each argument should be
a valid python expression. If the model class takes no arguments, just leave this as empty.

## Passing the Dataset:

There are two files that need to passed to obtain the training data, which are passed via different
command line arguments:

 -  `--data-set` is the actual dataset, which should just be a UTF-8 text file containing the corpus of
the training data used. The text from this file is used directly for training without any cleanup,
including spacing, indentation, newline etc. The one I have used for training is `input.txt`, and is
the text of a Shakespeare play.

 -  `--data-meta` is a file that contains some metadata used for processing the dataset file. This
includes a dict mapping characters to indices and vice versa, the vectors passed to and from the
LSTM are converted to characters using this mapping. This also includes how many bytes of the
training file to use. Using only a part of the training file can be useful to speed up the training
process. The metadata file I have used is `meta_20000.txt`, and asks to use only the first 20 KB of
the `input.txt` file.

### Creating the Metadata File:

There is an utility script that can be used to create the metadata file for a given dataset file.
Call it via:

```
python utils.py gen_metadata <name_of_dataset_file> <name_of_metadata_file_to_be_generated> [size_of_file_to_use_in_bytes]
```

Note that if the size to be used is not passed, the entire dataset file is used during training,
which may be very slow.

## Training Parameters:

There are two sets of parameters controlling two aspects of the training process:

**Note** that the training should work more or less fine if you leave most of these arguments to
default.

### The Training Loop:

The dataset is broken into chunks of specified length. For each chunk, we pass it in as an input and
compare it with the LSTM output to get a loss value. Then, we update the weights of the LSTM to
minimize this loss. Once we have gone over the entire dataset, we complete an _epoch_. The training
process is repeated for a specified number of epochs. This loop has two parameters that are
controlled via two command line arguments:

 -  `--num-epoch` is the number of epochs to train for.

 -  `--train-seq-size` is the size of the chunk the dataset is split into. Larger chunks will produce
LSTMs that are more context-aware, but will slow down the training speed significantly.

### Learning Rate Falloff:

As the training slowly reaches a limiting point, the learning rate needs to be reduced to make any
progress. So, if we notice that the best reward yet has not been beaten for past `stag_limit` number
of epochs, we reduce the learning rate by multiplying with a factor. Again, there are three parameters
controlled by command line arguments:

 -  `--learning-rate` is the initial learning rate. Should be a float, `0.001` seems to work well.

 -  `--stagnation-limit` is the number of epochs after which to reduce learning rate, or `stag_limit`.

 -  `--lr-reduction-factor` is the factor multiplied with the learning rate to reduce it. Should be a
float less than `1`.

## Saving and Loading Weights:

There are two relevant command line arguments:

 -  `--model-load` The weights are loaded from this file. It should have been generated via
 -  `--model-save` while training a model using the same model class and model arguments.

 -  `--model-save` During training, at the end of each epoch, if the average loss obtained was the best yet among each
epoch, the weights of the model are saved to a file. 
