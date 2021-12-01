"""
A number of models that would be trained as verification targets. All models should be initialized
with `Model(n_chars, *args)` where `n_chars` is the number of input characters inferred from the
training data, and `*args` are further optional arguments.
"""



import torch
import torch.nn as nn
import torch.nn.functional as F



class TorchLSTM(nn.Module):
    """
    A simple LSTM for text generation. Members are:
    
    char_size   -   Alphabet size of text to generate
    num_layers  -   The number of LSTM layers
    lstm        -   The internal LSTM model
    decoder     -   DNN to produce output from hidden state
    """
    def __init__(self, char_size, hidden_size, num_layers):
        super(TorchLSTM, self).__init__()
        self.char_size = char_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=self.char_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, self.char_size)
        
        self.internal_state = None 
    
    def forward(self, input_seq):
        x = F.one_hot(input_seq, self.char_size).float()
        x = torch.unsqueeze(x, dim=1)                               # Add a batch
        x, new_internal_state = self.lstm(x, self.internal_state)
        x = torch.squeeze(x)                                        # Remove batch
        output = self.decoder(x)
        self.internal_state = (new_internal_state[0].detach(), new_internal_state[1].detach())
        return output
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        try:
            self.load_state_dict(torch.load(path))
        except Exception as err:
            print("Error loading model from file", path)
            print(err)
            print("Initializing model weights to default")
            self.__init__(self.char_size, self.hidden_size, self.num_layers)

            
            
class CustomLSTM(nn.Module):
    """
    A custom LSTM for text generation. Members are:
    
    char_size   -   Alphabet size of text to generate
    hidden_size -   The latent state dimensions
    decoder     -   Linear layer of DNN to produce output from hidden state
    cell_inp    -   Linear layer of DNN calculating input to memory cell
    i_gate      -   Input gate linear layer
    f_gate      -   Forget gate linear layer
    o_gate      -   Output gate linear layer
    sigmoid     -   Various activation functions
    tanh        -   Another activation function
    h, c        -   Hidden state and memory cell, initially None. This does not retain compute
                    graph.
    """ 
    def __init__(self, char_size, hidden_size, init_rand = True): 
        """
        Create a new LSTM with given `hidden_size`. If `init_rand` is False, no random
        initialization is done.
        """
        super(CustomLSTM, self).__init__() 

        # Copy data
        self.char_size      = char_size
        self.hidden_size    = hidden_size

        # Parameters
        self.decoder    = nn.Linear(self.hidden_size, self.char_size)
        self.cell_inp   = nn.Linear(self.hidden_size + self.char_size, self.hidden_size)
        self.i_gate     = nn.Linear(self.hidden_size + self.char_size, self.hidden_size)
        self.f_gate     = nn.Linear(self.hidden_size + self.char_size, self.hidden_size)
        self.o_gate     = nn.Linear(self.hidden_size + self.char_size, self.hidden_size)

        # Initialize params
        if init_rand:
            k = 1 / hidden_size ** 0.5
            nn.init.uniform_(self.decoder.weight, -k, k)
            nn.init.uniform_(self.decoder.bias, -k, k)
            nn.init.uniform_(self.cell_inp.weight, -k, k)
            nn.init.uniform_(self.cell_inp.bias, -k, k)
            nn.init.uniform_(self.i_gate.weight, -k, k)
            nn.init.uniform_(self.i_gate.bias, -k, k)
            nn.init.uniform_(self.f_gate.weight, -k, k)
            nn.init.uniform_(self.f_gate.bias, -k, k)
            nn.init.uniform_(self.o_gate.weight, -k, k)
            nn.init.uniform_(self.o_gate.bias, -k, k)

        # Activations
        self.sigmoid    = nn.Sigmoid()
        self.tanh       = nn.Tanh()

        # Latent state
        self.h, self.c = None, None

        
    def forward(self, input_seq):
        """
        Get the output sequence for given input sequence of indices. The input should be a 1-tensor
        with a single sequence. Returns a sequence of vectors.
        """
        # Convert input sequence to one hot
        xs = F.one_hot(input_seq, self.char_size).float()

        # Initialize LSTM state
        h = torch.zeros(self.hidden_size) if self.h is None else self.h.detach()
        c = torch.zeros(self.hidden_size) if self.c is None else self.c.detach()

        # Initialize output
        output = []

        # Loop over sequence
        for x in xs:

            # Combine with h
            hx = torch.cat((h, x))

            # Calculate gate values
            f = self.sigmoid( self.f_gate(hx) )
            i = self.sigmoid( self.i_gate(hx) )
            o = self.sigmoid( self.o_gate(hx) )

            # Calculate new state
            g = self.tanh( self.cell_inp(hx) )
            c = f * c + i * g

            # Calculate new hidden state
            h = o * self.tanh(c)

            # Add output
            output.append(self.decoder(h))

        # Detach compute graph for h and c.
        self.h = h.detach()
        self.c = h.detach()
        
        # Return output
        return torch.stack(output)
    

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        try:
            self.load_state_dict(torch.load(path))
        except Exception as err:
            print("Error loading model from file", path)
            print(err)
            print("Initializing model weights to default")
            self.__init__(self.char_size, self.hidden_size, init_rand = False)
