from random import randrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm 

from models import *



def train_single_epoch(rnn_model, data, optimizer, trn_seq_len):
    """
    Train RNN model for one epoch. Arguments are: 
    
    rnn_model   -   The RNN
    data        -   1D Tensor of indices to train on
    optimizer   -   The optimizer to train with
    trn_seq_len -   Train on a sequence of this length. 
    
    Returns
    
    1. Average loss
    """
    
    # Switch model to training mode
    rnn_model.train()

    # Set up loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Collect the average loss
    tot_loss = 0
    n = 0
    ads = data.size()[0] - trn_seq_len
    sti = randrange( min(trn_seq_len, ads) )
    for data_ptr in tqdm(range( sti, ads, trn_seq_len)):

        # Get sequences
        input_seq = data[data_ptr : data_ptr+trn_seq_len]
        target_seq = data[data_ptr+1 : data_ptr+trn_seq_len+1]
        
        # Training step
        optimizer.zero_grad()
        output = rnn_model(input_seq)
        loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
        loss.backward()
        optimizer.step()
        
        # Update loss
        tot_loss += loss.item()
        n = n+1

    # Return
    return tot_loss / n
            


def generate_from_one_input(rnn_model, inp, seq_len):
    """
    Generate and return a sequence of length seq_len
    """

    # print loss and a sample of generated text periodically
    tst_out = []

    # set model to eval mode
    rnn_model.eval()

    # random character from data to begin
    input_seq = torch.tensor([inp])
    
    # Generate
    for _ in range(seq_len):

        # forward pass
        output = rnn_model(input_seq)

        # construct categorical distribution and sample a character
        output = F.softmax(torch.squeeze(output), dim=0)
        dist = Categorical(output)
        index = dist.sample().item()

        # append the sampled character to test_output
        tst_out.append(index)

        # next input is current output
        input_seq[0] = index

    return tst_out



if __name__=="__main__":
    """
    Run the script. Refer to help for arguments.
    """

    import sys
    from argparse import ArgumentParser
    
    # Argument parsing
    parser = ArgumentParser()
    parser.add_argument("--model-name", default = "TorchLSTM", dest = "model_name",
                        help = "The name of the model to train")
    parser.add_argument("--model-args", default = "300,2", dest = "model_args",
                        help = "The arguments passed to the model")
    parser.add_argument("--model-load", default = None, dest = "load_fname",
                        help = ("The model is loaded from this file, if given. Should be compatible "
                                "with model name and args") )
    parser.add_argument("--model-save", default = None, dest = "save_fname",
                        help = "If given, best model yet is saved to this file")
    parser.add_argument("--data-set", dest = "data_fname",
                        help = "Read training data from this file")
    parser.add_argument("--data-meta", dest = "meta_fname",
                        help = "Read training data's metadata from this file")
    parser.add_argument("--num-epochs", default = 1000, type = int, dest = "num_epochs",
                        help = "The number of epochs to train for")
    parser.add_argument("--train-seq-size", default = 200, type = int, dest = "seq_size",
                        help = "The size of input sequences to train over")
    parser.add_argument("--stagnation_limit", default = 500, type = int, dest = "stag_lim",
                        help = "The number of times we stagnated after which to reduce learning rate")
    parser.add_argument("--learning-rate", default = 0.001, type = float, dest = "init_lr",
                        help = "The initial learning rate")
    parser.add_argument("--lr-reduction-factor", default = 0.7, type = float, dest = "lr_red_fac",
                        help = "The factor by which learning rate is reduced every time we stagnate")
    args = parser.parse_args()
    
    # Open the text and metadata file
    meta = eval( open(args.meta_fname, 'r').read() )
    data = open(args.data_fname, 'r').read(meta['data_size'])
    chars = meta['charset']
    data_size, vocab_size = len(data), len(chars) 
    print("Data has {0} characters, {1} unique".format(data_size, vocab_size))
    char_to_ix = meta['char_to_ix'] 
    ix_to_char = meta['ix_to_char'] 

    # convert data from chars to indices
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]
    print("Data converted")

    # data tensor on device
    data = torch.tensor(data)
    
    # Generate a model
    model = eval("{0}(vocab_size, {1})".format( args.model_name, args.model_args ))
    print("Created model")

    # Load params
    if args.load_fname is not None:
        model.load_model(args.load_fname)
        print("Loaded from file, generating text before any training")
        # Print out some generated text
        gen_data = generate_from_one_input(model, randrange(vocab_size), args.seq_size)
        gen_str = ''.join(( ix_to_char[c] for c in gen_data ))
        print("Generated batch of text:\n {0}".format(gen_str))

    # And optimizer
    lrate = args.init_lr
    optimizer = torch.optim.Adam(model.parameters(), lr = lrate)
    n_stag = 0
    stag_limit = args.stag_lim

    # Epoch loop
    best_loss = float('inf')
    for epoch in range(0, args.num_epochs):
        print("Training epoch {0}".format(epoch))
        
        # Train for one epoch
        epoch_loss = train_single_epoch(model, data, optimizer, args.seq_size)
        print("Loss was {0}".format(epoch_loss))

        # Print stats from test
        gen_data = generate_from_one_input(model, randrange(vocab_size), args.seq_size)
        gen_str = ''.join(( ix_to_char[c] for c in gen_data ))
        print("Generated batch of text:\n {0}".format(gen_str))

        # Save if best yet
        if epoch_loss < best_loss and args.save_fname is not None:
            print("Best yet, saving model")
            best_loss = epoch_loss
            model.save_model(args.save_fname)
            n_stag = 0
        else:
            n_stag += 1
            
        # Reset optimizer with lower learning rate
        if n_stag >= stag_limit:
            lrate *= args.lr_red_fac
            stag_limit /= args.lr_red_fac
            optimizer = torch.optim.Adam(model.parameters(), lr = lrate)
            n_stag = 0
            print("Reducing LR to {0}".format(lrate))
