# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:56:41 2019

@author: shams
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class myNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=8, batch_size, output_dim=1, num_lstm_layer = 1):    
        # initialize 
        super(myNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_lstm_layer = num_lstm_layer
        
        # add one lstm layer
        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, 1)
        # add one batch-norm layer
        self.batchnorm_layer = nn.BatchNorm1d(hidden_dim)
        # add one dense layer
        self.dense_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        # add output layer 
        self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        
    
    def init_hidden(self):
        #  initialise lstm hidden state variabls (h0,c0) to zero
        return (torch.zeros(self.num_lstm_layer, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_lstm_layer, self.batch_size, self.hidden_dim))

        
    def forward(self, x):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm_layer(x.view(len(x), self.batch_size, -1))
        out = self.batchnorm_layer(lstm_out...)
        out = self.dense_layer(out,...)
        out = self.output_layer(out,...)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)
    
     def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out