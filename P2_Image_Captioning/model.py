import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_size
        
        # embedding layer that turns words into a vector of a specified size
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # the linear layer that maps the hidden state output dimension 
        # to the number of words we want as output, output_size
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the hidden state (see code below)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on previously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence
        embeds = self.embeddings(captions[:,:-1])
        emb_input = torch.cat((features.unsqueeze(1), embeds), 1)
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(emb_input)
        
        # get the scores for the most likely next word
        outputs = self.fc(lstm_out)
        
        return outputs
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        output_length = 0
        
        states = (torch.randn(1, 1, self.hidden_dim).to(inputs.device), 
                  torch.randn(1, 1, self.hidden_dim).to(inputs.device))
        
        while (output_length != max_len+1):
            output, states = self.lstm(inputs,states)
            output = self.fc(output.squeeze(dim = 1))
            _, predicted_index = torch.max(output, 1)
            outputs.append(predicted_index.cpu().numpy()[0].item())
            # Stop prediction ones we reach to the end_index
            if (predicted_index == 1):
                break
            inputs = self.embeddings(predicted_index)   
            inputs = inputs.unsqueeze(1)
            output_length += 1
        
        return outputs
        