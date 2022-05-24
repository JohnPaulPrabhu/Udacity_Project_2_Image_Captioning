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
        self.batchnorm_2d = nn.BatchNorm2d(resnet.fc.in_features)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        torch.nn.init.xavier_uniform_(self.embed.weight)
        self.bnFinal = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = self.batchnorm_2d(features)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bnFinal(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size))
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bias=True,
                            num_layers=num_layers, dropout=0, batch_first = True, 
                            bidirectional=False)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.init_hidden()
    
    def init_hidden(self):
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, features, captions):
        cap_embed = self.embedding(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(dim=1),cap_embed), dim=1)
        lstm_out, self.hidden = self.lstm(inputs)
        output = self.out(lstm_out)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        
        for i in range(max_len):
            #passing the image into lstm 
            hid, states = self.lstm(inputs, states) 
            outputs = self.out(hid.squeeze(1)) 
            _, predicted = outputs.max(1)
            
            #add the predicted item to sentence
            sentence.append(predicted.item())
            
            #preparing for the next word
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)  
        return sentence 