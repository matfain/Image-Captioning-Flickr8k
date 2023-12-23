import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import torch.nn.init as init
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super().__init__()
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        for param in resnet.parameters():
            param.requires_grad_(train_CNN)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.embed.weight.data.normal_(0., 0.02)
        self.embed.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        reshaped = features.view(features.size(0), -1)      # reshaped.shape = (B, 2048*7*7)
        out = self.bn(self.embed(reshaped))                 # out.shape = (B, embed_size)
        return out
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)    # Because batch_first=False expected input shape is (T, B, embed_size) where T is longest sequence size
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))                     #captions.shape = (T, B) --> embeddings.shape = (T, B, embed_size)
        combined = torch.cat((features.unsqueeze(0), embeddings), dim=0)    # after unsqueeze features.shape = (1, B, embed_size) --> combined.shape = (T+1, B, embed_size) so now every sequence first "word" is the image as [img, <SOS>, ..., <EOS>]
        hiddens, _ = self.lstm(combined)                                    # hiddens.shape = (T+1, B, hidden_size)
        out = self.linear(hiddens)                                          # out.shape = (T+1, B, vocab_size)
        return out
    

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train_CNN=False):
        super().__init__()
        self.encoder = EncoderCNN(embed_size, train_CNN)                                           
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)                     #features.shape = (B, embed_size)
        out = self.decoder(features, captions)              #out.shape = (T+1, B, vocab_size)
        return out
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for i in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                logits = self.decoder.linear(hiddens.squeeze(0))
                predicted = logits.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
    


"""
During training the following proccess happens:
* imgs.shape = (32, 3, 224, 224) --> (batch_size, channels, height, width)
* captions.shape = (22, 32)   --> (longest_sequence, batch_size) --> each column in the matrix is a sequence
  and each number represents an idx in the vocabulary. The longest sequence varies for each batch.

First the imgs are passed through the encoderCNN which results in:
* Resnet output dimension = (32, 2048, 7, 7) --> each img was converted to 2048 feature maps of 7x7
* Reshaping by features.view(features.size(0), -1) ---> result is (32 , 2048*7*7)
* Another Reshape by Linear layer ---> result is (32, embed_size) ---> After BN this will be the input shape for the decoder

Second phase is the decoderRNN:
* The embedding layer takes the caption of shape (22,32) ---> converts it to (22, 32, embed_size)
* Featurs & Embeddings concatenation:
  features.unsqueeze(0) takes the (32, embed_size) feature vector to --> (1, 32, embed_size)
  torch.cat takes the features and concatenates them so for every timestamp the features will repat which results
  in a combined features&embeddings input of shape (23, 32, embed_size)
* LSTM layer take as input the combined (23, 32, embed_size) ---> returns output of the same shape (23, 32, embed_size)
* Linear layer (almost softmax) this layer is responsible to create logits for each word in the vocabulary and thus
  the input is (23, 32, embed_size) which is mapped to ---> (23, 32, vocab_size)
"""