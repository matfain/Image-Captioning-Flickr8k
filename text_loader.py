# This module will load the captions and preprocess them for LSTM RNN usage later
# Based on https://www.youtube.com/watch?v=9sHcLvVXsns&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=10

import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        out = [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
        return out


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Init vocabulary and build the vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)
    

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)                     # imgs.shape will be (B, 3, H, W)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)  #batch_first=False will result in targets.shape to be (T, B) where T: longest sequence size in the batch and B: is batch size.

        return imgs, targets
    
def get_loader(root_folder, caption_file, transform, batch_size=32, num_workers=8, shuffle=True, pin_memory=True):
    dataset = FlickrDataset(root_folder, caption_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=shuffle, pin_memory=pin_memory, collate_fn=MyCollate(pad_idx=pad_idx))
    
    return loader, dataset

def TVT_split(root_folder, caption_file, split_ratio, transform, batch_size=32, num_workers=8, shuffle=True, pin_memory=True):
    dataset = FlickrDataset(root_folder, caption_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    #Splits the Dataset
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, split_ratio)

    #Creates relevant loaders
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, num_workers=num_workers,
                              shuffle=shuffle, pin_memory=pin_memory, collate_fn=MyCollate(pad_idx=pad_idx))

    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, pin_memory=pin_memory, collate_fn=MyCollate(pad_idx=pad_idx))

    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, pin_memory=pin_memory, collate_fn=MyCollate(pad_idx=pad_idx))

    out = {"train_ds": train_ds, "train_loader": train_loader,
           "val_ds": val_ds, "val_loader": val_loader,
           "test_ds": test_ds, "test_loader": test_loader}
    return out