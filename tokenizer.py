import torch
class CharTokenizer:
    def __init__(self,chars):
        self.chars = chars
        self.stoi = {ch:i for i, ch in chars}
        self.itos = {i:ch for i ,ch in chars}

    def encode(self,s):
        return [self.stoi[c] for c in s]

    def decode(self,l):
        return ''.join([self.itos[i] for i in l])

