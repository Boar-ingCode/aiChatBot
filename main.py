import os
import json
import random
import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset

nltk.download('punkt')  
nltk.download('wordnet')
class ChatBotManual(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatBotManual, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

class ChatBotAssistent:

    def __init__(self, intent_path, fucntion_mappings = None):
        self.Model = None
        self.intent_path = intent_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = []

        self.fucntion_mappings = fucntion_mappings

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemanize(sentence):
        lemmatizer = nltk.stem.WordNetLemmatizer()

        words = nltk.word_tokenize(sentence)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words


chatbot = ChatBotAssistent("intents.json")
print(chatbot.tokenize_and_lemanize("Hello, how are you?"))