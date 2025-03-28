import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
    def __init__(self, intent_path, function_mappings=None):
        self.model = None  
        self.intent_path = intent_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings 
        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(sentence): 
        lemmatizer = nltk.stem.WordNetLemmatizer()
        words = nltk.word_tokenize(sentence)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_word(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def pass_intents(self):
        if os.path.exists(self.intent_path):
            with open(self.intent_path, 'r') as f:
                intents_data = json.load(f)
            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']
                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))
            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []
        for document in self.documents:
            words = document[0]
            bag = self.bag_of_word(words)
            intent_index = self.intents.index(document[1])  # Fixed typo
            bags.append(bag)
            indices.append(intent_index)
        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatBotManual(self.X.shape[1], len(self.intents)) 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()  
            print(f'Epoch: {epoch + 1}, Loss: {running_loss / len(loader):.4f}')

    def save_model(self, model_path, dimensions_path):
        if self.model is None:
            raise ValueError("No model to save.")
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        self.model = ChatBotManual(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message): 
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_word(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        if self.model is None:
            raise ValueError("Model not loaded or trained.")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
            predicted_class_index = torch.argmax(predictions, dim=1).item()
            predicted_intent = self.intents[predicted_class_index]

            if self.function_mappings and predicted_intent in self.function_mappings:
                return self.function_mappings[predicted_intent]()
            elif predicted_intent in self.intents_responses:
                return random.choice(self.intents_responses[predicted_intent])
            return "Sorry, I don’t understand."

def get_stocks():
    stocks = ['Apple', 'Meta', 'Nvidia', 'Tesla', 'Microsoft']
    return random.sample(stocks, 5)

if __name__ == '__main__':
    assistant = ChatBotAssistent('intents.json', {'stocks': get_stocks})
    assistant.pass_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)
    assistant.save_model('model.pth', 'dimensions.json')

    assistant.load_model('model.pth', 'dimensions.json')

    while True:
        message = input('Enter your message: ')
        if message == '/quit':
            break
        print(assistant.process_message(message))