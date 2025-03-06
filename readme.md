# ChatBotAssistent

A simple intent-based chatbot built with PyTorch and NLTK. This chatbot processes user input, classifies intents using a neural network, and responds accordingly. It supports custom function mappings for dynamic responses (e.g., retrieving a list of stocks).

## Features
- Intent recognition using a custom neural network (`ChatBotManual`)
- Tokenization and lemmatization with NLTK
- Training and saving/loading model weights
- Custom function mappings for specific intents (e.g., `stocks`)
- Command-line interface for interaction

## Prerequisites
- Python 3.x
- Required libraries:
  - `torch` (PyTorch)
  - `nltk` (Natural Language Toolkit)
  - `numpy`
- NLTK data:
 
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
  

## Prerequisites
1. Clone this repository:
```bash
    git clone <repository-url>
    cd <repository-directory>
```
2. Install dependencies
```bash
pip install torch nltk numpy
```
3. Run script
```bash
python chatbot.py
```

