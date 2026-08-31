import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import re
import random

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

corpus = """natural language processing enables computers to understand human language.
word embeddings map words into continuous vector space.
neural networks can be used to learn these embeddings from context words in a document corpus."""

tokens = tokenize(corpus)

word_counts = Counter(tokens)
vocab = {w: i for i, (w, _) in enumerate(word_counts.items())}
id2word = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)

def generate_pairs(tokens, window_size=2):
    pairs = []
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                pairs.append((center, tokens[j]))
    return pairs

pairs = generate_pairs(tokens, window_size=2)
print("sample training pairs:", pairs[:10])

training_data = [(vocab[c], vocab[context_word]) for c, context_word in pairs]

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_word):
        embed = self.embedding(center_word)
        output = self.output(embed)
        return output


embed_dim = 50
model = Word2Vec(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    total_loss = 0
    for center, context in training_data:
        center_tensor = torch.tensor([center], dtype=torch.long)
        context_tensor = torch.tensor([context], dtype=torch.long)

        optimizer.zero_grad()
        output = model(center_tensor)
        loss = criterion(output, context_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, loss: {total_loss:.4f}")

embeddings = model.embedding.weight.data
print("\n word embedding for 'language':\n", embeddings[vocab["language"]])
print("\n word embedding for 'neural':\n", embeddings[vocab["neural"]])