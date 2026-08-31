# Python program for Find-S Algorithm

def finds(examples):
    h = list(examples[0][:-1]) # Initialize hypothesis with first example
    for ex in examples:
        if ex[-1] == 'Yes': # Consider only positive examples
            for i in range(len(h)):
                if h[i] != ex[i]:
                    h[i] = '?'
    return h

# Training Data
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

print("Training Examples:")
for row in data:
    print(row)

hypothesis = finds(data)
print("\nMost Specific Hypothesis:")
print(hypothesis)


# Candidate Elimination Algorithm
import copy

# Training Dataset
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Number of attributes
num_attr = len(data[0]) - 1

# Initialize Specific Hypothesis
S = ['0'] * num_attr

# Initialize General Hypothesis
G = [['?' for i in range(num_attr)] for j in range(num_attr)]

def match(hypothesis, example):
    """Checks whether hypothesis covers the example"""
    for i in range(num_attr):
        if hypothesis[i] == '?':
            continue
        elif hypothesis[i] != example[i]:
            return False
    return True

print("\nTraining Examples:\n")
for row in data:
    print(row)
print("\n-------------------------------------")

for example in data:
    if example[-1] == 'Yes': # Positive Example
        # Initialize S with first positive example
        if S == ['0'] * num_attr:
            S = example[:-1]
        else:
            for i in range(num_attr):
                if S[i] != example[i]:
                    S[i] = '?'
        
        # Remove inconsistent hypotheses from G
        G = [g for g in G if match(g, example)]
    else: # Negative Example
        for i in range(num_attr):
            if S[i] != '?' and S[i] != example[i]:
                G[i][i] = S[i]
            else:
                G[i][i] = '?'

# Remove duplicate and all-'?' hypotheses
G_final = []
for g in G:
    if g != ['?'] * num_attr and g not in G_final:
        G_final.append(g)

print("\nFinal Specific Hypothesis (S):")
print(S)

print("\nFinal General Hypotheses (G):")
for g in G_final:
    print(g)
