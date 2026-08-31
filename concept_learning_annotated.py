# ---- FIND-S ALGORITHM ----

def finds(examples): # Define function; receives full dataset as input
    h = list(examples[0][:-1]) # Take 1st row, drop label -> starting hypothesis
    for ex in examples: # Loop through every training example one by one
        if ex[-1] == 'Yes': # Only process POSITIVE examples; skip 'No' rows
            for i in range(len(h)): # Loop through each attribute position (0 to 5)
                if h[i] != ex[i]: # If hypothesis value differs from example value
                    h[i] = '?' # Generalise: '?' means any value is OK here
    return h # Return final most-specific hypothesis

data = [ # Define training data as a list of lists
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'], # Example 1 - POSITIVE (Yes)
    ['Sunny','Warm','High','Strong','Warm','Same','Yes'], # Example 2 - POSITIVE (Yes)
    ['Rainy','Cold','High','Strong','Warm','Change','No'], # Example 3 - NEGATIVE (No)
    ['Sunny','Warm','High','Strong','Cool','Change','Yes'] # Example 4 - POSITIVE (Yes)
] # End of dataset

print("Training Examples:") # Print heading before showing data
for row in data: # Loop through each row in data
    print(row) # Print one training example per line

hypothesis = finds(data) # Call finds() with full dataset; store result
print("\nMost Specific Hypothesis:") # Print result label
print(hypothesis) # Print final hypothesis e.g. ['Sunny','Warm','?','Strong','?','?']

# PART 2 - Candidate Elimination Algorithm
import copy # Import copy module to make independent copies of lists

num_attr = len(data[0]) - 1 # Count attributes: length of 1st row minus label = 6

S = ['0'] * num_attr # S = ['0',...] - most specific; rejects everything

G = [['?' for i in range(num_attr)] for j in range(num_attr)] # G = list of 6 rows, each all '?' - most general
# Creates 6x6 grid of candidate general hypotheses

def match(hypothesis, example): # Helper function: checks if hypothesis says YES to example
    for i in range(num_attr): # Loop through each attribute position
        if hypothesis[i] == '?': # If '?' -> any value accepted here
            continue # Skip this attribute - it automatically matches
        elif hypothesis[i] != example[i]: # Specific value doesn't match example's value
            return False # Hypothesis does NOT cover this example
    return True # All attributes passed -> hypothesis covers example

print("\nTraining Examples:\n") # Print heading
for row in data: # Loop through all rows
    print(row) # Print each training example
print("\n" + "-"*37) # Print separator line

for example in data: # Process each example through CE algorithm
    if example[-1] == 'Yes': # POSITIVE # Check if label is Yes -> positive example
        if S == ['0'] * num_attr: # If S still has initial zeros (not yet set)
            S = example[:-1] # Set S = first positive example (drop label)
        else: # For all subsequent positive examples:
            for i in range(num_attr): # Loop through each attribute
                if S[i] != example[i]: # If S value differs from this example's value
                    S[i] = '?' # Generalise S: '?' covers both values
        
        G = [g for g in G if match(g, example)] # Positive example: filter G
        # Keep only G hypotheses that match (say YES to) this positive example
    else: # NEGATIVE EXAMPLE # Label is No -> negative example
        for i in range(num_attr): # Loop through each attribute
            if S[i] != '?' and S[i] != example[i]: # S has specific value AND differs from negative
                G[i][i] = S[i] # Specialise G: fix attribute to S value -> excludes negative
            else: # S has '?' or same value as negative here
                G[i][i] = '?' # No useful constraint -> leave '?'

G_final = [] # Create empty list for clean final G hypotheses
for g in G: # Go through every hypothesis in G
    if g != ['?'] * num_attr and g not in G_final: # Skip if all '?' - too general, useless
        # Skip if already added (remove duplicates)
        G_final.append(g) # Add valid unique hypothesis to G_final

print("\nFinal Specific Hypothesis (S):") # Print label for S boundary
print(S) # Print S - most specific boundary

print("\nFinal General Hypotheses (G):") # Print label for G boundary
for g in G_final: # Loop through all valid general hypotheses
    print(g) # Print each G hypothesis - most general boundaries
