import tensorflow as tf
import numpy as np
import os
import random
import sys


def load_data(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text


def create_mappings(text):
    chars = sorted(set(text))
    char_to_index = {char: idx for idx, char in enumerate(chars)}
    index_to_char = {idx: char for idx, char in enumerate(chars)}
    return char_to_index, index_to_char, len(chars)


def prepare_sequences(text, seq_length, char_to_index):
    sequences = []
    targets = []
    for i in range(0, len(text) - seq_length, 1):
        seq_in = text[i:i+seq_length]
        seq_out = text[i+seq_length]
        sequences.append([char_to_index[char] for char in seq_in])
        targets.append(char_to_index[seq_out])
    return np.array(sequences), np.array(targets)


def create_model(seq_length, num_chars):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_chars, 256, input_length=seq_length),
        tf.keras.layers.LSTM(512, return_sequences=False),
        tf.keras.layers.Dense(num_chars, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def generate_text(model, index_to_char, char_to_index, seed_text, seq_length, num_gen_chars=100):
    generated_text = seed_text
    for _ in range(num_gen_chars):
        
        input_sequence = np.array([char_to_index[char] for char in seed_text[-seq_length:]]).reshape(1, seq_length)
      
        predicted_probs = model.predict(input_sequence)
        predicted_char_idx = np.argmax(predicted_probs)
        predicted_char = index_to_char[predicted_char_idx]
        
     
        generated_text += predicted_char
        seed_text += predicted_char
    return generated_text


def train_model(file_path, seq_length=100, num_epochs=50, batch_size=64):
    
    text = load_data(file_path)
    char_to_index, index_to_char, num_chars = create_mappings(text)
    
    
    sequences, targets = prepare_sequences(text, seq_length, char_to_index)
    
    
    model = create_model(seq_length, num_chars)
    
    
    model.fit(sequences, targets, epochs=num_epochs, batch_size=batch_size)
    
   
    seed_text = "Once upon a time"  
    generated_text = generate_text(model, index_to_char, char_to_index, seed_text, seq_length, num_gen_chars=500)
    
  
    print("Generated Text:")
    print(generated_text)
    return model


file_path = 'handwritten_text.txt' 
model = train_model(file_path, seq_length=100, num_epochs=50, batch_size=64)
