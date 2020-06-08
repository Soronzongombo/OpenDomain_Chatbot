#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:29:29 2020

@author: user
"""

import pandas as pd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
from transformer_model import preprocess_sentence
from transformer_model import transformer
from transformer_model import CustomSchedule
from transformer_model import accuracy, loss_function
from nltk.translate.bleu_score import sentence_bleu

strategy = tf.distribute.get_strategy()

# Maximum sentence length
MAX_LENGTH = 40

# For tf.data.Dataset
BATCH_SIZE = int(64 * strategy.num_replicas_in_sync)
BUFFER_SIZE = 20000

# For Transformer
NUM_LAYERS = 2 #6
D_MODEL = 512 #512
NUM_HEADS = 8
UNITS = 2048 #2048
DROPOUT = 0.1

EPOCHS = 100

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

def evaluate(sentence, model):
  sentence = preprocess_sentence(sentence)
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)
    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]):
      break
    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)




def predict(sentence,model):
  prediction = evaluate(sentence,model)
  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])
  return predicted_sentence.lstrip()


print("Importing trained model...")

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model = transformer(
      vocab_size=VOCAB_SIZE,
      num_layers=NUM_LAYERS,
      units=UNITS,
      d_model=D_MODEL,
      num_heads=NUM_HEADS,
      dropout=DROPOUT)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
model.load_weights('openDomain_weights.h5')

dataSet = pd.read_csv('OpenDomain.csv')
questions = list()
answers = list()
generatedAnswers = list()
print("Generating candidate responses...")
for i in range(100):
    questions.append(dataSet.values[i][1])
    answers.append(dataSet.values[i][2])
    cleared_sentence = preprocess_sentence(dataSet.values[i][1])
    generatedAnswers.append(predict(cleared_sentence, model))
    



score_1 = 0.0
score_2 = 0.0
score_3 = 0.0
score_4 = 0.0

print('Calculating average BLEU score...')

for i in range(len(answers)):
    reference = answers[i].split()
    candidate = generatedAnswers[i].split()
    score_1 = score_1 + sentence_bleu(reference, candidate, weights=[1, 0, 0, 0])
    score_2 = score_2 + sentence_bleu(reference, candidate, weights=[0.5, 0.5, 0, 0])
    score_3 = score_3 + sentence_bleu(reference, candidate, weights=[0.33, 0.33, 0.33, 0])
    score_4 = score_4 + sentence_bleu(reference, candidate, weights=[0.25, 0.25, 0.25, 0.25])

score_1_avg = score_1/len(answers)
score_2_avg = score_2/len(answers)   
score_3_avg = score_3/len(answers)   
score_4_avg = score_4/len(answers)

print("Average 1-gram {0}".format(round(score_1_avg,2)))
print("Average 2-gram {0}".format(round(score_2_avg,2)))
print("Average 3-gram {0}".format(round(score_3_avg,2)))
print("Average 4-gram {0}".format(round(score_4_avg,2)))



print('Calculating maximum BLEU score...')
score_1_max = list()
score_2_max = list()
score_3_max = list()
score_4_max = list()

for i in range(len(answers)):
    reference = answers[i].split()
    candidate = generatedAnswers[i].split()
    score_1_max.append(sentence_bleu(reference, candidate, weights=[1, 0, 0, 0]))
    score_2_max.append(sentence_bleu(reference, candidate, weights=[0.5, 0.5, 0, 0]))
    score_3_max.append(sentence_bleu(reference, candidate, weights=[0.33, 0.33, 0.33, 0]))
    score_4_max.append(sentence_bleu(reference, candidate, weights=[0.25, 0.25, 0.25, 0.25]))
    

print("Maximum 1-gram {0}".format(round(max(score_1_max),2)))
print("Maximum 2-gram {0}".format(round(max(score_2_max),2)))
print("Maximum 3-gram {0}".format(round(max(score_3_max),2)))
print("Maximum 4-gram {0}".format(round(max(score_4_max),2)))
    


