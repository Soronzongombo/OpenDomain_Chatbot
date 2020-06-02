#import files
from flask import Flask, render_template, request
app = Flask(__name__)



import pandas as pd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
from transformer_model import preprocess_sentence
from transformer_model import transformer
from transformer_model import CustomSchedule
from transformer_model import accuracy, loss_function
from colorama import init
from colorama import Fore, Back
init()



your_name = "Soronzoo"
bot_name = "ThesisBot"

#print(f"{Back.BLUE}\n{bot_name} almost ready...{Back.RESET}")

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

#print(f"{Back.BLUE}\nPlease start the conversation: {Back.RESET}")
""" while True:
    print(Fore.LIGHTYELLOW_EX + "")
    prompt = input(f"{your_name}: ")
    print(Fore.RESET + "")
    print(f"{Fore.LIGHTMAGENTA_EX}{bot_name}: {predict(prompt, model)}{Fore.RESET}\n")
""" 



@app.route("/")
def index():    
    return render_template("index.html") 
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')
    response = predict(userText, model)    
    return str(response) 
if __name__ == "__main__":    
    app.run()
