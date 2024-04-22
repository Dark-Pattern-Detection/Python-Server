from flask_cors import CORS
from bs4 import BeautifulSoup
import time
import json
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast
from transformers import TFBertModel
import re, string
import demoji
from flask import Flask, jsonify, request

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

MAX_LEN=128
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def create_model(bert_model, max_len=MAX_LEN):
    
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-7)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()


    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    embeddings = bert_model([input_ids,attention_masks])[1]
    
    output = tf.keras.layers.Dense(2, activation="softmax")(embeddings)
    
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks], outputs = output)
    
    model.compile(opt, loss=loss, metrics=accuracy)
    
    
    return model

model = create_model(bert_model, MAX_LEN)
model.load_weights("model.hdf5")

def tokenize(data) :
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)

def strip_emoji(text):
    return demoji.replace(text, '') #remove emoji

#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

#Filter special characters such as & and $ present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)

def getScore(body):
    test_input_ids, test_attention_masks = tokenize([body], MAX_LEN)
    score = model.predict([test_input_ids, test_attention_masks])
    print(score)
    
    return np.argmax(score[0])

def preProcessData(data):
  textData = []
  soup = BeautifulSoup(data, 'html.parser')
    # Remove script, noscript, and style elements
  [s.extract() for s in soup(['script', 'noscript', 'style'])]

    # Find all div elements with text
  div_elements = soup.find_all('div')
  # print(div_elements)
  # Extract text and class attributes from div elements
  for div in div_elements:
     # Check if the div has no child div elements
        if not div.find('div'):
          text = div.get_text(separator=' ', strip=True)  # Combine all text into a single line with one space between words
          text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces, tabs, and newlines with a single space
          class_attr = ' '.join(div.get('class', []))
          if text:
              textData.append({
                  'text': text,
                  'class': class_attr
              })
              
  with open('tempdata.json', 'w') as file:
    json.dump(textData, file)
  # print(textData)
  return textData

@app.route('/ping')
def hello_world():  
  return "pong"

def predict(data):
  # data format:
  # [
  #   {
  #     "text" : "some text",
  #     "class" : "unique class to identify the classname for the tag"
  #   },
  #   ....
  # ]
  
  print("Prediction started")
  start_time = time.time()
  predicted_list = []
  for i in range(len(data)):
    score = getScore(data[i]['text'])
    if score == 1:
      predicted_list.append({**data[i],'label': 'Dark Pattern'})
    
  end_time = time.time()
  time_taken = end_time - start_time
  print("Prediction ended")
  print("Time taken:", time_taken, "seconds")
  
  #############
  # Return format: 
  # [
  #   {
  #     text: "", 
  #     class: "",
  #     label: ["", ""] (length of darkPatterns should not be zero. i.e only send data in which dark pattern is found)
  #   },
  #   ....
  # ]
  return predicted_list
  
  


@app.route('/', methods=['POST'])
def predict_post_route():
  try:
    data = request.get_json()['htmlString']
    # data is html of strings
    htmlTextArray = preProcessData(data)
    patternsData = predict(htmlTextArray)
    
    # Return the prediction result as JSON
    # print(patternsData)
    return jsonify({'success': True, 'data': patternsData})  # Change the status code to 200
    
  except Exception as e:
    print(e)
    return jsonify({'success': False, 'error': str(e)}), 400  # Change the status code to 400

if __name__ == "__main__":
  app.run(debug=True, host = 'localhost',port=3000)
  print("app running successfully on ", 3000)
