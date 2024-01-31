from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, BertTokenizerFast, BertForSequenceClassification
from bs4 import BeautifulSoup
import time
import re


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

model_name = 'model.yesno.update'
model = BertForSequenceClassification.from_pretrained(model_name)  
tokenizer = BertTokenizerFast.from_pretrained(model_name)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def preProcessData(data):
  textData = []
  soup = BeautifulSoup(data, 'html.parser')
    # Remove script, noscript, and style elements
  [s.extract() for s in soup(['script', 'noscript', 'style'])]

    # Find all div elements with text
  div_elements = soup.find_all('div', text=True)

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
    if nlp(data[i]['text'])[0]['score'] > 0.95:
      if nlp(data[i]['text'])[0]['label'] != 'No dark pattern':
        predicted_list.append({**data[i],'label': nlp(data[i]['text'])[0]['label']})
    
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
