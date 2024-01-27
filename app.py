from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app) 

@app.route('/')
def hello_world():
  return "Hello World"

@app.route('/products')
def get_products():
  return "This is the products page"

def predict(data):
  # data format:
  # [
  #   {
  #     "text" : "some text",
  #     "class" : "unique class to identify the classname for the tag"
  #   },
  #   ....
  # ]
  
  print("Predict Function")
  
  #############
  # Return format: 
  # [
  #   {
  #     class: "",
  #     darkPatterns: ["", ""] (length of darkPatterns should not be zero. i.e only send data in which dark pattern is found)
  #   },
  #   ....
  # ]
  
  # Destructure data[0] and add darkPatterns key with an empty array as its value
  
  # result = [{'class' : data[0]['class'], 'darkPatterns': ['Pattern 1', 'Pattern 2']}]
  result = [{**data[0], 'darkPatterns': ['Pattern 1', 'Pattern 2']}]
  
  return result
  
  


@app.route('/api/v1/predict', methods=['POST'])
def predict_post_route():
  try:
    data = request.get_json()
    
    htmlTextArray = data['data']
    
    
    patternsData = predict(htmlTextArray)
    
    # Perform prediction logic here using the data from the request body
    # ...
    # Return the prediction result as JSON
    print(patternsData)
    return jsonify({'data': patternsData})  # Change the status code to 200
    
  except Exception as e:
    print(e)
    return jsonify({'error': str(e)}), 400  # Change the status code to 400

if __name__ == "__main__":
  app.run(debug=True, host = 'localhost',port=8000)
  print("app running successfully")
