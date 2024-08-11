from flask import Flask, request, jsonify
from flask_cors import CORS
from scripts.fitnessLangchain import perform_rag_fitness
from scripts.inspirationLangchain import perform_rag_inspiration
from scripts.journalLangchain import perform_rag_journal
from scripts.mindfulLangchain import perform_rag_mindful
from scripts.moodLangchain import perform_rag_mood

# app instance 
app = Flask(__name__)
CORS(app)

@app.route('/api/fitness', methods=['POST'])
def return_fitness():
    data = request.get_json()
    # print('****** Data: ****** ', data)  # Print the data for debugging

    # Assuming the query is the last item in the list
    query = data[-1].get('query', '') if isinstance(data, list) and data else ''

    print('here is the query:', query)

    try:
        response = perform_rag_fitness(query)  # Get the response using the custom function
        # print('Response:', response)  # Print the response for debugging 
        return jsonify (
            {'response': response}  # Return the response as JSON
        )
    except Exception as e:
        print('Error:', str(e))  # Print the error message for debugging purposes
        return jsonify (
            {'error': str(e)}  # Return the error message as JSON
        ), 500  # Return a 500 Internal Server Error status code
    
@app.route('/api/inspiration', methods=['POST'])
def return_inspiration():
    data = request.get_json()
    # print('****** Data: ****** ', data)  # Print the data for debugging

    # Assuming the query is the last item in the list
    query = data[-1].get('query', '') if isinstance(data, list) and data else ''

    print('here is the query:', query)

    try:
        response = perform_rag_inspiration(query)  # Get the response using the custom function
        # print('Response:', response)  # Print the response for debugging 
        return jsonify (
            {'response': response}  # Return the response as JSON
        )
    except Exception as e:
        print('Error:', str(e))  # Print the error message for debugging purposes
        return jsonify (
            {'error': str(e)}  # Return the error message as JSON
        ), 500  # Return a 500 Internal Server Error status code

@app.route('/api/journal', methods=['POST'])
def return_journal():
    data = request.get_json()
    # print('****** Data: ****** ', data)  # Print the data for debugging

    # Assuming the query is the last item in the list
    query = data[-1].get('query', '') if isinstance(data, list) and data else ''

    print('here is the query:', query)

    try:
        response = perform_rag_journal(query)  # Get the response using the custom function
        # print('Response:', response)  # Print the response for debugging 
        return jsonify (
            {'response': response}  # Return the response as JSON
        )
    except Exception as e:
        print('Error:', str(e))  # Print the error message for debugging purposes
        return jsonify (
            {'error': str(e)}  # Return the error message as JSON
        ), 500  # Return a 500 Internal Server Error status code
    

## Shouldnt need support route as of now

# @app.route('/api/liveSupport', methods=['POST'])
# def return_home():
#     data = request.get_json()
#     # print('****** Data: ****** ', data)  # Print the data for debugging

#     # Assuming the query is the last item in the list
#     query = data[-1].get('query', '') if isinstance(data, list) and data else ''

#     print('here is the query:', query)

#     try:
#         response = perform_rag(query)  # Get the response using the custom function
#         # print('Response:', response)  # Print the response for debugging 
#         return jsonify (
#             {'response': response}  # Return the response as JSON
#         )
#     except Exception as e:
#         print('Error:', str(e))  # Print the error message for debugging purposes
#         return jsonify (
#             {'error': str(e)}  # Return the error message as JSON
#         ), 500  # Return a 500 Internal Server Error status code
    
@app.route('/api/mindfulness', methods=['POST'])
def return_mindfulness():
    data = request.get_json()
    # print('****** Data: ****** ', data)  # Print the data for debugging

    # Assuming the query is the last item in the list
    query = data[-1].get('query', '') if isinstance(data, list) and data else ''

    print('here is the query:', query)

    try:
        response = perform_rag_mindful(query)  # Get the response using the custom function
        # print('Response:', response)  # Print the response for debugging 
        return jsonify (
            {'response': response}  # Return the response as JSON
        )
    except Exception as e:
        print('Error:', str(e))  # Print the error message for debugging purposes
        return jsonify (
            {'error': str(e)}  # Return the error message as JSON
        ), 500  # Return a 500 Internal Server Error status code
    
@app.route('/api/mood', methods=['POST'])
def return_mood():
    data = request.get_json()
    # print('****** Data: ****** ', data)  # Print the data for debugging

    # Assuming the query is the last item in the list
    query = data[-1].get('query', '') if isinstance(data, list) and data else ''

    print('here is the query:', query)

    try:
        response = perform_rag_mood(query)  # Get the response using the custom function
        # print('Response:', response)  # Print the response for debugging 
        return jsonify (
            {'response': response}  # Return the response as JSON
        )
    except Exception as e:
        print('Error:', str(e))  # Print the error message for debugging purposes
        return jsonify (
            {'error': str(e)}  # Return the error message as JSON
        ), 500  # Return a 500 Internal Server Error status code
    



# GET request TEST
@app.route('/api/home', methods=['GET'])
def hello_world():
    return jsonify({'message': 'helloworld'})  # Return "helloworld" as JSON

if __name__ == '__main__':
    app.run(debug=True, port=8080) # remove debug=True in production