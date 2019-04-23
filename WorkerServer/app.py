from flask import Flask

app = Flask(__name__)


# Example function for classification task
@app.route('/classification', methods=['POST'])
def classification_handler():
    # Extract POST data
    # Call classification function
    return None