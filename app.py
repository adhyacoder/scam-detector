from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# --- 1. Dataset for AI Training (Training Data) ---
# Real-world examples of safe and scam messages/links
data = [
    ("hi how are you", "safe"),
    ("hey can we meet tomorrow", "safe"),
    ("please send me the project report", "safe"),
    ("congratulations you won a cash prize click here", "scam"),
    ("urgent your bank account is blocked verify now", "scam"),
    ("free gift card and bonus money inside bit.ly", "scam"),
    ("claim your 25 lakh lottery prize money now", "scam"),
    ("get free crypto rewards and double your investment", "scam"),
    ("amazon offers click to win free iphone", "scam"),
    ("hello buddy call me when you are free", "safe")
]

# Separating messages (features) and their labels (targets)
texts = [row[0] for row in data]
labels = [row[1] for row in data]

# --- 2. Feature Extraction & AI Model Training ---
# Vectorizer converts text data into numerical format for the AI
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Initializing and training the Naive Bayes Classifier model
ai_model = MultinomialNB()
ai_model.fit(X, labels) 


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Capturing user input and converting to lowercase
        user_input = request.form['message'].lower()
        
        # Transforming user input using the same vectorizer
        input_vector = vectorizer.transform([user_input])
        
        # Predicting whether the input is safe or a scam
        prediction = ai_model.predict(input_vector)[0]
        
        # Setting the result message based on AI prediction
        if prediction == "scam":
            result = "Warning: AI thinks this looks like a Scam/Fraud message or unsafe link!"
        else:
            result = "Safe: AI thinks this message/link looks clean."
            
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)