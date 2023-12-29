import pandas as pd
import numpy as np
#precily=pd.read_csv("D:/New download files/Sr. Data Scientist_DataNeuron_Task/Precily_Task/Precily_Text_Similarity.csv")
precily=pd.read_csv("Precily_Text_Similarity.csv")
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text
# Apply preprocessing to each column
precily['text1_processed'] = precily['text1'].apply(preprocess_text)
precily['text2_processed'] = precily['text2'].apply(preprocess_text)

from gensim.models import Word2Vec

# Tokenize the preprocessed text into word tokens
tokenized_text1 = precily['text1_processed'].apply(lambda x: x.split())
tokenized_text2 = precily['text2_processed'].apply(lambda x: x.split())

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_text1 + tokenized_text2, vector_size=100, window=5, min_count=1, workers=4)

# Function to get average word vectors for a sentence
def get_average_word_vectors(tokens_list, model, vector_size):
    if len(tokens_list) < 1:
        return np.zeros(vector_size)
    vectors = [model.wv[word] for word in tokens_list if word in model.wv]
    if len(vectors) < 1:
        return np.zeros(vector_size)
    avg_vector = np.mean(vectors, axis=0)
    return avg_vector

# Get average word vectors for each paragraph pair
precily['word2vec_vector1'] = tokenized_text1.apply(lambda x: get_average_word_vectors(x, word2vec_model, 100))
precily['word2vec_vector2'] = tokenized_text2.apply(lambda x: get_average_word_vectors(x, word2vec_model, 100))

from sklearn.metrics.pairwise import cosine_similarity

# Assuming you have Word2Vec vectors for text1 and text2 in your DataFrame
# Convert the Word2Vec vectors to arrays for similarity calculation
vectors_text1 = np.array(precily['word2vec_vector1'].values.tolist())
vectors_text2 = np.array(precily['word2vec_vector2'].values.tolist())

# Calculate cosine similarity between the vectors
cosine_similarities = [cosine_similarity([vectors_text1[i]], [vectors_text2[i]])[0][0] for i in range(len(vectors_text1))]

# Assign similarity scores to the DataFrame
precily['similarity_score'] = cosine_similarities

import numpy as np
from sklearn.model_selection import train_test_split


# Combine the features for training
X = np.concatenate([precily['word2vec_vector1'].values.tolist(), precily['word2vec_vector2'].values.tolist()], axis=1)

y = precily['similarity_score'].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import LSTM, Dense

# LSTM model architecture
model = Sequential()
# Assuming features are input shape, adjusting units
model.add(LSTM(units=128, input_shape=(X_train.shape[1], 1)))
# Output layer for similarity score between 0 and 1  
model.add(Dense(units=1, activation='sigmoid'))  

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshaping the input data for LSTM (assuming features have a 2D shape)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# Train the model
model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), epochs=10, batch_size=32)

from sklearn.metrics import mean_squared_error

# Evaluate the model on the validation data
validation_evaluation = model.evaluate(X_val_reshaped, y_val)

# Print the evaluation metrics from model.evaluate() output
print(f"Validation Loss: {validation_evaluation[0]}")  # This would still be the loss value

# Make predictions on the validation data
val_predictions = model.predict(X_val_reshaped)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_val, val_predictions)

# Print the calculated MSE
print(f"Validation MSE: {mse}")

from flask import Flask, request, jsonify
import numpy as np  # Import necessary libraries for model usage and preprocessing

app = Flask(__name__)

# Load your trained model and necessary preprocessing functions here
# For example, load your model using pickle or another serialization method

# Define a route for your API endpoint
@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    # Get data from the request
    data = request.json
    text1 = data['text1']
    text2 = data['text2']

    # Preprocess the text data
    processed_new_text1 = preprocess_text(text1)
    processed_new_text2 = preprocess_text(text2)
    # Convert text to features (Word2Vec)
    word2vec_vector_new_text1 = get_average_word_vectors(processed_new_text1.split(" "), word2vec_model, 100)
    word2vec_vector_new_text2 = get_average_word_vectors(processed_new_text2.split(" "), word2vec_model, 100)
    # Perform similarity calculation using your trained model
    X_new = np.concatenate([word2vec_vector_new_text1.reshape(1, -1), word2vec_vector_new_text2.reshape(1, -1)], axis=1)
    X_new_reshaped = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))
    # Return the similarity score in the response
    similarity_score = float(model.predict(X_new_reshaped)[0][0])  # Replace with your model prediction function

    # Prepare the response body
    response_body = {"similarity score": similarity_score}

    return jsonify(response_body)



# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)  # Run your Flask app


