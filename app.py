from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import io
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import InvalidCollectionException
import json
from docx import Document
import re

app = Flask(__name__)

# Constants
API_KEY = 'sk-proj-gLCCi6Z5NKybkm2HNwewT3BlbkFJOopee8WNBRmul7lhbHK8'
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4-turbo-preview"
COLLECTION_NAME = "questions_collection"

# Set up OpenAI
openai.api_key = API_KEY

# Initialize ChromaDB client and embedding function
chroma_client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name=EMBEDDING_MODEL
)

def get_embedding(text, model=EMBEDDING_MODEL):
    """Get the embedding for a given text."""
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def get_or_create_collection(chroma_client, collection_name):
    """Get an existing collection or create a new one if it doesn't exist."""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
        is_new = False
    except InvalidCollectionException:
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
        is_new = True
    
    collection.embedding_function = openai_ef
    return collection, is_new

def add_questions_to_db(df, collection):
    """Add questions and answers to the ChromaDB collection."""
    for index, row in df.iterrows():
        question = row['Question']
        answer = row['Answer_Response']
        embedding = get_embedding(question)
        
        # Add to ChromaDB
        collection.add(
            documents=[question],
            embeddings=[embedding],
            metadatas=[{"answer": answer}],
            ids=[f"question_{index}"]
        )
    print("Questions added to ChromaDB successfully!")

def find_similar_questions(query, collection, n=5):
    """Find the n most similar questions to the query using ChromaDB."""
    query_emb = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n,
        include=["documents", "metadatas"]
    )
    
    similar_questions_dict = {}
    for i in range(len(results['documents'][0])):
        question = results['documents'][0][i]
        answer = results['metadatas'][0][i].get('answer', 'No answer available')
        similar_questions_dict[question] = answer

    return similar_questions_dict

def find_most_suitable_response(query, r_list, num_questions=10):
    """Find the most suitable response using GPT-4."""
    response_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(r_list)])
    prompt = f"""You are a text processing agent working with request for proposal documents within the education sector.
    
    Your task is to find the most suitable response to the given query from the list of responses provided.
    Consider semantic meaning, context, and number of key concepts when determining similarity. 

    Query: {query}
    
    List of responses:
    {response_text}
    
    Please return your response as a JSON object with the following structure:
    {{
        "most_suitable_answer": "The full original text of the most similar question.",
        "similarity_score": A number between 0 and 1 indicating the similarity (1 being identical, 0 being completely different),
        "reasoning": "A brief explanation of why this question was chosen as the most similar"
    }}
    
    Ensure your response is a valid JSON object.
    """
    
    completion = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return completion.choices[0].message.content

def parse_json_string(json_string):
    """Parse a JSON string that's wrapped in triple backticks and a language identifier."""
    clean_json_string = json_string.strip().removeprefix("```json\n").removesuffix("```")
    try:
        json_data = json.loads(clean_json_string)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def remove_bullet_points(text):
    """Remove bullet points and any leading whitespace."""
    return re.sub(r'^[\s*]*[a-zA-Z0-9]+\.\s*', '', text)

def process_csv_file(file):
    """Process the uploaded CSV file."""
    # Read the uploaded CSV file
    queries = pd.read_csv(file)

    # Load QA pairs (you might need to adjust this part based on your setup)
    qa_df = pd.read_csv('path/to/your/qa_pairs.csv', usecols=['Question', 'Answer_Response'])
    qa_df = qa_df.dropna(subset=['Question', 'Answer_Response'])
    qa_df['Question'] = qa_df['Question'].apply(remove_bullet_points)

    # Get or create ChromaDB collection
    collection, is_new_collection = get_or_create_collection(chroma_client, COLLECTION_NAME)

    # Add questions to ChromaDB if it's a new collection
    if is_new_collection:
        add_questions_to_db(qa_df, collection)
        print("New collection created and questions added to ChromaDB successfully!")
    else:
        print("Using existing collection. No new questions added.")

    qa_pairs = {}
    for index, query in queries.iterrows():
        print(f"Query {index + 1}: {query.iloc[0]}")
        similar_questions_dict = find_similar_questions(query.iloc[0], collection)
        similar_questions_ans = list(similar_questions_dict.values())
        response = find_most_suitable_response(query.iloc[0], similar_questions_ans)
        response = parse_json_string(response)
        most_suitable_response = response['most_suitable_answer']
        qa_pairs[query.iloc[0]] = {
            'answer': most_suitable_response,
            'suggestions': similar_questions_ans[:3]  # Include top 3 suggestions
        }
    
    return qa_pairs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        results = process_csv_file(file)
        return jsonify(results)
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/save', methods=['POST'])
def save_word():
    content = request.json.get('content')
    doc = Document()
    doc.add_paragraph(content)
    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return send_file(file_stream, as_attachment=True, download_name='results.docx', mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')

if __name__ == '__main__':
    app.run(debug=True)