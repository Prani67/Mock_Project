import os
import requests
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import uuid
import threading
import pickle
import time
import webbrowser
import threading
from dotenv import load_dotenv
load_dotenv()


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# File to store embeddings
EMBEDDINGS_FILE = 'embeddings.pkl'
QUESTIONS_CSV = 'New_Question_set_100.csv'

# Load questions from CSV at startup
questions_df = pd.read_csv(QUESTIONS_CSV,encoding='latin1')
questions_df.columns = questions_df.columns.str.strip()
preloaded_questions = questions_df.to_dict(orient='records')

# Vector database to store question embeddings
class VectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.load_from_disk()
        
    def add_documents(self, documents):
        # Check if documents are already embedded
        if self.documents and len(self.documents) > 0:
            existing_questions = set(doc['Question'] for doc in self.documents)
            new_docs = [doc for doc in documents if doc['Question'] not in existing_questions]
            
            if not new_docs:
                print("All documents already embedded. Using cached embeddings.")
                return
                
            print(f"Adding {len(new_docs)} new documents to existing {len(self.documents)}")
            self.documents.extend(new_docs)
            
            # Only embed the new documents
            texts = [f"{doc['Question']} {doc['Ideal Answer']} {doc['Keywords']}" for doc in new_docs]
            new_embeddings = model.encode(texts)
            
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            else:
                self.embeddings = new_embeddings
        else:
            # First time embedding
            self.documents = documents
            texts = [f"{doc['Question']} {doc['Ideal Answer']} {doc['Keywords']}" for doc in documents]
            self.embeddings = model.encode(texts)
        
        # Save updated embeddings to disk
        self.save_to_disk()
        
    def load_from_disk(self):
        """Load embeddings from disk if they exist"""
        if os.path.exists(EMBEDDINGS_FILE):
            print("Loading embeddings from disk...")
            try:
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.embeddings = data['embeddings']
                print(f"Loaded {len(self.documents)} documents with embeddings")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                self.documents = []
                self.embeddings = None
        
    def save_to_disk(self):
        """Save embeddings to disk"""
        try:
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings
                }, f)
            print(f"Saved {len(self.documents)} documents with embeddings")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
        
    def search(self, query, top_k=5):
        query_embedding = model.encode([query])[0]
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]

# Initialize vector database with preloaded questions
vector_db = VectorDB()
vector_db.add_documents(preloaded_questions)

# Store evaluation results
# Sessions will just be a mapping from session_id to a list of question dicts (filtered from preloaded_questions)
evaluation_results = {}
session_focus_data = {}
sessions = {}  # session_id: list of question dicts

# Mistral API configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-medium"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def evaluate_with_rag(question, user_answer, ideal_answer, keywords, question_id, session_id):
    # Retrieve relevant context from our vector database
    similar_questions = vector_db.search(question, top_k=3)
    
    # Format the context
    context = "\n\n".join([
        f"Similar Question: {q['Question']}\nIdeal Answer: {q['Ideal Answer']}\nKeywords: {q['Keywords']}"
        for q in similar_questions
    ])
    
    prompt = f"""
    You are an AI specialized in evaluating software testing interview responses.
    
    Evaluate the following interview response:
    
    Question: {question}
    User's Answer: {user_answer}
    Ideal Answer: {ideal_answer}
    Keywords to look for: {keywords}
    
    Additional context from similar questions:
    {context}
    
    Based on both the specific question and the relevant context, provide detailed evaluation in JSON format:
    {{
        "accuracy": 0-10 (how factually accurate the answer is),
        "relevance": 0-10 (how relevant the answer is to the question),
        "depth": 0-10 (how deep and thorough the answer is),
        "grammar": 0-10 (grammatical correctness),
        "confidence": 0-10 (how confidently the concepts were explained),
        "detailedFeedback": "provide specific, actionable feedback",
        "strengths": "identify 2-3 specific strengths in the answer",
        "improvements": "suggest 2-3 specific areas for improvement"
    }}
    
    When evaluating, consider industry best practices in software testing and align with the ideal answer.
    The response must be valid JSON that can be parsed.
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        
        # Ensure we have valid JSON (find JSON object in the response)
        result = result.strip()
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            result = result[start_idx:end_idx]
        
        evaluation = json.loads(result)
        
        # Store the evaluation result
        if session_id not in evaluation_results:
            evaluation_results[session_id] = {}
        
        evaluation_results[session_id][question_id] = {
            "question": question,
            "answer": user_answer,
            "evaluation": evaluation
        }
        
        return evaluation
    except Exception as e:
        print(f"Mistral API Error: {e}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        default_evaluation = {
            "accuracy": 5,
            "relevance": 5,
            "depth": 5,
            "grammar": 5,
            "confidence": 5,
            "detailedFeedback": "Error during evaluation. Please try again.",
            "strengths": "Attempted answer",
            "improvements": "Please retry"
        }
        
        # Store default evaluation on error
        if session_id not in evaluation_results:
            evaluation_results[session_id] = {}
        
        evaluation_results[session_id][question_id] = {
            "question": question,
            "answer": user_answer,
            "evaluation": default_evaluation
        }
        
        return default_evaluation

# Background evaluation to make the process non-blocking
def background_evaluate(question, user_answer, ideal_answer, keywords, question_id, session_id):
    try:
        evaluate_with_rag(question, user_answer, ideal_answer, keywords, question_id, session_id)
    except Exception as e:
        print(f"Background evaluation error: {e}")

@app.route('/api/questions', methods=['POST'])
def get_questions():
    data = request.get_json()
    domain = data['domain']
    qtype = data['questionType']
    count = int(data.get('count', 10))
    session_id = data.get('sessionId')
    if not session_id:
        session_id = str(uuid.uuid4())
    # Filter by domain and type from preloaded_questions
    filtered = [q for q in preloaded_questions if q['Domain'] == domain and q['Type'] == qtype]
    # Get unique questions only
    unique_questions = []
    seen_questions = set()
    for q in filtered:
        if q['Question'] not in seen_questions:
            seen_questions.add(q['Question'])
            unique_questions.append(q)
    available_count = len(unique_questions)
    return_count = min(count, available_count)
    # Store the questions for this session
    sessions[session_id] = unique_questions[:return_count]
    # Initialize focus tracking for this session
    if session_id not in session_focus_data:
        session_focus_data[session_id] = {
            "lastUpdated": time.time(),
            "focusEvents": [],
            "outOfFocusTime": 0
        }
    return jsonify({'sessionId': session_id, 'questions': unique_questions[:return_count]})

@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json()
    question = data['question']
    answer = data['answer']
    ideal = data['ideal']
    keywords = data['keywords']
    question_id = data['questionId']
    session_id = data['sessionId']
    is_final = data.get('isFinal', False)
    # Start evaluation in background
    threading.Thread(
        target=background_evaluate,
        args=(question, answer, ideal, keywords, question_id, session_id)
    ).start()
    # Return immediately without waiting for evaluation
    return jsonify({
        'status': 'success',
        'message': 'Answer submitted for evaluation'
    })

@app.route('/api/get-results', methods=['POST'])
def get_results():
    data = request.get_json()
    session_id = data['sessionId']
    question_ids = data['questionIds']
    results = {}
    for q_id in question_ids:
        if session_id in evaluation_results and q_id in evaluation_results[session_id]:
            results[q_id] = evaluation_results[session_id][q_id]
    return jsonify({
        'results': results,
        'completed': len(results) == len(question_ids)
    })

@app.route('/api/track-focus', methods=['POST'])
def track_focus():
    data = request.get_json()
    session_id = data['sessionId']
    event_type = data['eventType']  # 'focus' or 'blur'
    timestamp = time.time()
    
    if session_id not in session_focus_data:
        session_focus_data[session_id] = {
            "lastUpdated": timestamp,
            "focusEvents": [],
            "outOfFocusTime": 0
        }
    
    session_data = session_focus_data[session_id]
    
    # Record the event
    session_data["focusEvents"].append({
        "type": event_type,
        "timestamp": timestamp
    })
    
    # Calculate out of focus time if this is a focus event (returning to the window)
    if event_type == "focus" and len(session_data["focusEvents"]) > 1:
        # Find the most recent blur event
        for i in range(len(session_data["focusEvents"])-2, -1, -1):
            if session_data["focusEvents"][i]["type"] == "blur":
                blur_time = session_data["focusEvents"][i]["timestamp"]
                out_of_focus_duration = timestamp - blur_time
                session_data["outOfFocusTime"] += out_of_focus_duration
                break
    
    session_data["lastUpdated"] = timestamp
    
    return jsonify({
        'status': 'success',
        'outOfFocusTime': session_data["outOfFocusTime"]
    })

@app.route('/api/get-focus-data', methods=['POST'])
def get_focus_data():
    data = request.get_json()
    session_id = data['sessionId']
    
    if session_id not in session_focus_data:
        return jsonify({
            'status': 'error',
            'message': 'No focus data for this session'
        })
    
    focus_data = session_focus_data[session_id]
    
    return jsonify({
        'status': 'success',
        'outOfFocusTime': focus_data["outOfFocusTime"],
        'focusEvents': focus_data["focusEvents"]
    })

@app.route('/')
def serve_index():
    return send_from_directory('.', 'try2.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# Store cheating incidents data
cheating_incidents = {}

@app.route('/api/track-incident', methods=['POST'])
def track_incident():
    data = request.get_json()
    session_id = data['sessionId']
    incident_type = data['incidentType']
    timestamp = data.get('timestamp', time.time())
    question_index = data.get('questionIndex', -1)
    
    if session_id not in cheating_incidents:
        cheating_incidents[session_id] = []
    
    cheating_incidents[session_id].append({
        "type": incident_type,
        "timestamp": timestamp,
        "questionIndex": question_index
    })
    
    return jsonify({
        'status': 'success',
        'incidentCount': len(cheating_incidents[session_id])
    })

@app.route('/api/get-incidents', methods=['POST'])
def get_incidents():
    data = request.get_json()
    session_id = data['sessionId']
    
    if session_id not in cheating_incidents:
        return jsonify({
            'status': 'success',
            'incidents': []
        })
    
    return jsonify({
        'status': 'success',
        'incidents': cheating_incidents[session_id]
    })

if __name__ == '__main__':
    threading.Timer(1.25, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=False)
