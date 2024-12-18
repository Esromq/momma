import os
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer
from docx import Document
from dotenv import load_dotenv
import onnxruntime as ort
import numpy as np
import random

# Flask setup
app = Flask(__name__, template_folder='templates')

@app.route('/test')
def test():
    return "Flask is working!"

# Load environment variables
load_dotenv()

# Define the path to the folder containing doc files
folder_path = "./static/files"

# Load tokenizer and ONNX model
MODEL_NAME = "gpt2"  # Smaller model for iSH
ONNX_MODEL_PATH = "gpt2.onnx"  # Path to your exported ONNX model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# Load documents at startup
def load_documents(folder_path):
    """
    Load and parse all .docx files in the specified folder.
    """
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            file_path = os.path.join(folder_path, filename)
            doc = Document(file_path)
            content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            docs[filename] = content
            documents = content
    return docs

documents_content = load_documents(folder_path)
if not documents_content:
    print("No documents found in the folder.")
else:
    print(f"Loaded documents: {', '.join(documents_content.keys())}")

# Function to generate response using ONNX Runtime
def generate_response_with_onnx(input_text):
    """
    Generate text using ONNX model and tokenizer.
    """
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="np", max_length=1024, truncation=True)
    input_ids = inputs["input_ids"].astype(np.int64)  # Convert to ONNX-compatible numpy format

    # Run inference
    ort_inputs = {"input_ids": input_ids}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Decode the output
    output_ids = ort_outputs[0][0]  # Extract first sequence from ONNX output
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response

# Define chat function using ONNX Runtime
def chat_with_theresa(user_input, documents):
    """
    Generate a response using the ONNX model and document references.
    """
    if user_input.lower() == "what documents do you have access to?":
        # List all available documents
        document_list = "\n".join(f"- {name}" for name in documents.keys())
        return f"I have access to the following documents:\n{document_list}"
    
    # Randomly select a document to reference
    document_name, document_content = random.choice(list(documents.items()))
    prompt = f"""
    You are Theresa, the user's mother, a wise and nurturing figure. You have access to the following writings:
    ### {document_name}:
    {document_content[:1000]}  # Truncate for brevity

    Respond to the user's queries based on this content. Provide advice and wisdom as Theresa would.
    User says: {user_input}
    """
    
    # Generate response using ONNX
    response = generate_response_with_onnx(prompt)
    return response

# Flask route for serving the index.html page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route for user chat
@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint to interact with Theresa via API.
    Expects JSON input with a "user_input" key.
    """
    data = request.json
    user_input = data.get("user_input", "")
    if not user_input:
        return jsonify({"error": "No input provided."}), 400

    # Respond to user input using all document content
    response = chat_with_theresa(user_input, documents_content)
