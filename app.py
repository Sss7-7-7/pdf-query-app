from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import fitz  # PyMuPDF
import os

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATION_MODEL_NAME = 'google/flan-t5-small'
K = 5  # Number of nearest neighbors to retrieve
MAX_SEQ_LENGTH = 512  # Maximum sequence length for the model

# Initialize models
model = SentenceTransformer(MODEL_NAME)
generation_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
generator = pipeline('text2text-generation', model=generation_model, tokenizer=tokenizer)

app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return texts

def chunk_text(text, max_length):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding=True).input_ids[0]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        start = end
    return chunks

def create_faiss_index(texts):
    embeddings = []
    chunks = []
    for text in texts:
        text_chunks = chunk_text(text, MAX_SEQ_LENGTH)
        chunks.extend(text_chunks)
        embeddings.extend(model.encode(text_chunks))
    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

def answer_query(user_query, index, chunks):
    query_embedding = model.encode([user_query])
    D, I = index.search(query_embedding, 1)
    retrieved_doc = chunks[I[0][0]]
    context = retrieved_doc
    input_text = f"Answer the question based on the following context: {context}\nQuestion: {user_query}"
    response = generator(input_text, max_length=150, num_return_sequences=1)
    answer = response[0]['generated_text'].strip()
    return answer, context

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        pdf_path = os.path.join('/tmp', file.filename)
        file.save(pdf_path)
        query = request.form['query']

        try:
            pdf_texts = extract_text_from_pdf(pdf_path)
            index, chunks = create_faiss_index(pdf_texts)
            answer, context = answer_query(query, index, chunks)
            return jsonify({"answer": answer, "context": context})
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
