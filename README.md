# PDF Query App

This application allows users to upload a PDF and query its contents. It uses a combination of NLP models and FAISS for efficient querying and text generation.

## Features

- Extract text from PDF files
- Chunk and index the text using FAISS
- Answer user queries based on the indexed content
- Provide context for the answers
- Interactive and colorful front end with loading animations

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sss7-7-7/pdf-query-app.git
   cd pdf-query-app

## File Descriptions

'app.py'

• The main application file that sets up the Flask web server and handles the core logic of the application:

• Extracts text from PDF files using 'PyMuPDF".

• Chunks the text to fit within the maximum sec sequence length of the models.

• Indexes the text chunks using FAISS for efficient similarity search.

• Answers user queries by finding the most relevant text chunk and generating a response using the text generation model.

## Models Used

### Sentence Transformer

The application uses the `all-MiniLM-L6-v2 model from Hugging Face's Sentence Transformers library to generate embeddings for the text chunks and user queries. This model is efficient and provides high-quality sentence embeddings.

• Model Name: `all-MiniLM-L6-v2

• Usage: Generating embeddings for text chunks and user queries.

### Text Generation Model

The application uses the 'google/flan-t5-small model from Hugging Face's Transformers library for text generation. This model is used to generate answers based on the retrieved text chunks.

• Model Name: google/flan-t5-small

• Usage: Generating responses to user queries based on the most relevant text chunk.

## Requirements
• Python 3.7+
• Flask
• sentence-transformers
• faiss-cpu
• transformers
• PyMuPDF
• torch
