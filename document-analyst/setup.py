from sentence_transformers import SentenceTransformer
import nltk

# Define the model name and path
model_name = 'all-MiniLM-L6-v2'
model_path = './models/all-MiniLM-L6-v2'

# Download and save the model to the specified path
print("Downloading sentence-transformer model...")
model = SentenceTransformer(model_name)
model.save(model_path)
print(f"Model saved to {model_path}")

# Download the NLTK tokenizers needed by sumy
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('punkt_tab') # <-- ADD THIS LINE

print("NLTK resources downloaded. You can now run the main script offline.")
