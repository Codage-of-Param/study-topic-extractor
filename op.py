import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from google.colab import files
import re
import requests
import tempfile
import numpy as np

# Ensure sentence-transformers is installed for MiniLM embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    os.system("pip install -q sentence-transformers")
    from sentence_transformers import SentenceTransformer

class MainTopicDetector:
    def init(self):
        # Initialize the MiniLM model for semantic sentence embeddings
        print("Loading MiniLM model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def identify_topic(self, text: str) -> str:
        """Analyzes text to deduce the primary overarching topic/subject."""
        if not text.strip():
            return "No text provided."

        # Use TF-IDF with bigrams and trigrams to find prominent descriptive phrases
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(2, 4),
            max_features=5
        )

        try:
            X = vectorizer.fit_transform([text])
            phrases = vectorizer.get_feature_names_out()

            if len(phrases) > 0:
                primary_theme = " / ".join([p.title() for p in phrases[:3]])
                return primary_theme
            return "General Text"
        except Exception as e:
            return f"Could not determine topic: {e}"

    def get_summary(self, text: str, num_sentences: int = 3) -> str:
        """Extracts an advanced semantic summary using MiniLM embeddings and returns a formatted string with bullet points."""
        if not text.strip():
            return "• No summary available."

        # Step 1: Normalize newlines and whitespace
        cleaned_text = re.sub(r'\n\s*\n+', '. ', text)
        cleaned_text = re.sub(r'\n', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Step 2: Split into sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

        # Step 3: Filter out empty strings and very short/non-informative sentences
        sentences = []
        original_indices = []
        for i, s in enumerate(raw_sentences):
            s_stripped = s.strip()
            if s_stripped and (len(s_stripped.split()) > 4 or re.search(r'[a-zA-Z]{5,}', s_stripped)):
                sentences.append(s_stripped)
                original_indices.append(i)

        if not sentences:
            return "• No meaningful sentences for summary."

        # Limit sentence count if text is exceptionally long to keep embedding fast
        max_sentences_to_process = 100
        if len(sentences) > max_sentences_to_process:
            sentences = sentences[:max_sentences_to_process]
            original_indices = original_indices[:max_sentences_to_process]

        try:
            # Step 4: Compute MiniLM embeddings for all sentences
            embeddings = self.model.encode(sentences, show_progress_bar=False)

            # Step 5: Rank sentences based on their similarity to the average document representation
            doc_embedding = np.mean(embeddings, axis=0, keepdims=True)

            # Compute cosine similarities between each sentence and the average document vector
            norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            norm_doc = doc_embedding / np.linalg.norm(doc_embedding)
            scores = np.dot(norm_embeddings, norm_doc.T).flatten()

            # Step 6: Select and sort top sentences chronologically
            scored_sentences_info = [
                {'sentence': sentences[i], 'score': scores[i], 'original_idx': original_indices[i]}
                for i in range(len(sentences))
            ]

            # Sort descending by score
            scored_sentences_info.sort(key=lambda x: x['score'], reverse=True)
            top_n_sentences_info = scored_sentences_info[:num_sentences]

# Sort back to chronological order
            top_n_sentences_info.sort(key=lambda x: x['original_idx'])

            # Join elements as a formatted string with bullet points
            bullet_summary = "\n".join([f"• {info['sentence']}" for info in top_n_sentences_info])
            return bullet_summary
        except Exception as e:
            return f"• Error generating MiniLM summary: {e}"

    def process_pdf(self, pdf_path: str, max_size_mb: int = 50) -> str:
        """Extracts text from a PDF, handling size limits."""
        if not os.path.exists(pdf_path):
            return f"Error: File not found: {pdf_path}"

        if not pdf_path.lower().endswith('.pdf'):
            return f"Error: '{pdf_path}' does not appear to be a PDF file."

        file_size_bytes = os.path.getsize(pdf_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return f"Error: PDF exceeds allowed limit."

        full_text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        except Exception as e:
            return f"Error reading PDF: {e}"

        if not full_text.strip():
            return "Error: Could not extract any readable text."

        return full_text

def download_pdf_from_url(url, output_path, max_size_mb=50):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type:
            return f"Error: URL does not point to a PDF."

        with open(output_path, 'wb') as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    pdf_file.write(chunk)
        return output_path
    except Exception as e:
        return f"Error downloading PDF: {e}"

if name == "main":
    detector = MainTopicDetector()

    print("--- Custom Option Selection ---")
    choice = input("Enter '1' for custom text, '2' to upload a PDF, or '3' to provide a PDF URL: ").strip()

    if choice == '1':
        custom_text = input("\nEnter or paste your custom text here: ").strip()
        if custom_text:
            print("\n--- Analyzing Custom Text Topic ---")
            topic = detector.identify_topic(custom_text)
            summary = detector.get_summary(custom_text)
            print(f"\nDetected Main Topic: {topic}")
            print("\nKey Summary Points:")
            print(summary)
        else:
            print("No text entered.")

    elif choice == '2':
        print("\n--- Upload Your Custom PDF Document ---")
        uploaded = files.upload()

        if uploaded:
            custom_pdf_name = list(uploaded.keys())[0]
            print(f"\nProcessing: {custom_pdf_name}")
            pdf_content_or_error = detector.process_pdf(custom_pdf_name)
            if pdf_content_or_error.startswith("Error:"):
                print(pdf_content_or_error)
            else:
                full_text = pdf_content_or_error
                pdf_topic = detector.identify_topic(full_text)
                pdf_summary = detector.get_summary(full_text)
                print(f"\nDetected Main Topic: {pdf_topic}")
                print("\nKey Summary Points:")
                print(pdf_summary)
        else:
            print("No file uploaded.")

    elif choice == '3':
        print("\n--- Download and Analyze PDF from URL ---")
        pdf_url = input("Enter the URL of the PDF document: ").strip()
        if pdf_url:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
                temp_file_path = temp_pdf_file.name
download_result = download_pdf_from_url(pdf_url, temp_file_path)
            if download_result.startswith("Error:"):
                print(download_result)
            else:
                pdf_content_or_error = detector.process_pdf(temp_file_path)
                if pdf_content_or_error.startswith("Error:"):
                    print(pdf_content_or_error)
                else:
                    full_text = pdf_content_or_error
                    pdf_topic = detector.identify_topic(full_text)
                    pdf_summary = detector.get_summary(full_text)
                    print(f"\nDetected Main Topic: {pdf_topic}")
                    print("\nKey Summary Points:")
                    print(pdf_summary)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        else:
            print("No URL was entered.")
    else:
        print("Invalid choice.")