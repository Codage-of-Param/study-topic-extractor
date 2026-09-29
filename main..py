import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from google.colab import files
import re # Added for sentence tokenization
import requests # For downloading PDFs from URLs
import tempfile # For creating temporary files for downloaded PDFs

class MainTopicDetector:
    def __init__(self):
        pass

    def identify_topic(self, text: str) -> str:
        """Analyzes text to deduce the primary overarching topic/subject."""
        if not text.strip():
            return "No text provided."

        # Use TF-IDF with bigrams and trigrams to find prominent descriptive phrases
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(2, 4),  # Focus on multi-word phrases
            max_features=5
        )

        try:
            X = vectorizer.fit_transform([text])
            phrases = vectorizer.get_feature_names_out()

            # Formulate a clean topic statement based on top phrases
            if len(phrases) > 0:
                # Capitalize and join top concepts
                primary_theme = " / ".join([p.title() for p in phrases[:3]])
                return primary_theme
            return "General Text"
        except Exception as e:
            return f"Could not determine topic: {e}"

    def get_summary(self, text: str, num_sentences: int = 3) -> str:
        """Extracts the first few sentences as a summary."""
        if not text.strip():
            return "No summary available."

        # Step 1: Normalize newlines and other whitespace
        # Replace multiple newlines (paragraph breaks) with a period and space to ensure sentence splitting
        cleaned_text = re.sub(r'\n\s*\n+', '. ', text)
        # Replace single newlines (line breaks within paragraphs) with a space
        cleaned_text = re.sub(r'\n', ' ', cleaned_text)
        # Normalize any remaining excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Step 2: Split into sentences using a common regex
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

        # Step 3: Filter out empty strings and very short/non-informative sentences
        # (e.g., page numbers, single words, or just a few characters)
        sentences = [s.strip() for s in sentences if s.strip() and (len(s.split()) > 4 or re.search(r'[a-zA-Z]{5,}', s))]

        if not sentences:
            return "No meaningful sentences for summary."

        summary_sentences = sentences[:num_sentences]
        summary = ' '.join(summary_sentences)
        if len(sentences) > num_sentences:
            summary += '...'
        return summary


    def process_pdf(self, pdf_path: str, max_size_mb: int = 50) -> str:
        """Extracts text from a PDF, handling size limits."""
        if not os.path.exists(pdf_path):
            return f"Error: File not found: {pdf_path}"

        # Add a simple check for file extension as an initial filter
        if not pdf_path.lower().endswith('.pdf'):
            return f"Error: '{pdf_path}' does not appear to be a PDF file. Please upload an actual .pdf document."

        # Check file size
        file_size_bytes = os.path.getsize(pdf_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return f"Error: PDF file size ({file_size_mb:.2f} MB) exceeds the maximum allowed limit of {max_size_mb} MB. Please upload a smaller PDF."

        full_text = ""
        try:
            reader = PdfReader(pdf_path)
            # Read all pages now
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        except Exception as e:
            return f"Error reading PDF: {e}. This might be due to a corrupted, encrypted, or malformed PDF. Please try another PDF."

        if not full_text.strip():
            return "Error: Could not extract any readable text from the PDF."

        return full_text # Return the full text for external processing

def download_pdf_from_url(url, output_path, max_size_mb=50):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Check content type
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type:
            return f"Error: URL does not point to a PDF. Content-Type: {content_type}"

        # Check content length if available
        content_length = int(response.headers.get('Content-Length', 0))
        if content_length > max_size_mb * 1024 * 1024:
            return f"Error: PDF from URL exceeds maximum allowed size of {max_size_mb} MB."

        with open(output_path, 'wb') as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    pdf_file.write(chunk)
        return output_path

    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF from URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred during download: {e}"

# ==========================================
# Example Usage & Custom Inputs
# ==========================================
if __name__ == "__main__":
    detector = MainTopicDetector()

    print("--- Custom Option Selection ---")
    choice = input("Enter '1' for custom text, '2' to upload a PDF, or '3' to provide a PDF URL: ").strip()

    if choice == '1':
        custom_text = input("\nEnter or paste your custom text here: ").strip()
        if custom_text:
            print("\n--- Analyzing Custom Text Topic ---")
            topic = detector.identify_topic(custom_text)
            summary = detector.get_summary(custom_text)
            print(f"Detected Main Topic: {topic}")
            print(f"Summary: {summary}")
        else:
            print("No text was entered.")

    elif choice == '2':
        # 2. Upload and load custom PDF
        print("\n--- Upload Your Custom PDF Document ---")
        uploaded = files.upload()

        if uploaded:
            # Get the first uploaded file name
            custom_pdf_name = list(uploaded.keys())[0]
            print(f"\nProcessing uploaded file: {custom_pdf_name}")

            # Process PDF to get full text
            pdf_content_or_error = detector.process_pdf(custom_pdf_name, max_size_mb=20) # Set a default max size

            if pdf_content_or_error.startswith("Error:"):
                print(pdf_content_or_error)
            else:
                full_text = pdf_content_or_error
                pdf_topic = detector.identify_topic(full_text)
                pdf_summary = detector.get_summary(full_text)
                print(f"Detected Main Topic of Uploaded PDF: {pdf_topic}")
                print(f"Summary: {pdf_summary}")
        else:
            print("No file uploaded. Checking for default file...")
            pdf_file_path = "sample_study_material.pdf"
            if os.path.exists(pdf_file_path):
                print(f"Processing default file: {pdf_file_path}")
                pdf_content_or_error = detector.process_pdf(pdf_file_path, max_size_mb=20)
                if pdf_content_or_error.startswith("Error:"):
                    print(pdf_content_or_error)
                else:
                    full_text = pdf_content_or_error
                    pdf_topic = detector.identify_topic(full_text)
                    pdf_summary = detector.get_summary(full_text)
                    print(f"Detected Main Topic of Default PDF: {pdf_topic}")
                    print(f"Summary: {pdf_summary}")
            else:
                print("Please upload a PDF to analyze its topic.")

    elif choice == '3':
        print("\n--- Download and Analyze PDF from URL ---")
        pdf_url = input("Enter the URL of the PDF document: ").strip()
        if pdf_url:
            # Create a temporary file to save the downloaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
                temp_file_path = temp_pdf_file.name

            print(f"Attempting to download PDF from: {pdf_url}")
            download_result = download_pdf_from_url(pdf_url, temp_file_path, max_size_mb=50) # Increased max size for URL downloads

            if download_result.startswith("Error:"):
                print(download_result)
            else:
                print(f"Successfully downloaded PDF to: {download_result}")
                pdf_content_or_error = detector.process_pdf(download_result, max_size_mb=50)

                if pdf_content_or_error.startswith("Error:"):
                    print(pdf_content_or_error)
                else:
                    full_text = pdf_content_or_error
                    pdf_topic = detector.identify_topic(full_text)
                    pdf_summary = detector.get_summary(full_text)
                    print(f"Detected Main Topic of URL PDF: {pdf_topic}")
                    print(f"Summary: {pdf_summary}")
            
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
        else:
            print("No URL was entered.")

    else:
        print("\nInvalid choice. Running default paragraph analysis as a fallback...")
        sample_paragraph = """
        Quantum computing is an emerging technology that harnesses the laws of quantum mechanics
        to solve problems too complex for classical computers. By utilizing qubits, which can exist
        in multiple states simultaneously through superposition and entanglement, quantum computers
        promise revolutionary breakthroughs in cryptography, material science, and molecular modeling.
        """
        print("--- Paragraph Topic Analysis ---")
        topic = detector.identify_topic(sample_paragraph)
        summary = detector.get_summary(sample_paragraph)
        print(f"Detected Main Topic: {topic}")
        print(f"Summary: {summary}")