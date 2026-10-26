"""Download and cache embedding models for offline demo use.

Task 9: Ensure sentence-transformers model works offline at demo time.
"""
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = Path(__file__).parent.parent / ".cache" / "sentence-transformers"

def download_models():
    """Download and cache embedding models locally."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE_DIR))

    # Test encoding
    test_text = "Store floor is dirty"
    embedding = model.encode(test_text)

    print(f"✓ Model downloaded and cached to {CACHE_DIR}")
    print(f"✓ Test encoding successful: {len(embedding)} dimensions")
    print("\nModel is ready for offline use.")

    return model

if __name__ == "__main__":
    download_models()
