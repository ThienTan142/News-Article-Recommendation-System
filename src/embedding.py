import sys
import os

# Thêm thư mục Project vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

class NewsEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[INFO] Loading SBERT model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode_news(self, texts, batch_size=32):
        """
        Encode news articles into embeddings.
        """
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch = texts[i:i + batch_size]
            batch_emb = self.model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(batch_emb)

        embeddings = np.vstack(all_embeddings)
        return embeddings
