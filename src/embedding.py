import numpy as np
from sentence_transformers import SentenceTransformer
import tqdm

class NewsEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[INFO] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode_news(self, texts):
        """
        texts: list of cleaned news text
        return: numpy array of shape (num_news, 384)
        """
        print(f"[INFO] Generating embeddings for {len(texts)} articles...")
        vectors = []

        for text in tqdm(texts):
            vec = self.model.encode(text)
            vectors.append(vec)

        return np.array(vectors)
