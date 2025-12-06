import numpy as np
arr = np.load("data/embeddings/news_embeddings.npy", allow_pickle=True)
print(arr.dtype)
print(arr.shape)