import numpy as np

#  Encode a column of textual utterances into dense vector representations using SentenceTransformer
def useSentenceTransformer(dataframe_normalized, model):
    df_teste = dataframe_normalized.copy()

    # Transform the 'utterance' column into a list of sentences
    sentences = df_teste['utterance'].tolist()

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    vectors = np.array(embeddings)
    print("Shape: ", vectors.shape)
    print("--> sentence-transformers used with success!")
    
    return vectors

