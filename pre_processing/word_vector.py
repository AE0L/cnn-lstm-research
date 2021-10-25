import numpy as np
import tqdm
# from scipy import spatial
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

EMBEDDING_VECTOR_LENGTH = 50


def construct_embedding_matrix(glove_file, input_word_index):
    embedding_dict = {}

    # read glove file
    with open(glove_file, 'r'):
        for line in f:
            values = line.split()
            word = values[0]
            if word in input_word_index.keys():
                vector = np.asarray(values[1:], 'float32')
                embedding_dict[word] = vector

    num_words = len(input_word_index) + 1
    # initiate matrix with zeroes
    embedding_matrix = np.zeros((num_words, EMBEDDING_VECTOR_LENGTH))

    for word, i in tqdm.tqdm(input_word_index.items()):
        if i < num_words:
            vect = embedding_dict.get(word, {})
            if len(vect) > 0:
                embedding_matrix[i] = vect[:EMBEDDING_VECTOR_LENGTH]

    return embedding_matrix

# def find_closest_embeddings(embedding):
#     return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

# tsne = TSNE(n_components=2, random_state=0)
# words = list(embeddings_dict.keys())
# vectors = [embeddings_dict[word] for word in words]

# Y = tsne.fit_transform(vectors[:100])
# plt.scatter(Y[:, 0], Y[:, 1])

# for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

# plt.show()
