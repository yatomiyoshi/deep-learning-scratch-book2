import sys

import numpy as np
sys.path.append('..')
from common.util import cos_similarity, create_co_matrix, most_similar, ppmi, preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

W = ppmi(C)

np.set_printoptions(precision=3)
print(W)