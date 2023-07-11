import sys
sys.path.append('..')
from common.util import convert_one_hot, cos_similarity, create_co_matrix, create_contexts_target, most_similar, preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)

contexts, target = create_contexts_target(corpus, window_size=1)

target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

print(contexts[:, 0].shape)