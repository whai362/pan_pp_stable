import numpy as np
import cv2
import torch
cimport numpy as np
cimport cython
cimport libcpp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _vec2word(np.ndarray seqs,
               np.ndarray seq_scores,
               np.ndarray id2char,
               int eos):
    EPS = 1e-6
    num_words = seqs.shape[0]

    cdef list words = ['' for _ in range(num_words)]
    cdef np.ndarray[np.float32_t, ndim=1] word_scores = np.zeros((num_words,), dtype=np.float32)

    cdef str tmp_word
    cdef str tmp_char
    cdef float tmp_score
    for i in range(num_words):
        tmp_word = ''
        tmp_score = 0
        for j, char_id in enumerate(seqs[i, :]):
            if char_id == eos:
                break
            tmp_char = str(id2char[char_id])
            if tmp_char in ['PAD', 'UNK']:
                continue
            tmp_word += tmp_char
            tmp_score += seq_scores[i, j]
        words[i] = tmp_word
        word_scores[i] = tmp_score / (len(tmp_word) + EPS)

    return words, word_scores

def vec2word(seqs, seq_scores, id2char, eos):
    return _vec2word(seqs, seq_scores, id2char, eos)
