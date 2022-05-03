from vec2word import vec2word
import numpy as np

a = [1, 2, 3, 5, 1, 2, 4, 5]
a = np.array(a).reshape(2, -1).astype(np.int32)
b = ['a', 'b', 'c', 'd', 'e']
b = np.array(b)
s = [1, 0.1, 0.2, 0.5, 1, 1, 1, 5]
s = np.array(s).reshape(2, -1)
s = np.array(s)

print(vec2word(a, s, b, 5))