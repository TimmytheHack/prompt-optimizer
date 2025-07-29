# validate.py

import random
from src.fitness import gsm8k_accuracy
from src.ga import PREFIX

# use full dev set (k=None) or a large sample
K = None       # or None to use entire file
N_TRY = 5
RAND = random.Random(42)

prompt = PREFIX
acc = gsm8k_accuracy(prompt, k=K, n_try=N_TRY, rand=RAND)
print(f"Validation → k={K}, n_try={N_TRY}, accuracy={acc:.3f}")
