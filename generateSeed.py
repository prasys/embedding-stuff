import pickle
import random 
import sys

seedVal = []



for x in range(10):
    seed = random.randrange(2**32-1)
    rng = random.Random(seed)
    print(seed)
    seedVal.append(seed)

with open('randSeed.pkl', 'wb') as f:
    pickle.dump(seedVal, f)




