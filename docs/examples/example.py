from rerand.Randomisation import Randomisation
import numpy as np

x = np.random.normal(0, 1, 100)
dis = "Euclidean"
tol = 0.1
max_reps = 100

variants = {"a": 0.5, "b": 0.3, "c": 0.2}

rand = Randomisation(x, dis, tol, max_reps=max_reps, variants=variants)
new_vec = rand.randomise()
print(new_vec)
