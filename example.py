import rerand
import numpy as np

x = np.random.normal(0, 1, 100)
dis = "Euclidean"
tol = 0.1
max_reps = 100

rand = rerand.Randomisation(x, dis, tol, max_reps=max_reps)
new_vec = rand.randomise()
print(new_vec)
