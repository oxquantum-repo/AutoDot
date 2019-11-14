import numpy as np

import numpy as np

num_active_gates = 8
dvec = np.array([100, 0, 100, 100, 100, 100, 70, 100])
priority = np.array([2, 2, 1, 2, 2, 2, 1, 2])
names = ['c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']

unique_priority = np.unique(priority)
priority_pairs = [(i,j) for i in unique_priority for j in unique_priority if i < j]
print(priority_pairs)

for ppair in priority_pairs:
    idx_pairs = [(i,j) for i in range(num_active_gates) for j in range(num_active_gates) if priority[i] == ppair[0] and priority[j] == ppair[1]]
    name_pairs = [(names[i], names[j]) for (i,j) in idx_pairs]
    dpairs = [(dvec[i], dvec[j]) for (i,j) in idx_pairs]
    print(ppair)
    print(name_pairs)
    print(dpairs)

    violation_all = [ np.maximum(a-b,0.) for (a,b) in dpairs]
    print(np.sum(violation_all))
    print(np.sqrt(np.sum(np.square(violation_all))))

def priority_penalty(priority, dall
# speed optimization
# priority penalty
# separate gp for each d
# non-pinchoff penalty (for each d)
