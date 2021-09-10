import numpy as np

a = np.array([[1, 2, 3],
             [100, 200, 300],
             [100, 200, 300]])

# Center
print(np.array([[
    (max(a[:,0]) - min(a[:,0])) / 2,
    (max(a[:,1]) - min(a[:,1])) / 2,
    (max(a[:,2]) - min(a[:,2])) / 2
]])
)

# Dimensions
print(np.array([[
    abs((max(a[:,0]) - min(a[:,0]))),
    abs((max(a[:,1]) - min(a[:,1]))),
    abs((max(a[:,2]) - min(a[:,2])))
]])
)

dim = np.array([[
    abs((max(a[:,0]) - min(a[:,0]))),
    abs((max(a[:,1]) - min(a[:,1]))),
    abs((max(a[:,2]) - min(a[:,2])))
]])

print("--- DIMENSIONS")
print(dim)
print(dim[0][0])
print(dim[0][1])
print(dim[0][2])

print("--- TEST")
print(np.mean(a[:,0]))
print(np.mean(a, axis=0))       