import pickle

# format is [x length, y length, height]
#Note, things may get messed up if there are any identical pieces
pieces = [
    [2,1,3],
    [2,1,1],
    [1,0,1],
    [2,0,1],
    [2,0,2],
    [2,2,1],
    [1,1,1],
    [3,1,2],
    [1,0,2],
    [3,0,1]
]
with open('via_pieces.pkl', 'wb') as f:
    pickle.dump(pieces, f)
