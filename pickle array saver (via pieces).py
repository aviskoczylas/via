import pickle

# format is [x length, y length, height]
#NO 2 PIECES CAN BE IDENTICAL! 
# If you want to use identical pieces, it will be better performance-wise to change the sum(piece_vars)<=1 constraint in the solver anyways.
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
