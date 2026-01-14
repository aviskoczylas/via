import time
start = time.time()
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
import os

puzzle_num = 67
write_to_file = 0
puzzle_path = f"sample_puzzles/type_a/puzzle{puzzle_num}.png"
sol_path = f"sample_solutions/type_a/sol{puzzle_num}"
if write_to_file: #don't allow previous puzzles to be overwritten
    if os.path.exists(puzzle_path) or os.path.exists(sol_path):
        print("Error - path already exists.")
        assert(False)
    else:
        os.makedirs(sol_path)

#Maximum z layer that pieces can occupy. Lower makes solver much faster, but if too low, may miss possible solutions
z_max = 5
#this solver sometimes has trouble proving infeasability, so the solver aborts if it can't find find a solution within a reasonable time
max_solve_time = 1800

#generate all possible orientations for the pieces
def rotate(piece):
    orientations = [
        [piece[0], piece[1], piece[2]],
        [-piece[0], piece[1], piece[2]],  
        [piece[1], piece[0], piece[2]],
        [-piece[1], piece[0], piece[2]],
    ]
    unique_orientations = []
    for orientation in orientations:
        if not any(np.array_equal(orientation, existing) for existing in unique_orientations):
            unique_orientations.append(orientation)
    return unique_orientations


# load pieces from a file.
with open('via_pieces.pkl', 'rb') as file:
    pieces = pickle.load(file)

#define board shape
board_rows = 5
board_cols = 5

#define which spots are considered terminals
t = [(0,i) for i in range(0,board_cols-1)]
b = [(board_rows-1,i) for i in range(1,board_cols)]
l = [(i,0) for i in range(1,board_rows-1)]
r = [(i,board_cols-1) for i in range(1,board_rows-1)]
terminals = t+r+b[::-1]+l[::-1] #clockwise
#for this particular setup, these are the terminal indices
#     0  1 2 3 x
#     13 x x x 4
#     12 x x x 5
#     11 x x x 6
#     x 10 9 8 7

active_terminals = [

    (terminals[1], terminals[3],terminals[4],terminals[6],terminals[7],terminals[8],terminals[9],terminals[10]),
]
                    
elevated_terminals = [terminals[12], terminals[0],terminals[2], terminals[4],terminals[6]]

#specify starting pieces as [piece, row, col], for example [(rotate(pieces[0])[0],0,0)]
#note that starting pieces can only be placed on the floor
starting_pieces = [
    [(rotate(pieces[0])[3],2,2)]
]   
for piece in pieces:
    print(rotate(piece))
    print("\n")

#generate all possible placements of a piece on the board as a list of [orientation, x, y]
#note that a "placement" on the board does specify an orientation, and layer is not included in this list
def possible_placements(piece):
    placements = []
    for orientation in rotate(piece):
        x_len, y_len = orientation[0], orientation[1]
        for row in range(board_rows):
            for col in range(board_cols):
                if row+y_len >= 0 and row+y_len < board_rows and col+x_len >= 0 and col+x_len < board_cols:
                    placements.append((orientation, row, col))
    return placements

#given the solution variables and placements, construct nested list to represent the solution.
def create_sol_grid(placements_list, piece_vars_list, solver):
    solution_grid = []
    for row in range(board_rows):
        row_list = []
        for col in range(board_cols):
            row_list.append([])
        solution_grid.append(row_list)
    for i, piece_placements in enumerate(placements_list):
        for j, (piece, row, col) in enumerate(piece_placements):
            x_len, y_len = piece[0], piece[1]
            if solver.Value(piece_vars_list[i][j]):
                solution_grid[row][col].append(i)
                solution_grid[row+y_len][col+x_len].append(i)
    if write_to_file:
            plot_grid(solution_grid, solver)
            plot_grid(solution_grid, solver, use_layer_view=True)
    else:
        plot_grid(solution_grid, solver)

size = 800
def plot_grid(solution_grid, solver = None, use_layer_view=False):
    plotted_pieces = [-1] 

    max_z = 0
    if use_layer_view:
        #this loop is to calculate how many 2d plots to generate
        if solver is not None:
            for i, piece_vars in enumerate(piece_vars_list):
                for var in piece_vars:
                    if solver.Value(var):
                        top = solver.Value(z_bottom_vars[i]) + piece_heights[i]
                        if top > max_z:
                            max_z = top    
    #loop once for each layer if 2d, or once if 3d.
    for current_layer in range(use_layer_view*max_z+1):
        plot_terminals()
        for row in range(board_rows):
            for col in range(board_cols):
                plt.scatter(col, -row, s = 1, c="black")

        for row in range(board_rows):
            for col in range(board_cols):
                if solver is not None:
                    node_id = solver.Value(node_ids[row*board_cols+col])
                    #Color the pieces according to the circuit they are part of
                else:
                    node_id = -1

                for piece_id in solution_grid[row][col]:
                    if piece_id not in plotted_pieces: #make sure each piece is only being plotted once
                        if solver is not None:
                            layer = solver.Value(z_bottom_vars[piece_id])+piece_heights[piece_id]
                        else:
                            layer = piece_heights[piece_id] #starting pieces can only be placed on the floor!
                        #find the other end of the piece we want to plot
                        match_row, match_col = find_match(piece_id, row, col, solution_grid) 
                        if use_layer_view:
                            if layer == current_layer:
                                plot_piece_2d(row, col, match_row, match_col, piece_heights[piece_id], node_id)
                                plotted_pieces.append(piece_id)
                        else:
                            plot_piece_3d(row, col, match_row, match_col, layer, piece_heights[piece_id], node_id)
                            plotted_pieces.append(piece_id)

        plt.xlim(-1.5, board_cols+0.5)
        plt.ylim(-board_rows-0.5,1.5)
        plt.gca().set_aspect(1, adjustable='box')
        plt.tight_layout()
        plt.gca().tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if write_to_file:
            if solver is not None:
                if use_layer_view:
                    plt.savefig(f"{sol_path}/layer_{current_layer}.png")
                else:
                    plt.savefig(f"{sol_path}/solution.png")
                plt.clf()
            else:
                plt.savefig(puzzle_path)
        else:
            plt.show()

def find_match(piece_id, og_row, og_col, solution_grid):
    for row in range(board_rows):
        for col in range(board_cols):
            if piece_id in solution_grid[row][col] and [row, col] != [og_row, og_col]:
                return (row, col)

def plot_terminals():
    for row,col in terminals:
        plt.scatter(col, -row, s = size*1.4, marker = "s", c = "black", zorder = 2)
        if (row, col) in elevated_terminals:
            plt.scatter(col, -row, s = size*1.6, marker = "s", c = "white", zorder = 1)
            plt.scatter(col, -row, s = size*2, marker = "s", c = "black", zorder = 0)
    for i, terminal_group in enumerate(active_terminals):
        for row, col in terminal_group:
            plt.scatter(col, -row, s = size*1.4, marker = "s", c = colors[i+1], zorder = 2)

colors = np.array(["black","red","blue","yellow","xkcd:green","orange", "purple"])
def plot_piece_2d(row, col, match_row, match_col, height, node_id = -1):
    plt.plot([col, match_col], [-row, -match_row], lw=9 + 2*height, c=("black" if node_id > 0 else "white"), zorder = 3)
    plt.plot([col, match_col], [-row, -match_row], lw=8, c=colors[node_id], zorder = 4)
    plt.scatter(col, -row, s = size*1.25, c=("black" if node_id > 0 else "white"), zorder = 5)
    plt.scatter(match_col, -match_row, s = size*1.25, c=("black" if node_id > 0 else "white"), zorder = 5)
    plt.scatter(col, -row, s = size, c = colors[node_id], zorder = 6)
    plt.scatter(match_col, -match_row, s = size, c = colors[node_id], zorder = 6)

def plot_piece_3d(row, col, match_row, match_col, layer, height, node_id = -1):
    offset = 0.15
    scaler = 20
    
    # calculate offset location of piece top
    top_offset = layer * offset
    top_row_1, top_col_1 = -row + top_offset, col + top_offset
    top_row_2, top_col_2 = -match_row + top_offset, match_col + top_offset
    
    # calculate offset location of piece bottom
    bot_offset = (layer - height) * offset
    bot_row_1, bot_col_1 = -row + bot_offset, col + bot_offset
    bot_row_2, bot_col_2 = -match_row + bot_offset, match_col + bot_offset
    
    # draw "legs"
    post_width = 20 
    plt.plot([top_col_1, bot_col_1], [top_row_1, bot_row_1], lw=post_width+2, c=("black" if node_id > 0 else "white"), zorder=layer*scaler-2)
    plt.plot([top_col_1, bot_col_1], [top_row_1, bot_row_1], lw=post_width, c=colors[node_id], zorder=layer*scaler-1)
    plt.plot([top_col_2, bot_col_2], [top_row_2, bot_row_2], lw=post_width+2, c=("black" if node_id > 0 else "white"), zorder=layer*scaler-2) 
    plt.plot([top_col_2, bot_col_2], [top_row_2, bot_row_2], lw=post_width, c=colors[node_id], zorder=layer*scaler-1) 

    #connect the ends
    plt.plot([top_col_1, top_col_2], [top_row_1, top_row_2], lw=14, c=("black" if node_id > 0 else "white"), zorder=layer*scaler-1)
    plt.plot([top_col_1, top_col_2], [top_row_1, top_row_2], lw=10, c=colors[node_id], zorder=layer*scaler)

    # draw the terminal circles
    plt.scatter(top_col_1, top_row_1, s=size*1.25, c=("black" if node_id > 0 else "white"), zorder=layer*scaler)
    plt.scatter(top_col_2, top_row_2, s=size*1.25, c=("black" if node_id > 0 else "white"), zorder=layer*scaler)
    plt.scatter(top_col_1, top_row_1, s=size, c=colors[node_id], zorder=layer*scaler+1)
    plt.scatter(top_col_2, top_row_2, s=size, c=colors[node_id], zorder=layer*scaler+1)

#plot puzzle starting configuration
starting_grid = []
for row in range(board_rows):
    row_list = []
    for col in range(board_cols):
        row_list.append([])
    starting_grid.append(row_list)

placements_list = []
is_starter_idx = [] # Tracks if this index is a starter
piece_heights = []

for piece_num, piece in enumerate(pieces): 
    matched_starter = None
    piece_heights.append(piece[2])

    # check if any orientation for this piece matches a starter
    possible_orientations = rotate(piece)
    for [(s_piece, s_row, s_col)] in starting_pieces:
        for orient in possible_orientations:
            if np.array_equal(s_piece, orient):
                #track starting pieces
                matched_starter = [(s_piece, s_row, s_col)]
                #plot starting pieces
                starting_grid[s_row][s_col].append(piece_num) 
                x_len, y_len = s_piece[0], s_piece[1]
                end_row, end_col = s_row+y_len, s_col+x_len
                if 0 <= end_row < board_rows and 0 <= end_col < board_cols:
                    starting_grid[end_row][end_col].append(piece_num)
                else:
                    print(f"Error: Starting piece {piece_num} at ({s_row},{s_col}) extends off-board to ({end_row},{end_col})")
                    assert(False)
                break
            
    if matched_starter:
        #if it's a starter, it can only have one possible placement
        placements_list.append(matched_starter)
        is_starter_idx.append(True)
    else:
        #otherwise it's an unspecified piece. Generate all possible placements for this piece on the board
        placements_list.append(possible_placements(piece))
        is_starter_idx.append(False)

plot_grid(starting_grid)

Model = cp_model.CpModel()

#create boolean variables: if a certain piece is at a certain placement, it's true. Otherwise, it's false.
#Each boolean variable in the list corresponds directly to the piece and placement at the variable's location in the placement list. 
piece_vars_list = []
for i, piece_list in enumerate(placements_list):
    piece_vars = []
    for j in range(len(piece_list)):
        var = Model.new_bool_var(f'piece_{i}_placement_{j}')
        piece_vars.append(var)
    piece_vars_list.append(piece_vars)

#constraints:

#Section 1: Intersection and layering constraints

#return a variable corresponding to the intersection of 2 placements.  
def get_intersection_var(Model, placement1, placement2, z1, z2):
    piece1, row1, col1 = placement1
    piece2, row2, col2 = placement2
    points = [[[row1, col1], [row1+piece1[1], col1+piece1[0]]], [[row2, col2], [row2+piece2[1], col2+piece2[0]]]]
    
    #for efficiency, if there is no chance of intersection, don't add a variable to the model
    if not doIntersect(points):
        return None 
        
    #placements can only intersect if the wires are on the same layer
    intersection_var = Model.new_bool_var("intersection_var")
    Model.add(z1 == z2).only_enforce_if(intersection_var)
    Model.add(z1 != z2).only_enforce_if(~intersection_var)
    return intersection_var

#below is from https://www.geeksforgeeks.org/dsa/check-if-two-given-line-segments-intersect/
def onSegment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
def orient(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2
def doIntersect(points):
    o1 = orient(points[0][0], points[0][1], points[1][0])
    o2 = orient(points[0][0], points[0][1], points[1][1])
    o3 = orient(points[1][0], points[1][1], points[0][0])
    o4 = orient(points[1][0], points[1][1], points[0][1])
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and onSegment(points[0][0], points[1][0], points[0][1]):
        return True
    if o2 == 0 and onSegment(points[0][0], points[1][1], points[0][1]):
        return True
    if o3 == 0 and onSegment(points[1][0], points[0][0], points[1][1]):
        return True
    if o4 == 0 and onSegment(points[1][0], points[0][1], points[1][1]):
        return True
    return False
#above is from https://www.geeksforgeeks.org/dsa/check-if-two-given-line-segments-intersect/

#bottom variable: defines the layer where the piece's bottom is
z_bottom_vars = []

for i,piece in enumerate(pieces):
    z_bottom_vars.append(Model.new_int_var(0, z_max-1, f'z_bottom_{i}'))

#apply physics: elevated terminals, piece ends can't overlap, no overhangs 
for row in range(board_rows):
    for col in range(board_cols):
        #interval variables for all pieces at this node (forces piece ends to be at distinct z intervals)
        node_intervals = []
        #height of all pieces at this node
        node_heights = []
        #top layer of all pieces at this node
        node_tops = []

        #first, handle any elevated terminals. basically treated as a piece stuck to the ground.
        if (row, col) in elevated_terminals:
            terminal_height = Model.new_constant(1)
            node_heights.append(1) 
            node_tops.append(terminal_height)
            terminal_interval = Model.new_fixed_size_interval_var(0,1,f"elevated_terminal_at_{row}_{col}")
            node_intervals.append(terminal_interval)
        
        #check every possible piece placement
        for i, piece_list in enumerate(placements_list):
            piece_height = piece_heights[i]
            for j, (orientation, p_row, p_col) in enumerate(piece_list):
                x_len, y_len = orientation[0], orientation[1]                
                start_node = (p_row, p_col)
                end_node = (p_row + y_len, p_col + x_len)
                current_node = (row, col)
                #physics only needs to be applied to a placement if the placement is at this spot.
                if current_node == start_node or current_node == end_node:
                    is_here = piece_vars_list[i][j]
                    
                    # height variable: adds to the cumulative stack height
                    current_h = Model.new_int_var(0, piece_height, f'h_{i}_{j}_at_{row}_{col}')
                    Model.add(current_h == piece_height).only_enforce_if(is_here)
                    #obviously height of a piece at a location is 0 if it isn't there
                    Model.add(current_h == 0).only_enforce_if(~is_here) 
                    node_heights.append(current_h)
                    
                    # top variable: defines the layer where the piece's top is
                    z_top = Model.new_int_var(0, z_max, f'top_{i}_{j}_at_{row}_{col}')
                    z_bottom = z_bottom_vars[i]
                    Model.add(z_top == z_bottom + piece_height).only_enforce_if(is_here)
                    Model.add(z_top == 0).only_enforce_if(~is_here)
                    node_tops.append(z_top)

                    # interval is used to force piece ends to occupy unique layers
                    node_intervals.append(Model.new_optional_interval_var(z_bottom,piece_height,z_top,is_here,f'interval_{i}_{j}'))

        # finally, apply the physics
        if node_intervals:
            # piece intervals cannot overlap - piece ends must occupy unique layers
            Model.add_no_overlap(node_intervals)
            
            # no overhangs allowed (the sum of piece end heights must equal the total stack height)
            max_stack_height = Model.new_int_var(0, z_max, f'max_h_{row}_{col}')
            Model.add_max_equality(max_stack_height, node_tops)
            Model.add(max_stack_height == sum(node_heights))

#handle collisionswith elevated terminals
for i, piece_list in enumerate(placements_list):
    z_wire_height = z_bottom_vars[i] + piece_heights[i]
    for j, (orientation, start_row, start_col) in enumerate(piece_list):
        wire_var = piece_vars_list[i][j]
        end1 = (start_row, start_col)
        end2 = (start_row + orientation[1], start_col + orientation[0])
        
        # Check against every elevated terminal
        for terminal_loc in elevated_terminals:
            # exclude endpoints 
            if terminal_loc == end1 or terminal_loc == end2:
                continue
            # is the elevated terminal on this line segment?
            # Check 1: Is the terminal collinear with the wire?
            # Check 2: Is the terminal in the wire's "box"?
            if orient(end1, terminal_loc, end2) == 0 and onSegment(end1, terminal_loc, end2):
                # there's a terminal in the way of this wire, so wire height must be > 1 if this placement is used.
                Model.add(z_wire_height > 1).only_enforce_if(wire_var)

z_tops = []
for i in range(len(pieces)):
    z_tops.append(z_bottom_vars[i] + piece_heights[i])

#check for collisions between wires
for i in range(len(placements_list)):
    i_vars = piece_vars_list[i]
    i_placements = placements_list[i]
    z_top_i = z_tops[i] 
    
    for j in range(i+1, len(placements_list)):
        j_vars = piece_vars_list[j]
        j_placements = placements_list[j]
        z_top_j = z_tops[j]

        for idx_i, (i_orientation, i_row, i_col) in enumerate(i_placements):
            #calculate a "box" enclosing placement i 
            i_row_max, i_row_min = i_row + max(0, i_orientation[1]), i_row + min(0, i_orientation[1])
            i_col_max, i_col_min = i_col + max(0, i_orientation[0]), i_col + min(0, i_orientation[0])

            for idx_j, (j_orientation, j_row, j_col) in enumerate(j_placements):
                #calculate a "box" enclosing placement j
                j_row_max, j_row_min = j_row + max(0, j_orientation[1]), j_row + min(0, j_orientation[1])
                j_col_max, j_col_min = j_col + max(0, j_orientation[0]), j_col + min(0, j_orientation[0])

                #in order to prevent introducing unnecesarry variables to the solver, only add the intersection check if the boxes overlap
                if (i_row_max < j_row_min) or (i_row_min > j_row_max) or (i_col_max < j_col_min) or (i_col_min > j_col_max):
                    continue
 
                #make a variable corresponding to whether these placements intersect
                intersect_var = get_intersection_var(Model, 
                                                    (i_orientation, i_row, i_col), 
                                                    (j_orientation, j_row, j_col), 
                                                    z_top_i, z_top_j)
            
                #if they do, only one of those placements can be used
                if intersect_var is not None:
                     Model.add_bool_or([~i_vars[idx_i], ~j_vars[idx_j]]).only_enforce_if(intersect_var)

# we have to check separately for collisions between wires and ends (since ends extend below their wires)
# first, create a dictionary of the ends at each node and their z range 
ends_at_node = {} 

for i, piece_list in enumerate(placements_list):
    z_end_bottom = z_bottom_vars[i]
    z_end_top = z_bottom_vars[i] + piece_heights[i]
    for j, (orientation, row, col) in enumerate(piece_list):
        var = piece_vars_list[i][j]
        #find the endpoints for each placement
        end1 = (row, col)
        end2 = (row + orientation[1], col + orientation[0])
        #add both ends to the dictionary
        for end in [end1, end2]:
            if end not in ends_at_node:
                ends_at_node[end] = []
            ends_at_node[end].append({'var': var, 'bottom': z_end_bottom, 'top': z_end_top})

# then, iterate wires and check against all known posts
for i, piece_list in enumerate(placements_list):
    z_wire = z_bottom_vars[i] + piece_heights[i]
    for j, (orientation, start_row, start_col) in enumerate(piece_list):
        wire_var = piece_vars_list[i][j]
        
        #find the endpoints for each placement
        end1 = [start_row, start_col]
        end2 = [start_row + orientation[1], start_col + orientation[0]]
        
        for end_loc, obstacles in ends_at_node.items():
            # exclude endpoints, handled by other logic
            if end_loc == tuple(end1) or end_loc == tuple(end2):
                continue
            # is the piece end on this line segment?
            # Check 1: Is the end collinear with the other wire?
            # Check 2: Is the end in the wire's "box"?
            if orient(end1, end_loc, end2) == 0 and onSegment(end1, end_loc, end2):
                # there's an end in the way of this wire
                for other_piece_end in obstacles:
                    #if these both exist in the same spot, the wire must be higher.
                    Model.add(z_wire > other_piece_end['top']).only_enforce_if([wire_var, other_piece_end['var']])

#only one boolean variable can be true for each piece. In this puzzle, not all pieces have to be used.
#of course, starting pieces have to be used. They also must be placed on the floor.
for i, piece_vars in enumerate(piece_vars_list):
    if is_starter_idx[i]:
        Model.add(piece_vars[0] == 1)
        Model.add(z_bottom_vars[i] == 0)
    else:
        Model.add(sum(piece_vars)<=1)

#Section 2: Connection constraints

#each spot on the board is assigned an id corresponding to the circuit it's a part of
num_networks = len(active_terminals)
node_ids = []
for row in range(board_rows):
    for col in range(board_cols):
        node_ids.append(Model.new_int_var(0, num_networks, f'node_id_{row}_{col}'))

#list of locations on the board for ease of iteration later
board_locs_list = []
for row in range(board_rows):
    for col in range(board_cols):
        board_locs_list.append((row, col))

#list of locations on the board with active terminals
active_locs = set().union(*active_terminals)

# inactive terminals are forced to have id 0
for loc in terminals:
    if loc not in active_locs:
        index = board_locs_list.index(loc)
        Model.add(node_ids[index] == 0)

# terminals are fixed to an id corresponding to their color
for i, terminal_set in enumerate(active_terminals):
    id = i+1
    for loc in terminal_set:
        index = board_locs_list.index(loc)
        Model.add(node_ids[index] == id)

#wires "conduct" their id from end to end
for i, piece_list in enumerate(placements_list):
    for j, (orientation, row, col) in enumerate(piece_list):
        start_index = board_locs_list.index((row, col))
        end_row, end_col = row + orientation[1], col + orientation[0]
        end_index = board_locs_list.index((end_row, end_col))
        
        #ids of nodes must match at both ends of a piece
        Model.add(node_ids[start_index] == node_ids[end_index]).only_enforce_if(piece_vars_list[i][j])
        
#In order to force all terminals to be connected (no self loops or other shenanigans)
# we set up a flow system.
# make one terminal in the set a "supply" and give all other terminals a "demand" that must be fulfilled
node_supply = [0] * len(board_locs_list)
for terminal_set in active_terminals:
    #define supply and demand
    supply_index = board_locs_list.index(terminal_set[0])
    demand_locs = terminal_set[1:]
    #all supplies are set to 0
    
    #except for the supply node, which is given 1 supply for each demand
    node_supply[supply_index] = len(demand_locs)
    #and the demand nodes, which are given -1 supply each
    for loc in demand_locs:
        demand_index = board_locs_list.index(loc)
        node_supply[demand_index] = -1

# define flow
piece_flow = [] 
# to keep bounds on integer variables tight, max flow = number of demand nodes in the largest terminal set
max_flow = max([len(terminal_set) for terminal_set in active_terminals]) - 1

for i, piece_list in enumerate(placements_list):
    placement_flow = []
    for j, (orientation, row, col) in enumerate(piece_list):
        flow = Model.new_int_var(-max_flow, max_flow, f'flow_{i}_{j}')

        #if a placement is not used, flow must be 0 for that placement
        Model.add(flow == 0).only_enforce_if(~piece_vars_list[i][j])

        #a placed piece can have flow zero (useless piece), but then it must have id = 0 (for clarity in ploting solutions).
        zero_flow = Model.new_bool_var(f'zero_flow_{i}_{j}')
        Model.add(flow == 0).only_enforce_if(zero_flow)
        Model.add(flow != 0).only_enforce_if(~zero_flow)
        id = node_ids[board_locs_list.index((row, col))]
        Model.add(id == 0).only_enforce_if([zero_flow, piece_vars_list[i][j]])

        placement_flow.append(flow)
    piece_flow.append(placement_flow)

#flow is created or removed in accordance with supply and demand. Otherwise, flow in = flow out of a node
for node_index in range(len(board_locs_list)):
    inflow = 0
    outflow = 0
    #check which pieces are touching the node
    for i, piece_list in enumerate(placements_list):
        for j, (orientation, row, col) in enumerate(piece_list):
            start_index = board_locs_list.index((row, col))
            end_index = board_locs_list.index((row + orientation[1], col + orientation[0]))
            
            if end_index == node_index:
                #if the "end" of the piece is touching the node, flow enters the node
                inflow += piece_flow[i][j]
            elif start_index == node_index:
                #if the "start" of the piece is touching the node, flow exits the node
                outflow += piece_flow[i][j] 
    net_flow = inflow - outflow
    Model.add(net_flow == node_supply[node_index])

# Solve the model
print(f'Solve starting at {time.time()-start} s')
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = max_solve_time
all_piece_bools = [var for sublist in piece_vars_list for var in sublist]
Model.add_decision_strategy(all_piece_bools, cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)
#solver.parameters.log_search_progress = True
solution = solver.Solve(Model)
print(f'Solve finished at {time.time()-start} s')

#if there's a solution, plot it.
if solution in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        create_sol_grid(placements_list, piece_vars_list, solver)
elif solution == cp_model.UNKNOWN:
    print(f"Search aborted after {max_solve_time/60} minutes.")
    if write_to_file:
        #clear things out so we can rerun with the same puzzle number
        os.remove(puzzle_path)
        import shutil  # Import here if not at top
        shutil.rmtree(sol_path)
else:
    print("No solution found.")
    if write_to_file:
        #clear things out so we can rerun with the same puzzle number
        os.remove(puzzle_path)
        import shutil
        shutil.rmtree(sol_path)
