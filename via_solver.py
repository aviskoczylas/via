import time
start = time.time()
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
import os
#TODO once everything else is done - make a solid 10 puzzles by hand, then write a script to automate puzzle generation.
#TODO this code is in serious need of better documentation/commenting. Read thru everything myself - no ai.
#TODO ask ai to make my code cleaner after all else is done

puzzle_num = 1
write_to_file = True
if write_to_file:
    if os.path.exists(f"sample_puzzles/puzzle{puzzle_num}.png") or os.path.exists(f"sample_solutions/sol{puzzle_num}"):
        print("Error - path already exists.")
        assert(False)

#Two plotting options: a single isometric view, or a simple 2d plot of each layer separately
separate_layer_view = False
#Maximum z layer that pieces can occupy. Lower makes solver much faster, but if too low, may miss possible solutions
z_max = 5

max_solve_time = 360

#generate all possible orientations for the pieces
def flip_and_rotate(piece):
    orientations = [piece]
    orientations.append([piece[0], -piece[1], piece[2]])
    orientations.append([piece[1], piece[0], piece[2]])
    orientations.append([-piece[1], piece[0], piece[2]])
    if piece[0] != 0 and piece[1] !=0:
        orientations.append([-piece[0], piece[1], piece[2]])
        orientations.append([-piece[0], -piece[1], piece[2]])
        orientations.append([piece[1], -piece[0], piece[2]])
        orientations.append([-piece[1], -piece[0], piece[2]])
    return orientations

# load pieces from a file.
with open('via_pieces.pkl', 'rb') as file:
    pieces = pickle.load(file)

#define board shape
board_rows = 5
board_cols = 5
t = [(0,i) for i in range(0,board_cols-1)]
b = [(board_rows-1,i) for i in range(1,board_cols)]
l = [(i,0) for i in range(1,board_rows-1)]
r = [(i,board_cols-1) for i in range(1,board_rows-1)]
terminals = t+r+b[::-1]+l[::-1] #clockwise
#     0  1 2 3 x
#     13 x x x 4
#     12 x x x 5
#     11 x x x 6
#     x 10 9 8 7

active_terminals = [(terminals[11], terminals[1], terminals[4]),
                    (terminals[2], terminals[13], terminals[7])]
                    
elevated_terminals = [terminals[12], terminals[0],terminals[2], terminals[4],terminals[6]]

#specify starting pieces as [piece, row, col], for example [(flip_and_rotate(pieces[0])[0],0,0)]
starting_pieces = [
]   

#generate all possible placements as a list of [orientation, x, y]
#note that a "placement" on the board does specify an orientation
def possible_placements(piece):
    placements = []
    for orientation in flip_and_rotate(piece):
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
            row_list.append([-1])
        solution_grid.append(row_list)
    for i, piece_placements in enumerate(placements_list):
        for j, (piece, row, col) in enumerate(piece_placements):
            x_len, y_len = piece[0], piece[1]
            if solver.Value(piece_vars_list[i][j]):
                solution_grid[row][col].append(i)
                solution_grid[row+y_len][col+x_len].append(i)
    print(solution_grid)
    plot_grid(solution_grid, solver)

size = 800
def plot_grid(solution_grid, solver = None):
    if solver is not None:
        os.makedirs(f"sample_solutions/sol{puzzle_num}")
    plotted_pieces = [-1] # -1 is essentially a padding number. 
    max_z = 0
    if separate_layer_view:
        #we need to calculate how many 2d plots to generate
        if solver is not None:
            for i, piece_vars in enumerate(piece_vars_list):
                for var in piece_vars:
                    if solver.Value(var):
                        top = solver.Value(z_bottom_vars[i]) + piece_heights[i]
                        if top > max_z:
                            max_z = top    
    for current_layer in range(separate_layer_view*max_z+1):
        plot_terminals()
        for row in range(board_rows-2):
            for col in range(board_cols-2):
                plt.scatter(col+1, -row-1, s = 1, c="black")
        for row in range(board_rows):
            for col in range(board_cols):
                if solver is not None:
                    node_id = solver.Value(node_ids[row*board_cols+col])
                    #If we are plotting the solution, color the pieces according to the circuit they are part of
                else:
                    node_id = -1
                for piece_id in solution_grid[row][col]:
                    if piece_id not in plotted_pieces:
                        if solver is not None:
                            layer = solver.Value(z_bottom_vars[piece_id])+piece_heights[piece_id]
                        else:
                            layer = piece_heights[piece_id] #starting pieces can only be placed at ground level!
                        match_row, match_col = find_match(piece_id, row, col, solution_grid)
                        if separate_layer_view:
                            if layer == current_layer:
                                plot_piece_2d(row, col, match_row, match_col, piece_heights[piece_id], node_id)
                                plotted_pieces.append(piece_id)
                        else:
                            plot_piece_3d(row, col, match_row, match_col, layer, piece_heights[piece_id], node_id)
                            plotted_pieces.append(piece_id)
        plt.xlim(-0.5, board_cols-0.5)
        plt.ylim(-board_rows-0.5,1.5)
        plt.gca().set_aspect(1, adjustable='datalim')
        plt.tight_layout()
        plt.gca().tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if write_to_file:
            if solver is not None:
                if separate_layer_view:
                    plt.savefig(f"sample_solutions/sol{puzzle_num}/layer_{current_layer}.png")
                else:
                    plt.savefig(f"sample_solutions/sol{puzzle_num}/solution.png")
            else:
                plt.savefig(f"sample_puzzles/puzzle{puzzle_num}.png")
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

            plt.scatter(col, -row, s = size*1.8, marker = "s", c = "black", zorder = 0)
    for i, terminal_group in enumerate(active_terminals):
        for row, col in terminal_group:
            plt.scatter(col, -row, s = size*1.4, marker = "s", c = colors[i+1], zorder = 2)

colors = np.array(["black","red","orange","yellow","xkcd:green","blue", "purple"])
def plot_piece_2d(row, col, match_row, match_col, height, node_id = -1):
    plt.plot([col, match_col], [-row, -match_row], lw=9 + 2*height, c=("black" if node_id > 0 else "white"), zorder = 3)
    plt.plot([col, match_col], [-row, -match_row], lw=8, c=colors[node_id], zorder = 4)
    plt.scatter(col, -row, s = size*1.25, c=("black" if node_id > 0 else "white"), zorder = 5)
    plt.scatter(match_col, -match_row, s = size*1.25, c=("black" if node_id > 0 else "white"), zorder = 5)
    plt.scatter(col, -row, s = size, c = colors[node_id], zorder = 6)
    plt.scatter(match_col, -match_row, s = size, c = colors[node_id], zorder = 6)

def plot_piece_3d(row, col, match_row, match_col, layer, height, node_id = -1):
    # Parameters
    parallax = 0.15
    x_zorder = 20
    
    # 1. Calculate Coordinates
    # Top: The visual position of the piece itself (The Wire)
    top_offset = layer * parallax
    t_row, t_col = -row + top_offset, col + top_offset
    t_mrow, t_mcol = -match_row + top_offset, match_col + top_offset
    
    # Bottom: The visual position of the piece's feet (The Base)
    # We drop down by 'height' layers to find where the post starts
    bot_offset = (layer - height) * parallax
    b_row, b_col = -row + bot_offset, col + bot_offset
    b_mrow, b_mcol = -match_row + bot_offset, match_col + bot_offset
    
    # 2. Draw Vertical Posts (The "Legs")
    # We draw these slightly BEHIND the caps (zorder - 2)
    # We assume 'size' is roughly 800, so width ~20-25 looks good
    post_width = 20 
    
    # Left/Start Post
    plt.plot([t_col, b_col], [t_row, b_row], lw=post_width+2, c=("black" if node_id > 0 else "white"), zorder=layer*x_zorder-2) # Outline
    plt.plot([t_col, b_col], [t_row, b_row], lw=post_width, c=colors[node_id], zorder=layer*x_zorder-1) # Color
    
    # Right/End Post
    plt.plot([t_mcol, b_mcol], [t_mrow, b_mrow], lw=post_width+2, c=("black" if node_id > 0 else "white"), zorder=layer*x_zorder-2) 
    plt.plot([t_mcol, b_mcol], [t_mrow, b_mrow], lw=post_width, c=colors[node_id], zorder=layer*x_zorder-1) 

    # 3. Draw The Wire (Horizontal Connection)
    wire_width = 9 + 2 * height # Your scaling preference
    
    # Wire Outline (Black)
    plt.plot([t_col, t_mcol], [t_row, t_mrow], 
             lw=wire_width + 4, c=("black" if node_id > 0 else "white"), zorder=layer*x_zorder-1)
    # Wire Fill (Color)
    plt.plot([t_col, t_mcol], [t_row, t_mrow], 
             lw=wire_width, c=colors[node_id], zorder=layer*x_zorder)

    # 4. Draw Caps (The Terminals)
    # These go on TOP of everything (zorder)
    
    # Cap Outlines
    plt.scatter(t_col, t_row, s=size*1.25, c=("black" if node_id > 0 else "white"), zorder=layer*x_zorder)
    plt.scatter(t_mcol, t_mrow, s=size*1.25, c=("black" if node_id > 0 else "white"), zorder=layer*x_zorder)
    
    # Cap Fills
    plt.scatter(t_col, t_row, s=size, c=colors[node_id], zorder=layer*x_zorder+1)
    plt.scatter(t_mcol, t_mrow, s=size, c=colors[node_id], zorder=layer*x_zorder+1)

#plot starting piece configuration
starting_grid = []
for row in range(board_rows):
    row_list = []
    for col in range(board_cols):
        row_list.append([-1])
    starting_grid.append(row_list)
#pieces cannot be identical!

for [(starting_piece,row,col)] in starting_pieces:
    starting_grid[row][col].append(1)
    x_len, y_len = starting_piece[0], starting_piece[1]
    starting_grid[row+x_len][col+y_len].append(1)

plot_grid(starting_grid)

#generate a list of sub-lists, each list corresponds to a different piece. 
#Each sub-list contains all possible placements for the piece.
placements_list = []
for piece in pieces: #pieces are allowed to be identical!
    if not any(
    any(np.array_equal(piece, orient) for orient in flip_and_rotate(starting_piece[0][0]))
    for starting_piece in starting_pieces):
        placements_list.append(possible_placements(piece))
#there's only one possible placement for starting pieces.
for piece_placement in starting_pieces:
    placements_list.append(piece_placement)

#TODO force starting piece placement to be true!

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

board_locs_list = []
for row in range(board_rows):
    for col in range(board_cols):
        board_locs_list.append((row, col))

#constraints:

#exactly one boolean variable can (and must) be true for each piece. Therefore, the sum of all boolean variables for each piece must be 1.
for piece_vars in piece_vars_list:
    Model.add(sum(piece_vars)<=1)

def get_intersection_var(Model, placement1, placement2, z1, z2):
    piece1, row1, col1 = placement1
    piece2, row2, col2 = placement2
    points = [[[row1, col1], [row1+piece1[1], col1+piece1[0]]], 
            [[row2, col2], [row2+piece2[1], col2+piece2[0]]]]
    
    if not doIntersect(points):
        return None 
        
    intersection_var = Model.new_bool_var(f"intersect_{row1}x{col1}_with_{row2}x{col2}")
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

z_bottom_vars = []
piece_heights = []

for i,piece in enumerate(pieces):
    z_bottom_vars.append(Model.new_int_var(0, z_max-1, f'z_bottom_{i}'))
    piece_heights.append(piece[2])

# 2. Per-Cell Physics Constraints (Corrected for Posts-Only)
for row in range(board_rows):
    for col in range(board_cols):
        
        cell_intervals = []
        cell_heights = []
        cell_tops = []
        
        #first, handle any elevated terminals
        if (row, col) in elevated_terminals:
            # Create a constant variable for the solver
            # (Use NewConstant so it can be used in "AddMaxEquality")
            terrain_constant = Model.new_constant(1)
            
            # 1. Gravity: The terrain counts as "stuff" in the stack
            # sum() can mix integers and variables, so just appending the int is fine
            cell_heights.append(1) 
            
            # 2. Max Height: The terrain sets the minimum "floor"
            # If no pieces are here, the "Max Top" is the terrain height.
            cell_tops.append(terrain_constant)
            
            # 3. Collision: Prevent pieces from clipping into the ground
            # We create a FIXED interval that is always active
            terrain_interval = Model.new_fixed_size_interval_var(
                0,          # Start
                1,     # Size
                f"terrain_{row}_{col}"
            )
            cell_intervals.append(terrain_interval)
        
        # Check every possible piece placement
        for i, piece_list in enumerate(placements_list):
            piece_height = piece_heights[i]
            for j, (orientation, p_row, p_col) in enumerate(piece_list):
                
                # --- IDENTIFY POST LOCATIONS ---
                # orientation is [x_len, y_len, height]
                y_len = orientation[1]
                x_len = orientation[0]
                
                # The two solid posts are at the Start and the End
                start_node = (p_row, p_col)
                end_node = (p_row + y_len, p_col + x_len)
                current_node = (row, col)

                # --- CHECK OCCUPANCY ---
                # Physics only applies if the current cell IS one of the posts.
                # (The wire in the middle is handled by the intersection loop later)
                if current_node == start_node or current_node == end_node:
                    
                    is_present = piece_vars_list[i][j]
                    
                    # 1. Height Variable (Adds to the stack height)
                    current_h = Model.new_int_var(0, piece_height, f'h_{i}_{j}_at_{row}_{col}')
                    Model.add(current_h == piece_height).only_enforce_if(is_present)
                    Model.add(current_h == 0).only_enforce_if(~is_present)
                    cell_heights.append(current_h)
                    
                    # 2. Top Variable (Defines the roof of the stack)
                    z_top = Model.new_int_var(0, z_max, f'top_{i}_{j}_at_{row}_{col}')
                    z_bottom = z_bottom_vars[i]
                    
                    Model.add(z_top == z_bottom + piece_height).only_enforce_if(is_present)
                    Model.add(z_top == 0).only_enforce_if(~is_present)
                    cell_tops.append(z_top)

                    # 3. The Interval (Prevents posts from merging)
                    interval = Model.new_optional_interval_var(
                        z_bottom,          # Start
                        piece_height,      # Size
                        z_top,             # End
                        is_present,        # Active?
                        f'interval_{i}_{j}'
                    )
                    cell_intervals.append(interval)

        # Apply physics
        if cell_intervals:
            # A. Posts cannot have the same z at the same spot
            Model.add_no_overlap(cell_intervals)
            
            # B. Gravity: No overhangs allowed 
            # (The stack of posts must equal the highest point reached)
            max_stack_height = Model.new_int_var(0, z_max, f'max_h_{row}_{col}')
            Model.add_max_equality(max_stack_height, cell_tops)
            Model.add(max_stack_height == sum(cell_heights))

#handle collisions with elevated terminals
for i, piece_list in enumerate(placements_list):
    # Calculate Wire Z-Level ONCE
    z_wire_height = z_bottom_vars[i] + piece_heights[i]
    #there can only be an intersection for a non diagonal piece
    if pieces[i][0] == 0 or pieces[i][1] == 0:
        for j, (orientation, start_row, start_col) in enumerate(piece_list):
            var = piece_vars_list[i][j]
            found_obstacle = False
            # Define End Point
            end_row = start_row + orientation[1]
            end_col = start_col + orientation[0]
            
            if start_row == end_row:
                c_min = min(start_col, end_col)
                c_max = max(start_col, end_col)
                if any((start_row, c) in elevated_terminals for c in range(c_min + 1, c_max)):
                    found_obstacle = True
            elif start_col == end_col:
                r_min = min(start_row, end_row)
                r_max = max(start_row, end_row)
                if any((r, start_col) in elevated_terminals for r in range(r_min + 1, r_max)):
                    found_obstacle = True
            if found_obstacle:
                # Wire must be strictly higher than the obstacle (Height 1)
                # So Wire Height > 1 (meaning it must be at least 2)
                Model.add(z_wire_height > 1).only_enforce_if(var)

# --- OPTIMIZED INTERSECTION CHECKS ---
# Pre-calculate Z-Tops once to save memory
z_tops = []
for i in range(len(pieces)):
    z_tops.append(z_bottom_vars[i] + piece_heights[i])

# Optimized Loop
for i in range(len(placements_list)):
    i_vars = piece_vars_list[i]
    i_placements = placements_list[i]
    z_top_i = z_tops[i] 
    
    for j in range(i+1, len(placements_list)):
        j_vars = piece_vars_list[j]
        j_placements = placements_list[j]
        z_top_j = z_tops[j]

        for idx_i, (i_orientation, i_row, i_col) in enumerate(i_placements):
            # Calculate Bounding Box I
            i_row_max, i_row_min = i_row + max(0, i_orientation[1]), i_row + min(0, i_orientation[1])
            i_col_max, i_col_min = i_col + max(0, i_orientation[0]), i_col + min(0, i_orientation[0])

            for idx_j, (j_orientation, j_row, j_col) in enumerate(j_placements):
                # Calculate Bounding Box J
                j_row_max, j_row_min = j_row + max(0, j_orientation[1]), j_row + min(0, j_orientation[1])
                j_col_max, j_col_min = j_col + max(0, j_orientation[0]), j_col + min(0, j_orientation[0])

                # 1. FAST FILTER: Bounding Box
                # If the rectangles don't overlap, skip EVERYTHING.
                if (i_row_max < j_row_min) or (i_row_min > j_row_max) or \
                   (i_col_max < j_col_min) or (i_col_min > j_col_max):
                    continue

                # 2. GEOMETRY CHECK
                intersect_var = get_intersection_var(Model, 
                                                    (i_orientation, i_row, i_col), 
                                                    (j_orientation, j_row, j_col), 
                                                    z_top_i, z_top_j)
                
                # 3. CRITICAL CHECK: Only add constraint if intersection is real
                if intersect_var is not None:
                     Model.add_bool_or([~i_vars[idx_i], ~j_vars[idx_j]]).only_enforce_if(intersect_var)







# we have to check seperately for collisions between wires and ends
# first, create a dictionary ofthe ends at each spot their z range 
ends_at_spot = {} 

for i, piece_list in enumerate(placements_list):
    # Calculate Post Z-Range ONCE per piece
    # Posts are solid from z_bottom to z_bottom + height
    z_end_bottom = z_bottom_vars[i]
    z_end_top = z_bottom_vars[i] + piece_heights[i]
    
    for j, (orientation, row, col) in enumerate(piece_list):
        var = piece_vars_list[i][j]
        
        # Identify the two endpoints (The Posts)
        end1 = (row, col)
        end2 = (row + orientation[1], col + orientation[0])
        
        for end in [end1, end2]:
            if end not in ends_at_spot:
                ends_at_spot[end] = []
            
            ends_at_spot[end].append({
                'var': var,
                'bottom': z_end_bottom,
                'top': z_end_top
            })

# then, iterate wires and check against all known posts
for i, piece_list in enumerate(placements_list):
    
    # Wire height is fixed relative to the piece
    z_wire = z_bottom_vars[i] + piece_heights[i]
    
    for j, (orientation, start_row, start_col) in enumerate(piece_list):
        wire_var = piece_vars_list[i][j]
        
        end1 = [start_row, start_col]
        end2 = [start_row + orientation[1], start_col + orientation[0]]
        
        for end_loc, obstacles in ends_at_spot.items():
            
            # exclude endpoints, handled by other logic
            if end_loc == tuple(end1) or end_loc == tuple(end2):
                continue
            
            # is the post endpoint on this line segment?
            # Check A: Is the post collinear with the wire?
            # Check B: Is the post strictly inside the bounding box of the wire?
            if orient(end1, end_loc, end2) == 0 and onSegment(end1, end_loc, end2):
                # We found a post in the way!
                for other_piece_end in obstacles:
                    #if these both exist in the same spot, the wire must be higher.
                    Model.add(z_wire > other_piece_end['top']).only_enforce_if([wire_var, other_piece_end['var']])

num_networks = len(active_terminals)
node_ids = []
for row in range(board_rows):
    for col in range(board_cols):
        node_ids.append(Model.new_int_var(0, num_networks, f'node_id_{row}_{col}'))

active_locs = set().union(*active_terminals)
# inactive terminals are forced to have id 0
for loc in terminals:
    if loc not in active_locs:
        index = board_locs_list.index(loc)
        Model.add(node_ids[index] == 0)

# terminals are fixed to their colors
for i, terminal_set in enumerate(active_terminals):
    id = i+1
    for loc in terminal_set:
        index = board_locs_list.index(loc)
        Model.add(node_ids[index] == id)

#define conduction through pieces
for i, piece_list in enumerate(placements_list):
    for j, (orientation, row, col) in enumerate(piece_list):
        piece_var = piece_vars_list[i][j]
        start_index = board_locs_list.index((row, col))
        end_row, end_col = row + orientation[1], col + orientation[0]
        end_index = board_locs_list.index((end_row, end_col))
        
        #ids of nodes must match at both ends of a piece
        Model.add(node_ids[start_index] == node_ids[end_index]).only_enforce_if(piece_var)
        
#In order to force all terminals to be connected (no self loops or other shenanigans)
# we set up an entirely seperate flow system.
# make one terminal in the set a "supply" and give all other terminals a "demand" that must be fulfilled
for network_index, terminal_set in enumerate(active_terminals):
    #define supply and demand
    id = network_index + 1
    supply_index = board_locs_list.index(terminal_set[0])
    demand_locs = terminal_set[1:]
    node_supply = [0] * len(board_locs_list)
    node_supply[supply_index] = len(demand_locs)
    for loc in demand_locs:
        demand_index = board_locs_list.index(loc)
        node_supply[demand_index] = -1

    # define flow
    piece_flow = [] 
    max_flow = len(terminal_set) - 1
    
    for i, piece_list in enumerate(placements_list):
        placement_flow = []
        for j, (orientation, row, col) in enumerate(piece_list):
            flow = Model.new_int_var(-max_flow, max_flow, f'flow_net{id}_p{i}_{j}')

            #if a placement is not used, flow must be 0 for that placement
            Model.add(flow == 0).only_enforce_if(~piece_vars_list[i][j])
            
            #if a placement is used, but it already has a different id, flow must be 0 for that placement (as seen by this network)
            start_index = board_locs_list.index((row, col))
            taken = Model.new_bool_var(f'taken{id}_placement{i}_{j}')
            Model.add(node_ids[start_index] == id).only_enforce_if(~taken)
            Model.add(node_ids[start_index] != id).only_enforce_if(taken)
            Model.add(flow == 0).only_enforce_if(taken)
            
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
                
        #apply the supply/demand constraint to all nodes in the network
        node_in_network = Model.new_bool_var(f'node_{node_index}_in_network{id}')
        Model.add(node_ids[node_index] == id).only_enforce_if(node_in_network)
        Model.add(node_ids[node_index] != id).only_enforce_if(~node_in_network)
        Model.add(net_flow == node_supply[node_index]).only_enforce_if(node_in_network)
        #otherwise, flow with respect to this network is 0
        Model.add(net_flow == 0).only_enforce_if(~node_in_network)

# Solve the model
print(f'Solve starting at {time.time()-start} s')
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = max_solve_time
solver.parameters.log_search_progress = True
solution = solver.Solve(Model)
print(f'Solve finished at {time.time()-start} s')

#if there's a solution, plot it.
if solution in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        create_sol_grid(placements_list, piece_vars_list, solver)
elif solution == cp_model.UNKNOWN:
    print(f"Search aborted after {max_solve_time/60} minutes.")
else:
    print("No solution found.")