# via
"Through silicon vias are vertical wires used to precisely connect stacked chips" - Applied Materials 

The vias used to layer circuits serve as the inspiration for this puzzle. Via is about connecting like-colored terminals using "wire" pieces. For a given via puzzle, you will see a number of colored terminals and a number of black terminals. Terminals of the same color must all be connected using a series of wires, but these wires cannot connect (either directly or indirectly) to terminals of another color. Similarly, no terminal can be connected to a black terminal, so black terminals are essentially off limits for building connections. Wires can be stacked on top of each other but cannot intersect. Terminals with a black outline are "elevated terminals", and exist one layer above the rest of the board. Finally, wires cannot float. A wire can only exist on a higher layer if both ends are fully supported underneath by other wires or the board itself. The rules are probably easiest to understand by looking at a sample puzzle and solution. 

This repo contains many sample puzzles and solutions using all the same piece shapes. Notably, unlike some other puzzles I have uploaded, a puzzle CAN have more than one solution. Puzzle types are simply grouped by how standardized they are. Type a has all terminals and elevated terminals in the same spots for each puzzle. Type b has all terminals on the edges of the board. Type c has no standard terminal number or locations.

The repo also contains a solver for any given starting conditions and a puzzle generator to create new puzzles. These plot solutions in 2 different ways, so you can easily see where to put the pieces. 

coming soon - make solver/generator more efficient with multiple terminal sets, delete bad/trivial puzzles, generate more (good) puzzles, and upload 3d models. 

This is a completely original project. Feel free to make copies of or modify any part of this project for personal, non-commercial use. For any commercial use, please request my permission.
