from ase.build import bulk, surface
import numpy as np

def check_structure(miller):
    print(f"--- Checking {miller} Surface ---")
    atoms = bulk('Cu', 'fcc', a=3.61)
    # Current method
    slab = surface(atoms, miller, layers=4)
    
    cell = slab.get_cell()
    print("Cell Vectors:")
    print(cell)
    print("Angles:", slab.get_cell_lengths_and_angles()[3:])
    
    # Check orthogonality
    is_orthogonal = (abs(np.dot(cell[0], cell[1])) < 1e-5 and 
                     abs(np.dot(cell[0], cell[2])) < 1e-5 and 
                     abs(np.dot(cell[1], cell[2])) < 1e-5)
    print(f"Is Orthogonal: {is_orthogonal}")
    
    # Check number of atoms per layer area to guess packing
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    print(f"Base Area: {area:.2f} A^2")

check_structure((1, 0, 0))
check_structure((1, 1, 0))
check_structure((1, 1, 1))
