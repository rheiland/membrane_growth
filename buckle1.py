import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class MembraneCell:
    """Model of a membrane cell that buckles under compression"""
    
    def __init__(self, length=2.0, thickness=0.02, n_nodes=50):
        self.L0 = length  # Original length
        self.thickness = thickness
        self.n_nodes = n_nodes
        
        # Material properties (normalized)
        self.E = 1.0  # Young's modulus
        self.I = thickness**3 / 12  # Second moment of area
        self.EI = self.E * self.I  # Bending stiffness
        
        # Critical buckling load (Euler buckling)
        self.P_critical = (np.pi**2 * self.EI) / (self.L0**2)
        
    def solve_buckling(self, compression_ratio):
        """Solve for buckled shape at given compression"""
        # Current length after compression
        L = self.L0 * (1 - compression_ratio)
        
        # Axial force increases with compression
        # P/P_critical determines buckling amplitude
        load_ratio = compression_ratio / 0.2  # Normalize
        
        # Position along beam
        x = np.linspace(0, L, self.n_nodes)
        
        if load_ratio < 1.0:
            # Before buckling - straight with slight imperfection
            amplitude = 0.02 * load_ratio
            y = amplitude * np.sin(np.pi * x / L)
        else:
            # Post-buckling - significant deflection
            # Amplitude grows with sqrt(P/Pcr - 1) for supercritical buckling
            amplitude = 0.15 * np.sqrt(max(0, load_ratio - 1))
            
            # Can have multiple modes - using first mode
            y = amplitude * np.sin(np.pi * x / L)
            
            # Add second mode for more interesting buckling
            if load_ratio > 1.5:
                y += 0.05 * np.sin(2 * np.pi * x / L)
        
        # Add initial imperfection to trigger buckling
        imperfection = 0.01 * np.sin(np.pi * x / L)
        y += imperfection
        
        return x, y

def simulate_buckling_stages():
    """Generate different buckling stages"""
    cell = MembraneCell(length=2.0, thickness=0.02, n_nodes=100)
    
    # Range of compression ratios
    compression_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    states = []
    
    for comp in compression_levels:
        x, y = cell.solve_buckling(comp)
        states.append({
            'x': x,
            'y': y,
            'compression': comp
        })
    
    return states, cell

def plot_buckling_progression(states, cell):
    """Plot multiple states of buckling"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (state, ax) in enumerate(zip(states, axes)):
        x, y = state['x'], state['y']
        comp = state['compression']
        thickness = cell.thickness
        
        # Center the membrane vertically
        y_center = 0.5
        y = y + y_center
        
        # Plot upper and lower membrane surfaces
        ax.plot(x, y + thickness/2, 'b-', linewidth=2.5)
        ax.plot(x, y - thickness/2, 'b-', linewidth=2.5)
        ax.fill_between(x, y - thickness/2, y + thickness/2, alpha=0.4, color='blue')
        
        # Add reference line for original position
        ax.axhline(y=y_center, color='r', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlim(-0.1, 2.1)
        ax.set_ylim(0.2, 0.8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Compression: {comp*100:.1f}%\nLoad ratio: {(comp/0.2):.2f}', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Position (x)', fontsize=10)
        ax.set_ylabel('Position (y)', fontsize=10)
        
        # Add text showing max deflection
        max_deflection = np.max(np.abs(y - y_center))
        ax.text(0.02, 0.98, f'Max deflection: {max_deflection:.3f}', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Membrane Cell Buckling Under Compression', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('membrane_buckling.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_buckling_modes():
    """Show different buckling modes"""
    cell = MembraneCell(length=2.0, thickness=0.02, n_nodes=100)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    compression = 0.5
    
    for mode, ax in enumerate(axes, start=1):
        L = cell.L0 * (1 - compression)
        x = np.linspace(0, L, cell.n_nodes)
        
        # Different buckling modes
        if mode == 1:
            amplitude = 0.15
            y = amplitude * np.sin(mode * np.pi * x / L)
            title = f'Mode {mode}: Single Half-Wave'
        elif mode == 2:
            amplitude = 0.12
            y = amplitude * np.sin(mode * np.pi * x / L)
            title = f'Mode {mode}: Full Wave'
        else:
            amplitude = 0.10
            y = amplitude * np.sin(mode * np.pi * x / L)
            title = f'Mode {mode}: Three Half-Waves'
        
        y_center = 0.5
        y = y + y_center
        thickness = cell.thickness
        
        ax.plot(x, y + thickness/2, 'b-', linewidth=2.5)
        ax.plot(x, y - thickness/2, 'b-', linewidth=2.5)
        ax.fill_between(x, y - thickness/2, y + thickness/2, alpha=0.4, color='blue')
        ax.axhline(y=y_center, color='r', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlim(-0.1, 2.1)
        ax.set_ylim(0.2, 0.8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Position (x)', fontsize=10)
        ax.set_ylabel('Position (y)', fontsize=10)
    
    plt.suptitle(f'Different Buckling Modes at {compression*100:.0f}% Compression', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('buckling_modes.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run simulation
print("=" * 60)
print("MEMBRANE CELL BUCKLING SIMULATION")
print("=" * 60)

print("\nSimulating progressive buckling...")
states, cell = simulate_buckling_stages()

print(f"Cell properties:")
print(f"  - Original length: {cell.L0:.2f}")
print(f"  - Thickness: {cell.thickness:.3f}")
print(f"  - Bending stiffness (EI): {cell.EI:.6f}")
print(f"  - Critical buckling load: {cell.P_critical:.6f}")

print("\nBuckling progression:")
for state in states:
    comp = state['compression']
    max_def = np.max(np.abs(state['y']))
    print(f"  Compression {comp*100:4.1f}% -> Max deflection: {max_def:.4f}")

print("\nCreating visualizations...")
plot_buckling_progression(states, cell)
plot_buckling_modes()

print("\n" + "=" * 60)
print("Simulation complete!")
print("=" * 60)
print("\nKey observations:")
print("  - Buckling begins around 20% compression (load ratio = 1.0)")
print("  - Deflection increases rapidly in post-buckling regime")
print("  - Multiple buckling modes possible at high compression")
