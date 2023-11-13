import os
from ProbAsn import utils as ut
import tkinter as tk
from ProbAsn import gui as gui

sys_vars = ut.get_default_values("SYS")

print(sys_vars)

application_params = {
    "steps": [
        "Input structure",
        "Select atoms",
        "Set chemical shifts",
        "Fetch database",
        "Build distributions",
        "Compute prior",
        "Run assignment"
    ],
    "cur_step": 0,
    "width": 1000,
    "height": 800,
    "db_dir": os.path.abspath(sys_vars["db_dir"]) + "/",
    "in_file": None,
    "in_types": [
        ("XYZ files", "*.xyz"),
        ("PDB files", "*.pdb"),
        ("MOL files", "*.mol"),
        ("SMILES files", "*.smi"),
    ],
    "content_min_height": 600,
    "content_min_width": 600,
    
}

print(application_params)

if __name__ == "__main__":

    root = tk.Tk()

    app = gui.Application(root, application_params)

    root.mainloop()

quit()

max_w = 6
allowed_elems = ["H", "C", "N", "O", "S", "F", "P", "Cl", "Na", "Mg", "K", "Ca"]
out_dir = "../working_out/test/"

# Load molecule
mol = gr.make_mol("O=C1N(C)C2N=CNC(=2)C(=O)N(C)1", make_3d=False)

# Get atom labels, generate graphs and cleanup equivalent
atoms, bonds = gr.get_bonds(mol)
counts = {e: 0 for e in np.unique(atoms)}
graphs = {}
neighbour_to_discard = []
for i, atom in enumerate(atoms):

    g, env = gr.generate_graph(atoms, bonds, i, max_w, elems=allowed_elems)

    # If we have a proton, check if it is a methyl
    if atom == "H":
        if len(bonds[i]) != 1:
            raise ValueError("Unexpected number of bonds to a hydrogen atom!")
        nei = bonds[i][0]
        if nei not in neighbour_to_discard:
            nei_num_h_connected = len([atoms[j] for j in bonds[nei] if atoms[j] == "H"])
            if nei_num_h_connected >= 3:
                neighbour_to_discard.append(nei)
                graphs[f"{atom}{counts[atom]+1}"] = (atom, [j for j in bonds[nei] if atoms[j] == "H"], g, env)
                counts[atom] += 1
            else:
                graphs[f"{atom}{counts[atom]+1}"] = (atom, [i], g, env)
                counts[atom] += 1

    else:
        graphs[f"{atom}{counts[atom]+1}"] = (atom, [i], g, env)
        counts[atom] += 1

for g in graphs:
    gr.print_graph(graphs[g][2], max_w, show=False, save=f"{out_dir}graph_{g}.pdf")

elems = []
labels = []
coords = []

class Plotter():

    def __init__(self, mol, graphs, bond_length_factor=0.3, figscale=0.75):

        self.mol = mol
        self.gs = graphs
        self.lf = bond_length_factor
        self.fs = figscale

        self.l0 = 1e12
        for bond in ob.OBMolBondIter(self.mol.OBMol):
            a1 = bond.GetBeginAtomIdx() - 1
            a2 = bond.GetEndAtomIdx() - 1
            btype = bond.GetBondOrder()
            x1, y1, _ = self.mol.atoms[a1].coords
            x2, y2, _ = self.mol.atoms[a2].coords
            d = np.linalg.norm([x2-x1, y2-y1])
            if d < self.l0:
                self.l0 = d
        
        self.hig_color = "gold"
        self.sel_color = "darkorange"
        self.shi_color = "orangered"

        self.plot()

        return
    
    def get_figsize(self):

        xmin = self.mol.atoms[0].coords[0]
        xmax = self.mol.atoms[0].coords[0]
        ymin = self.mol.atoms[0].coords[1]
        ymax = self.mol.atoms[0].coords[1]
        for atom in self.mol.atoms:
            x, y, _ = atom.coords
            if x > xmax:
                xmax = x
            if x < xmin:
                xmin = x
            if y > ymax:
                ymax = y
            if y < ymin:
                ymin = y

        dx = xmax-xmin
        dy = ymax-ymin
        return dx*self.fs, dy*self.fs
    


    def plot(self):

        self.fig = plt.figure(figsize=self.get_figsize())
        self.ax = self.fig.add_subplot(1,1,1)

        self.sites = []
        # First plot the hover-able objects
        for atom in self.mol.atoms:
            x, y, _ = atom.coords
            scatter = self.ax.scatter(x, y, marker="o", c="w", s=200, zorder=1)
            self.sites.append(scatter)
        
        self.elems = []
        self.coords = []
        # Plot atomic types
        for i, atom in enumerate(self.mol.atoms):
            x, y, _ = atom.coords
            for g in self.gs:
                if i in self.gs[g][1]:
                    label = self.gs[g][0]
                    break
            self.ax.text(x, y, g, ha="center", va="center", zorder=2)
            self.coords.append([x, y])
            if label not in self.elems:
                self.elems.append(label)
        self.coords = np.array(self.coords)

        # Plot bonds
        for bond in ob.OBMolBondIter(self.mol.OBMol):
            a1 = bond.GetBeginAtomIdx() - 1
            a2 = bond.GetEndAtomIdx() - 1
            btype = bond.GetBondOrder()
            x1, y1, _ = self.mol.atoms[a1].coords
            x2, y2, _ = self.mol.atoms[a2].coords
            if btype == 1:
                self.ax.plot([x1, x2], [y1, y2], "k", linewidth=1., zorder=0)
            elif btype == 2:
                dx = (y2 - y1) / 30.  # Offset distance for double bond
                dy = (x1 - x2) / 30.  # Offset distance for double bond
                self.ax.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], "k", linewidth=1., zorder=0)  # First line
                self.ax.plot([x1 - dx, x2 - dx], [y1 - dy, y2 - dy], "k", linewidth=1., zorder=0)  # Second line
            elif btype == 3:
                dx = (y2 - y1) / 15.  # Offset distance for double bond
                dy = (x1 - x2) / 15.  # Offset distance for double bond
                self.ax.plot([x1, x2], [y1, y2], "k", linewidth=1.)  # First line
                self.ax.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], "k", linewidth=1., zorder=0)  # Second line
                self.ax.plot([x1 - dx, x2 - dx], [y1 - dy, y2 - dy], "k", linewidth=1., zorder=0)  # Third line
        
        # Add button
        bax = self.fig.add_axes([0.8, 0.05, 0.1, 0.075])
        bnext = wid.Button(bax, "Next")
        bnext.on_clicked(self.press_next)

        # Fine-tune plot aspect
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')

        self.highlighted = []
        self.selected = []

        def onhover(event):

            if event.inaxes == self.ax:
                x = event.xdata
                y = event.ydata
                i = self.corresponding_atom(np.array([x, y]))

                for j in self.highlighted:
                    if j not in self.selected:
                        self.sites[j].set_facecolor("w")
                    else:
                        self.sites[j].set_facecolor(self.sel_color)
                
                if i is not None:
                    for g in self.gs:
                        if i in self.gs[g][1]:
                            self.highlighted = self.gs[g][1]
                            break
                else:
                    self.highlighted = []
                        
                for j in self.highlighted:
                    if j in self.selected:
                        self.sites[j].set_facecolor(self.shi_color)
                    else:
                        self.sites[j].set_facecolor(self.hig_color)

                plt.draw()

            return
        
        def onclick(event):

            if event.inaxes == self.ax:
                x = event.xdata
                y = event.ydata
                i = self.corresponding_atom(np.array([x, y]))

                if i is not None:

                    for g in self.gs:
                        if i in self.gs[g][1]:
                            js = self.gs[g][1]
                            break
                    
                    if i in self.selected:
                        for j in js:
                            self.sites[j].set_facecolor(self.hig_color)
                            self.selected.remove(j)
                    
                    else:
                        for j in js:
                            self.sites[j].set_facecolor(self.sel_color)
                            self.selected.append(j)

                plt.draw()

            return
        
        def select_element(elem):

            for i in self.selected:
                self.sites[i].set_facecolor("w")
            
            self.selected = []
            for g in self.gs:
                if self.gs[g][0] == elem:
                    print(g, self.gs[g][2])
                    self.selected.extend(self.gs[g][1])

            for i in self.selected:
                self.sites[i].set_facecolor(self.sel_color)
            
            plt.draw()

            return
        
        # Create buttons for each element
        buttons = []
        for i, elem in enumerate(self.elems):  # Example elements
            button = wid.Button(plt.axes([0.1*i, 0.9, 0.1, 0.05]), elem)
            button.on_clicked(lambda _, e=elem: select_element(e))
            buttons.append(button)

        self.fig.canvas.mpl_connect("motion_notify_event", onhover)
        self.fig.canvas.mpl_connect('button_press_event', onclick)

        # Display the plot
        plt.show()

        return
    


    def corresponding_atom(self, pos):

        ds = np.zeros(len(self.mol.atoms))
        for i, x in enumerate(self.coords):
            ds[i] = np.linalg.norm(pos - x)
        
        if np.min(ds) > self.l0 * self.lf:
            return None
        return np.argmin(ds)
    


    def press_next(self, _):
        plt.close()
        return

plotter = Plotter(mol, graphs)

print(plotter.selected)

quit()

# Create a figure and axis for plotting
fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(1,1,1)

elems = []
labels = []
coords = []
# Iterate over the atoms and bonds
for atom in mol.atoms:
    x, y, _ = atom.coords
    elem = ase.data.chemical_symbols[atom.atomicnum]
    ax.scatter(x, y, marker="o", c="w", s=200, zorder=1)
    ax.text(x, y, elem, ha="center", va="center", zorder=2)
    labels.append(f"{elem}{elems.count(elem)+1}")
    elems.append(elem)
    coords.append((x, y))

for bond in ob.OBMolBondIter(mol.OBMol):
    a1 = bond.GetBeginAtomIdx() - 1
    a2 = bond.GetEndAtomIdx() - 1
    btype = bond.GetBondOrder()
    x1, y1, _ = mol.atoms[a1].coords
    x2, y2, _ = mol.atoms[a2].coords
    if btype == 1:
        ax.plot([x1, x2], [y1, y2], "k", linewidth=1., zorder=0)
    elif btype == 2:
        dx = (y2 - y1) / 30.  # Offset distance for double bond
        dy = (x1 - x2) / 30.  # Offset distance for double bond
        ax.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], "k", linewidth=1., zorder=0)  # First line
        ax.plot([x1 - dx, x2 - dx], [y1 - dy, y2 - dy], "k", linewidth=1., zorder=0)  # Second line
    elif btype == 3:
        dx = (y2 - y1) / 15.  # Offset distance for double bond
        dy = (x1 - x2) / 15.  # Offset distance for double bond
        ax.plot([x1, x2], [y1, y2], "k", linewidth=1.)  # First line
        ax.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], "k", linewidth=1., zorder=0)  # Second line
        ax.plot([x1 - dx, x2 - dx], [y1 - dy, y2 - dy], "k", linewidth=1., zorder=0)  # Third line

ax.set_xticks([])
ax.set_yticks([])
# Set aspect ratio to equal
ax.set_aspect('equal')

def onclick(event):
    if event.inaxes == ax:  # Check if the click occurred within the plot axes
        x = event.xdata  # X-coordinate of the click
        y = event.ydata  # Y-coordinate of the click
        print(f"Clicked at x = {x}, y = {y}")

fig.canvas.mpl_connect('button_press_event', onclick)

# Display the plot
plt.show()
