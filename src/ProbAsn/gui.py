import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.widgets as wid
from openbabel import openbabel as ob
from openbabel import pybel as pb
from ProbAsn import utils as ut
from ProbAsn import graph as gr
import ase
mpl.use("TkAgg")

import tkinter as tk
from tkinter import ttk
import functools as ft
import matplotlib.backends.backend_tkagg as tka

class Molecule:
    def __init__(self, mol, in_type="smi", from_file=False, max_w=6, allowed_elems=["H", "C", "N", "O", "S", "F", "P", "Cl", "Na", "Ca", "Mg", "K"]):

        self.mol = gr.make_mol(
            mol,
            in_type=in_type,
            from_file=from_file,
            )

        self.max_w = max_w
        self.allowed_elems = allowed_elems
        self.atoms, self.bonds = gr.get_bonds(self.mol)

        self.elems = np.unique(self.atoms)

        self.graphs = None

        return
    
    def make_graphs(self):

        counts = {e: 0 for e in np.unique(self.atoms)}

        self.graphs = {}
        neighbour_to_discard = []
        for i, atom in enumerate(self.atoms):

            g, env = gr.generate_graph(self.atoms, self.bonds, i, self.max_w, elems=self.allowed_elems)

            # If we have a proton, check if it is a methyl
            if atom == "H":
                if len(self.bonds[i]) != 1:
                    raise ValueError("Unexpected number of bonds to a hydrogen atom!")
                nei = self.bonds[i][0]
                if nei not in neighbour_to_discard:
                    nei_num_h_connected = len([self.atoms[j] for j in self.bonds[nei] if self.atoms[j] == "H"])
                    if nei_num_h_connected >= 3:
                        neighbour_to_discard.append(nei)
                        self.graphs[f"{atom}{counts[atom]+1}"] = (atom, [j for j in self.bonds[nei] if self.atoms[j] == "H"], g, env)
                        counts[atom] += 1
                    else:
                        self.graphs[f"{atom}{counts[atom]+1}"] = (atom, [i], g, env)
                        counts[atom] += 1

            else:
                self.graphs[f"{atom}{counts[atom]+1}"] = (atom, [i], g, env)
                counts[atom] += 1

        return
    
    def get_graphs(self):
        if self.graphs is None:
            self.make_graphs()
        return self.graphs

class MolPlotter:

    def __init__(self, mol=None, allow_select=False, bond_length_factor=0.3, figscale=0.75):

        self.molecule = mol
        if mol is not None:
            self.mol = mol.mol
        self.lf = bond_length_factor
        self.fs = figscale
        self.allow_select = allow_select

        self.highlighted = []
        self.selected = []
        
        self.hig_color = "gold"
        self.sel_color = "darkorange"
        self.shi_color = "orangered"

        self.init_plot()

        return
    
    def update_mol(self, mol):

        self.molecule = mol
        self.mol = mol.mol

        self.l0 = 1e12
        for bond in ob.OBMolBondIter(self.mol.OBMol):
            a1 = bond.GetBeginAtomIdx() - 1
            a2 = bond.GetEndAtomIdx() - 1
            x1, y1, _ = self.mol.atoms[a1].coords
            x2, y2, _ = self.mol.atoms[a2].coords
            d = np.linalg.norm([x2-x1, y2-y1])
            if d < self.l0:
                self.l0 = d

        return
    
    def get_figsize(self):

        if self.molecule is None:
            return 6, 5

        else:
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
    
    def get_figlims(self, extend=0.1):

        if self.molecule is None:
            return [0, 0], [0, 0]

        else:
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
            return [xmin - extend*dx, xmax + extend*dx], [ymin - extend*dy, ymax + extend*dy]
    
    def init_plot(self):
        self.fig = plt.figure(figsize=self.get_figsize())
        self.ax = self.fig.add_subplot(1,1,1)

        self.ax.clear()

        return

    def plot(self):

        self.ax.clear()

        #w, h = self.get_figsize()
        #self.fig.set_size_inches(w, h)

        self.sites = []
        # First plot the hover-able objects
        for atom in self.mol.atoms:
            x, y, _ = atom.coords
            scatter = self.ax.scatter(x, y, marker="o", c="w", s=200, zorder=1, edgecolors="k", linewidths=0.5)
            self.sites.append(scatter)
        
        self.elems = []
        self.coords = []
        # Plot atomic types
        for i, atom in enumerate(self.mol.atoms):
            x, y, _ = atom.coords
            label = ase.data.chemical_symbols[atom.atomicnum]
            self.ax.text(x, y, label, ha="center", va="center", zorder=2, fontsize=8)
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

        # Fine-tune plot aspect
        xlim, ylim = self.get_figlims()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')
        self.fig.draw_without_rendering()
        tb = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
        self.fig.set_size_inches(tb.width, tb.height)
        self.fig.tight_layout()

        return
    
    def plot_selectable(self):

        self.ax.clear()
        self.gs = self.molecule.get_graphs()

        #w, h = self.get_figsize()
        #self.fig.set_size_inches(w, h)

        self.sites = []
        # First plot the hover-able objects
        for atom in self.mol.atoms:
            x, y, _ = atom.coords
            scatter = self.ax.scatter(x, y, marker="o", c="w", s=200, zorder=1, edgecolors="k", linewidths=0.5)
            self.sites.append(scatter)
        
        self.elems = []
        self.coords = []
        # Plot atomic types
        for i, atom in enumerate(self.mol.atoms):
            x, y, _ = atom.coords
            for g in self.gs:
                if i in self.gs[g][1]:
                    break
            self.ax.text(x, y, g, ha="center", va="center", zorder=2, fontsize=6)
            self.coords.append([x, y])
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

        # Fine-tune plot aspect
        xlim, ylim = self.get_figlims()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')
        self.fig.draw_without_rendering()
        tb = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
        self.fig.set_size_inches(tb.width, tb.height)
        self.fig.tight_layout()

        self.highlighted = []
        self.selected = []

        # Create buttons for each element
        buttons = []
        for i, elem in enumerate(self.elems):  # Example elements
            button = wid.Button(plt.axes([0.1*i, 0.9, 0.1, 0.05]), elem)
            button.on_clicked(lambda _, e=elem: self.select_element(e))
            buttons.append(button)

        self.fig.canvas.mpl_connect("motion_notify_event", self.onhover)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        return

    def onhover(self, event):

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
    
    def onclick(self, event):

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
    
    def select_element(self, elem):

        for i in self.selected:
            self.sites[i].set_facecolor("w")
        
        self.selected = []
        for g in self.gs:
            if self.gs[g][0] == elem:
                self.selected.extend(self.gs[g][1])

        for i in self.selected:
            self.sites[i].set_facecolor(self.sel_color)
        
        plt.draw()

        return
    
    def corresponding_atom(self, pos):

        ds = np.zeros(len(self.mol.atoms))
        for i, x in enumerate(self.coords):
            ds[i] = np.linalg.norm(pos - x)
        
        if np.min(ds) > self.l0 * self.lf:
            return None
        return np.argmin(ds)
    
    def close(self):

        plt.close()

        return



class Content(tk.Frame):
    def __init__(self, parent, bg_color="white"):
        super().__init__(parent.root)
        self.parent = parent
        self.bg_color = bg_color
        self.configure(bg=self.bg_color)
        self.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(self, borderwidth=0, background=self.bg_color)
        self.frame = tk.Frame(self.canvas, borderwidth=0, background=self.bg_color)
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas_window = self.canvas.create_window((4, 4), window=self.frame, anchor="nw", tags="self.frame")
        self.canvas.bind('<Configure>', self._resize_content)
        #self.frame.pack(side="top", fill="x", expand=True)

        self.frame.bind("<Configure>", self._on_frame_configure)
        self._set_mousewheel(self.canvas, self._on_mousewheel)
        
        for icol in range(2):
            self.frame.columnconfigure(icol, weight=1)
        for irow in range(100):
            self.frame.rowconfigure(irow, weight=1)
        return
    
    def init_step(self):

        # Clear frame
        for widget in self.frame.winfo_children():
            widget.destroy()
        
        irow = 0
        
        # Step 0 - Input structure
        if self.parent.params["cur_step"] == 0:
        
            # Initialise variable
            self.smiles_txt = tk.StringVar()
            # Add textbox for SMILES
            self.smiles_text = tk.Label(self.frame, text="Input the SMILES structure:", bg=self.bg_color)
            self.smiles_input = tk.Entry(self.frame, textvariable=self.smiles_txt, bg=self.bg_color, bd=2, highlightbackground=self.bg_color)
            self.smiles_button = tk.Button(self.frame, text="Confirm", command=ft.partial(self.update_mol, True), highlightbackground=self.bg_color)
            self.smiles_text.grid(row=irow,column=0, sticky="e")
            self.smiles_input.grid(row=irow,column=1, sticky="ew")
            self.smiles_input.focus()
            irow += 1
            self.smiles_button.grid(row=irow, column=0, columnspan=2)
            irow += 1
            self.smiles_input.bind('<Return>',ft.partial(self.update_mol, True))

            # Add separator
            self.sep1 = ttk.Separator(self.frame, orient="horizontal")
            self.sep1.grid(row=irow, column=0, columnspan=2, pady=5, sticky="ew")
            irow += 1

            # Add file upload
            self.upload_text = tk.Label(self.frame, text="Or upload a file:", bg=self.bg_color)
            self.upload_filetext = tk.Label(self.frame, text="No file selected", bg=self.bg_color)
            self.upload_button = tk.Button(self.frame, text="Browse...", command=ft.partial(self.browse_file, "in_file", self.upload_filetext, self.parent.params["in_types"]), highlightbackground=self.bg_color)
            self.upload_text.grid(row=irow, column=0, sticky="e")
            self.upload_button.grid(row=irow, column=1, sticky="w")
            irow += 1
            self.upload_filetext.grid(row=irow, column=0, columnspan=2, sticky="ew")
            irow += 1
            #TODO: Solve entry going back to grey after pressing index button

            # Add separator
            self.sep2 = ttk.Separator(self.frame, orient="horizontal")
            self.sep2.grid(row=irow, column=0, columnspan=2, pady=5, sticky="ew")
            irow += 1

            # Add plot of structure
            self.mol_row = irow
            if self.parent.mol is None:
                self.mol_canvas = None
            else:
                self.mplot = MolPlotter(mol=self.parent.mol)
                self.mplot.update_mol(self.parent.mol)
                self.mplot.init_plot()
                self.mplot.plot()
                self.mol_canvas = tka.FigureCanvasTkAgg(self.mplot.fig, master=self.frame)
                self.mol_canvas.get_tk_widget().grid(row=self.mol_row, column=0, columnspan=2, padx=5, pady=5)
                self.mol_canvas.draw()

            irow += 1

        if self.parent.params["cur_step"] == 1:

            # Add plot of structure
            self.mplot = MolPlotter(mol=self.parent.mol)
            self.mplot.update_mol(self.parent.mol)
            self.mplot.init_plot()
            self.mplot.plot_selectable()
            self.mol_canvas = tka.FigureCanvasTkAgg(self.mplot.fig, master=self.frame)
            self.mol_canvas.get_tk_widget().grid(row=irow, column=0, columnspan=2, padx=5, pady=5)
            self.mol_canvas.draw()
            irow += 1

            # Add element selection buttons
            self.elem_btns = []
            for elem in self.parent.mol.elems:
                self.elem_btns.append(tk.Button(self.frame, text=f"Select all {elem}", highlightbackground=self.bg_color, bg=self.bg_color, command=ft.partial(self.mplot.select_element, elem)))
                self.elem_btns[-1].grid(row=irow, column=0, columnspan=2)
                irow += 1

            # Add parameters
            self.db_label = tk.Label(self.frame, text=f"Database directory:", bg=self.bg_color)
            self.db_path = tk.Label(self.frame, text=self.parent.params["db_dir"], bg=self.bg_color)
            self.db_browse = tk.Button(self.frame, text="Browse...", command=ft.partial(self.browse_dir, "db_dir", self.db_path), highlightbackground=self.bg_color)
            self.db_label.grid(row=irow, column=0, sticky="e")
            self.db_browse.grid(row=irow, column=1, sticky="ew")
            irow += 1
            self.db_path.grid(row=irow, column=0, columnspan=2, sticky="ew")
            irow += 1

            # Add separator
            self.sep2 = ttk.Separator(self.frame, orient="horizontal")
            self.sep2.grid(row=irow, column=0, columnspan=2, pady=5, sticky="ew")
            irow += 1

        # Add next button
        self.next_btn = tk.Button(self.frame, text="Next", highlightbackground=self.bg_color, fg="blue", command=self.parent.next_step)
        self.next_btn.grid(row=irow,column=0, columnspan=2)
        irow += 1

        return
    
    def _resize_content(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=max(event.width, self.parent.params["content_min_width"]))
        return

    def _on_frame_configure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _set_mousewheel(self, widget, command):
        """Activate / deactivate mousewheel scrolling when 
        cursor is over / not over the widget respectively."""
        widget.bind("<Enter>", lambda _: widget.bind_all('<MouseWheel>', command))
        widget.bind("<Leave>", lambda _: widget.unbind_all('<MouseWheel>'))
        return
    
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(2 * int(event.delta < 0) - 1, "units")
        return
    
    def browse_dir(self, par, txt):
        dir = tk.filedialog.askdirectory()
        if not dir.endswith("/"):
            dir += "/"
        txt["text"] = dir
        self.parent.params[par] = dir
        return
    
    def browse_file(self, par, txt, filetypes):
        file = tk.filedialog.askopenfile(filetypes=filetypes)
        txt["text"] = file.name
        self.parent.params[par] = file.name
        if par == "in_file":
            self.update_mol(False)
        return

    def update_mol(self, *is_smi):

        if is_smi[0]:
            self.parent.mol = Molecule(self.smiles_txt.get())

        else:
            self.parent.mol = Molecule(self.parent.params["in_file"], in_type = self.parent.params["in_file"][-3:], from_file=True)

        if self.parent.mol is not None:
            if self.mol_canvas is not None:
                self.mol_canvas.get_tk_widget().destroy()
            self.mplot = MolPlotter(mol=self.parent.mol)
            self.mplot.update_mol(self.parent.mol)
            self.mplot.init_plot()
            self.mplot.plot()
            self.mol_canvas = tka.FigureCanvasTkAgg(self.mplot.fig, master=self.frame)
            self.mol_canvas.get_tk_widget().grid(row=self.mol_row, column=0, columnspan=2, padx=5, pady=5)
            self.mol_canvas.draw()
        
        self.parent.disable_next_steps()

        return


class Index(tk.Frame):
    def __init__(self, parent, width, bg_color):
        super().__init__(parent.root)
        # Configure panel
        self.configure(width=width, bg=bg_color)
        self.pack(side="left", fill="y")
        self.parent = parent
        self.bg_color = bg_color

        # Add step buttons
        self.btns = []
        for i, step in enumerate(self.parent.params["steps"]):
            self.btns.append(tk.Button(self, text=step, highlightbackground=bg_color, height=1, command=ft.partial(self.parent.update_step, i)))
            if i > self.parent.params["cur_step"]:
                self.btns[-1]["state"] = "disabled"
            self.btns[-1].pack(fill="x", side="top")
        
        return
    
    def update_cur_step(self, disable_next=False):
        for i in range(len(self.btns)):
            if i <= self.parent.params["cur_step"]:
                self.btns[i]["state"] = "normal"
            elif disable_next:
                self.btns[i]["state"] = "disabled"
        return



class Instructions(tk.Frame):
    def __init__(self, parent, width, bg_color):
        super().__init__(parent.root)
        # Configure panel
        self.configure(width=width, bg=bg_color)
        self.pack(side="right", fill="y")
        # Add widgets to the sidebar
        self.parent = parent
        return
    


class Application:
    def __init__(self, root, params):

        self.root = root
        self.root.title("Probabilistic assignment")
        self.root.geometry(f"{params['width']}x{params['height']}")
        self.params = params
        self.mol = None

        self.index_panel = Index(self, width=200, bg_color="gray")
        self.instruction_panel = Instructions(self, width=200, bg_color="lightgray")
        self.content = Content(self)

        self.update_step(0)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        return
    
    def next_step(self):

        step = self.params["cur_step"]

        #TODO: Check if the step is correctly done
        
        # Update step
        self.update_step(self.params["cur_step"] + 1)

        return
    
    def update_step(self, step, disable_next=False):
        self.params["cur_step"] = step
        self.index_panel.update_cur_step(disable_next=disable_next)
        self.content.init_step()
        return
    
    def disable_next_steps(self):
        self.index_panel.update_cur_step(disable_next=True)
        return

    def on_closing(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.destroy()
        quit()
        return