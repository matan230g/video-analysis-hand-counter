from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import *


class PlotWidget:
    """This class is tkinter widget that show matplotlib figure on canvas """
    def __init__(self, parent, fig):
        if fig is None:
            messagebox.showinfo('Error', 'There is not Plot to display')
            return
        self.parent = parent
        self.button = Button(parent, text="save", command=self.save_fig)
        self.button.pack()
        self.fig = fig
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

    def save_fig(self):
        plt.savefig('output.png')

