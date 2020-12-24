import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Combobox
from hand_gestures_to_data import HandGesturesToData
from PlotWidget import PlotWidget


class MainApplication(tk.Frame):
    """This class is the main class of the GUI."""
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.csv_output = IntVar()
        self.button_chose_file = Button(self, text="Choose a mp4 file!",
                                        font=("Helvetica", 9, "bold italic"), command=self.file_directory)
        self.button_chose_file.grid(row=1)
        self.video_path_entry = tk.Entry(self, state='disabled')
        self.video_path_entry.grid(row=1, column=1, columnspan=1)
        self.label_vert = Label(self, fg="black", text="How many people are in the zoom video vertically?",
                                font=("Helvetica", 9, "bold italic"))
        self.label_horiz = Label(self, fg="black", text="How many people are in the zoom video horizontally?",
                                 font=("Helvetica", 9, "bold italic"))
        self.label_vert.grid(row=2)
        self.label_horiz.grid(row=3)
        self.check_box_save_csv = Checkbutton(self, text="Output results to csv file?", fg="black",
                                              variable=self.csv_output)
        self.check_box_save_csv.grid(row=4, columnspan=3)

        self.button_final = Button(self, text="Tell me whats up!", fg="black", command=self.submit_final,
                                   font=("Helvetica", 16, "bold italic"))
        self.button_final.grid(row=6, columnspan=3)
        self.combo_horiz_divisions = Combobox(self, values=[1, 2], width=5)
        self.combo_vert_divisions = Combobox(self, values=[1, 2], width=5)
        self.combo_horiz_divisions.current(0)
        self.combo_vert_divisions.current(0)
        self.combo_horiz_divisions.grid(row=3, column=2)
        self.combo_vert_divisions.grid(row=2, column=2)

    def file_directory(self):
        """open file type of mp4 for video path"""
        filename = askopenfilename(filetypes=[("Mp4 files", "*.mp4")])
        self.video_path_entry.configure(state="normal")
        self.video_path_entry.insert(0, filename)
        self.video_path_entry.configure(state="disable")

    def submit_final(self):
        """Check the input, pass the args to the detector
        and call the detector object """
        video_path = self.video_path_entry.get()
        horiz_divisions = int(self.combo_horiz_divisions.get())
        vert_divisions = int(self.combo_vert_divisions.get())
        if len(video_path) == 0:
            messagebox.showinfo('Error', 'Please enter all required data!')
            return
        detector = HandGesturesToData(video_path, horiz_divisions, vert_divisions)
        # run the detector
        detector()
        print("Finish")
        fig = detector.show_plot_bar()
        self.show_bar_plot(fig)
        # if csv checkbox is True save the data to csv file
        if self.csv_output:
            detector.save_data_csv()

    @staticmethod
    def show_bar_plot(fig):
        """This static method show figure of the plot in anew windows """
        new_window = Toplevel(root)
        PlotWidget(new_window, fig)

        # sets the title of the
        # Toplevel widget
        new_window.title("Bar Plot")

        # sets the geometry of toplevel
        new_window.geometry("300x300")


if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.geometry('500x250')
    root.title("Welcome to hand recognition tool!")

    root.mainloop()
    # Toplevel object which will
    # be treated as a new window
