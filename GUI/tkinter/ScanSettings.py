import tkinter as tk
from tkinter import ttk
from tkinter.constants import BOTH, BOTTOM, LEFT, RIGHT, TOP
from tkinter.ttk import Label
from tkcalendar import Calendar,DateEntry


root = tk.Tk()
root.geometry('500x500')
root.resizable(False, False)
root.title('Scan Settings')

def ScanSettings():
    root.destroy()
    import ScanSettings

def loadPage():
    root.destroy()
    import loadPage
def scanUser():
    root.destroy()
    import scanUser


#Set the Geometry
root.title("Scan Settings")
#Create a Label
Label(root, text= "Start Date", background= 'gray61', foreground="white", font=("ariel 12"), justify=LEFT).pack(padx=20,pady=20)
#Create a Calendar using DateEntry
cal = DateEntry(root, width= 16, background= "magenta3", foreground= "white",bd=2)
cal.pack(pady=5)
#Create a Label
Label(root, text= "End Date", background= 'gray61', foreground="white", font=("ariel 12"), justify=LEFT).pack(padx=20,pady=20)
#Create a Calendar using DateEntry
cal = DateEntry(root, width= 16, background= "magenta3", foreground= "white",bd=2)
cal.pack(pady=5)

cal_button = ttk.Button(root, text='Classify User', style='success.TButton', command=scanUser)
cal_button.pack( padx=7, pady=5)
root.mainloop()
