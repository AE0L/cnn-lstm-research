import tkinter as tk
from tkinter import ttk
from tkinter.constants import BOTH, BOTTOM, LEFT, RIGHT, TOP
from tkinter.ttk import Label
from tkinter import *

root = tk.Tk()
root.geometry('500x500')
root.resizable(True, False)
root.title('Username')

def select(choice):
    if choice=="Okay":
        root.destroy()
        import ScanSettings
    else:
        root.destroy()
        import welcome

def get_value():
    user_input=username_entry.get()

def new():
    global message_window
    message_window = Toplevel()
    message_window.title("This is Custom messagebox")
    label = Label(message_window, text="Handle name:",
                  font=("ariel 12"), justify=LEFT)
    label.pack(padx=20, pady=15)
    label = Label(message_window, text="Name:",
                  font=("ariel 12"), justify=LEFT)
    label.pack(padx=20, pady=15)
    label = Label(message_window, text="Private:",
                  font=("ariel 12"), justify=LEFT)
    label.pack(padx=20, pady=15)
    back = Button(message_window, text="Back", font=("ariel 15 bold"), width=8, relief=GROOVE,
                bd=3, command=lambda: select("Back"))
    back.pack(padx=5, pady=10, side=RIGHT)
    okay = Button(message_window, text="Okay", font=("ariel 15 bold"), width=8, relief=GROOVE,
                 bd=3, command=lambda: select("Okay"))
    okay.pack(pady=10, side=RIGHT)


username = tk.StringVar()

user = ttk.Frame(root)
user.pack(padx=110, pady=100, fill='x', expand=True)

username_label = ttk.Label(user, text="Enter username:", font = ('Helvetica', 12))
username_label.pack( expand=True, side=TOP, fill=BOTH)

username_entry = ttk.Entry(user, textvariable=username, style='success.TEntry')
username_entry.pack(fill='x', expand=True, side=TOP)
username_entry.focus()

username_button = ttk.Button(user, text='Find User', style='success.TButton', command=new)
username_button.pack( padx=7, pady=5, side=BOTTOM)
root.mainloop()
