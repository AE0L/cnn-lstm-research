from tkcalendar import Calendar, DateEntry
from tweepy import OAuthHandler
import tweepy
from tkinter import *
from tkinter.ttk import Label
from tkinter.constants import BOTH, BOTTOM, LEFT, RIGHT, TOP
from tkinter import ttk
import tkinter as tk


def extractTweet(userid, sinceDate, endDate):
    sinceDate = sinceDate.get_date()
    endDate = endDate.get_date()
    from extractor import DataExtract
    DataExtract.process(userid, sinceDate, endDate)


def selectDate(choice, userid):

    for widgets in user.winfo_children():
        widgets.destroy()
    root.update()

    if choice == "Okay":
        # Create a Label
        Label(user, text="Start Date", background='gray61', foreground="white", font=(
            "ariel 12"), justify=LEFT).pack(padx=20, pady=20)
        # Create a Calendar using DateEntry
        sinceDate = DateEntry(user, width=16, background="lightblue",
                              foreground="white", bd=2, date_pattern='mm/dd/yyyy')
        sinceDate.pack(pady=5)
        # Create a Label
        Label(user, text="End Date", background='gray61', foreground="white", font=(
            "ariel 12"), justify=LEFT).pack(padx=20, pady=20)
        # Create a Calendar using DateEntry
        endDate = DateEntry(user, width=16, background="lightblue",
                            foreground="white", bd=2, date_pattern='mm/dd/yyyy')
        endDate.pack(pady=5)

        cal_button = ttk.Button(user, text='Classify User', style='success.TButton',
                                command=lambda: extractTweet(userid, sinceDate, endDate))
        cal_button.pack(padx=7, pady=5)

    else:
        root.destroy()
        import main


def get_value():
    user_input = username_entry.get()
    verify(user_input)


def userDetails(screen_name, userid, accAuth):
    #global message_window
    for widgets in user.winfo_children():
        widgets.destroy()

    screenname_label = ttk.Label(
        user, text="Screen Name: " + str(screen_name), font=('Helvetica', 12), justify=LEFT)
    screenname_label.pack(padx=20, pady=15)

    userid_label = ttk.Label(user, text="User ID: " +
                             str(userid), font=('Helvetica', 12), justify=LEFT)
    userid_label.pack(padx=20, pady=15)

    accAuth_label = ttk.Label(
        user, text="User Auth: " + str(accAuth), font=('Helvetica', 12), justify=LEFT)
    accAuth_label.pack(padx=20, pady=15)

    back = ttk.Button(user, text="Back", style='success.TButton',
                      width=8, command=lambda: selectDate("Back", userid))
    back.pack(padx=5, pady=10, side=RIGHT)

    okay = ttk.Button(user, text="Okay", style='success.TButton',
                      width=8, command=lambda: selectDate("Okay", userid))
    okay.pack(pady=10, side=RIGHT)

    if accAuth == 'Private':
        okay['state'] = DISABLED


def verify(user_input):

    access_token = '1449676365893038088-pdugze1ET3LcXHZ4MhgW5KLtDNgi2k'
    access_token_secret = 'bT4xTEVL5IrCkv1MkrorKjzdID4RCPNVankEMmOlALNJL'
    consumer_key = 'fSqFCU8DvkTd8XitEaWNA3ASN'
    consumer_secret = 'fnSlukoXfY4JZtodkqpPZLL8uBg3stk7B9rzQch2tpbgP7ldkT'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    username = api.get_user(screen_name=user_input)
    screen_name = username.screen_name
    userid = username.id

    if username.protected == False:
        accAuth = "Public"

    else:
        accAuth = "Private"

    userDetails(screen_name, userid, accAuth)


root = tk.Tk()
root.geometry('500x500')
root.resizable(True, False)
root.title('Username')

username = tk.StringVar()

user = ttk.Frame(root)
user.pack(padx=110, pady=100, fill='x', expand=True)

username_label = ttk.Label(
    user, text="Enter username:", font=('Helvetica', 12))
username_label.pack(expand=True, side=TOP, fill=BOTH)

username_entry = ttk.Entry(user, textvariable=username, style='success.TEntry')
username_entry.pack(fill='x', expand=True, side=TOP)
username_entry.focus()


username_button = ttk.Button(
    user, text='Find User', style='success.TButton', command=get_value)
username_button.pack(padx=7, pady=5, side=BOTTOM)
root.mainloop()


def extractTweet(userid, sinceDate, endDate):
    sinceDate = sinceDate.get_date()
    endDate = endDate.get_date()
    from extractor import DataExtract
    DataExtract.process(userid, sinceDate, endDate)


def selectDate(choice, userid):

    for widgets in user.winfo_children():
        widgets.destroy()
    root.update()

    if choice == "Okay":
        # Create a Label
        Label(user, text="End Date", background='gray61', foreground="white", font=(
            "ariel 12"), justify=LEFT).pack(padx=20, pady=20)
        # Create a Calendar using DateEntry
        endDate = DateEntry(user, width=16, background="lightblue",
                            foreground="white", bd=2, date_pattern='mm/dd/yyyy')
        endDate.pack(pady=5)

        cal_button = ttk.Button(user, text='Classify User', style='success.TButton',
                                command=lambda: extractTweet(userid, sinceDate, endDate))
        cal_button.pack(padx=7, pady=5)

    else:
        root.destroy()
        import main


def get_value():
    user_input = username_entry.get()
    verify(user_input)


def userDetails(screen_name, userid, accAuth):
    #global message_window
    for widgets in user.winfo_children():
        widgets.destroy()

    screenname_label = ttk.Label(
        user, text="Screen Name: " + str(screen_name), font=('Helvetica', 12), justify=LEFT)
    screenname_label.pack(padx=20, pady=15)

    userid_label = ttk.Label(user, text="User ID: " +
                             str(userid), font=('Helvetica', 12), justify=LEFT)
    userid_label.pack(padx=20, pady=15)

    accAuth_label = ttk.Label(
        user, text="User Auth: " + str(accAuth), font=('Helvetica', 12), justify=LEFT)
    accAuth_label.pack(padx=20, pady=15)

    back = ttk.Button(user, text="Back", style='success.TButton',
                      width=8, command=lambda: selectDate("Back", userid))
    back.pack(padx=5, pady=10, side=RIGHT)

    okay = ttk.Button(user, text="Okay", style='success.TButton',
                      width=8, command=lambda: selectDate("Okay", userid))
    okay.pack(pady=10, side=RIGHT)

    if accAuth == 'Private':
        okay['state'] = DISABLED


def verify(user_input):

    access_token = '1449676365893038088-pdugze1ET3LcXHZ4MhgW5KLtDNgi2k'
    access_token_secret = 'bT4xTEVL5IrCkv1MkrorKjzdID4RCPNVankEMmOlALNJL'
    consumer_key = 'fSqFCU8DvkTd8XitEaWNA3ASN'
    consumer_secret = 'fnSlukoXfY4JZtodkqpPZLL8uBg3stk7B9rzQch2tpbgP7ldkT'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    username = api.get_user(screen_name=user_input)
    screen_name = username.screen_name
    userid = username.id

    if username.protected == False:
        accAuth = "Public"

    else:
        accAuth = "Private"

    userDetails(screen_name, userid, accAuth)


root = tk.Tk()
root.geometry('500x500')
root.resizable(True, False)
root.title('Username')

username = tk.StringVar()

user = ttk.Frame(root)
user.pack(padx=110, pady=100, fill='x', expand=True)

username_label = ttk.Label(
    user, text="Enter username:", font=('Helvetica', 12))
username_label.pack(expand=True, side=TOP, fill=BOTH)

username_entry = ttk.Entry(user, textvariable=username, style='success.TEntry')
username_entry.pack(fill='x', expand=True, side=TOP)
username_entry.focus()


username_button = ttk.Button(
    user, text='Find User', style='success.TButton', command=get_value)
username_button.pack(padx=7, pady=5, side=BOTTOM)
root.mainloop()
