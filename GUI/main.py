import PySimpleGUI as sg


sg.theme('SystemDefault1')
sg.set_options(font=("Courier New", 12))

#Enter Twitter username
def make_win1():
    layout = [[sg.Text("Input Username", size=(20, 1)), sg.Input(size=(30, 1), key='-USERNAME-')],
          [sg.Column([[sg.Button("Exit"), sg.Button("Find User")]], justification='center')],
          [sg.StatusBar("", size=(0, 1), key='-STATUS-')]]
    return sg.Window('Welcome', layout, finalize=True)

#Finding username using Twint
def make_win2():
    layout = [[sg.Text('Finding ' + values['-USERNAME-'])],
              [sg.Text('Please wait...')],
              [sg.Text(size=40)],
              [sg.Button('Cancel')]]
    return sg.Window('Finding user', layout, finalize=True)

window1, window2 = make_win1(), None

while True:             # Event Loop
    window, event, values = sg.read_all_windows()
    if event == sg.WIN_CLOSED or event == 'Exit':
        window.close()
        if window == window2:       # if closing win 2, mark as closed
            window2 = None
        elif window == window1:     # if closing win 1, exit program
            break
    elif event == 'Find User' and not window2:
        username = values['-USERNAME-']
        if username: 
            window2 = make_win2()
        else:
            state = "Username required!"
            window['-STATUS-'].update(state)
    elif event == event == sg.WIN_CLOSED or event == 'Cancel':
        window.close()
    
window.close()
