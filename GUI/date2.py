import PySimpleGUI as sg

sg.theme('Dark Blue 3')  # please make your windows colorful

layout = [[sg.Text('Rename files or folders')],
      [sg.InputText(), sg.CalendarButton('Start Date')],
      [sg.InputText(), sg.CalendarButton('End Date')],
      [sg.Submit(), sg.Cancel()]]
# layout = [[sg.Text('Rename files or folders')],
      
        
window = sg.Window('Rename Files or Folders', layout)

event, values = window.read()
window.close()
folder_path, file_path = values[0], values[1]       # get the data from the values dictionary
print(folder_path, file_path)
