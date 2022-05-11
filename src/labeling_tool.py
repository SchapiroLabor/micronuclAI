# labeling_tool.py

import PySimpleGUI as sg
from PIL import ImageTk, Image
import pandas as pd
import os
from os.path import abspath, join



def main():
    QT_ENTER_KEY1 =  'special 16777220'
    QT_ENTER_KEY2 =  'special 16777221'

    # Create layout
    layout = [
        [sg.Text("Press KEYs from 0-9 to specify the number of micronuclei on the image", key="-TEXT-")],
        [sg.Image(key="-IMAGE-")],
        [sg.Button("UNDERSTOOD", key="-OK-")]
    ]
    # Create window
    window = sg.Window(title="Labeling tool", layout=layout, return_keyboard_events=True, finalize=True)

    # Pathway to list of files
    dir = abspath("../data/mesmer_sc")
    filelist = [join(dir,f) for f in os.listdir(dir)]

    out = []
    i = 0

    # Create an event loop

    while True:
        event, values = window.read()

        # Close window
        if event == sg.WIN_CLOSED:
            break
        elif event in "0123456789":
            out.append(event)
            i+=1
            if i >= len(filelist):
                break
            window["-IMAGE-"].update(filename=filelist[i],)
        elif event == "-OK-":
            window["-IMAGE-"].update(filename=filelist[i],)

    window.close()
    out_data = pd.DataFrame(out, index=filelist, columns=["n_micronuclei"])
    out_data.to_csv(abspath("../out/labels_mc_mesmer_miguel.csv"))

if __name__ == '__main__':
    main()
