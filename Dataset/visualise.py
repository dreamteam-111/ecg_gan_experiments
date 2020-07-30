__author__ = "Sereda"
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import torch
import numpy as np
import os

def plot_ecg_to_fig(ecg, leads_names):
    numleads = len(leads_names)
    fig, axs = plt.subplots(numleads, 1, figsize=(8, 2*numleads), sharex=True, sharey=True)
    axs = axs.ravel()
    #fig.suptitle(title)
    for i in range(numleads):
        name = leads_names[i]
        signal = get_lead_signal(ecg, name)
        axs[i].plot(signal)
        axs[i].set_title(name)

    return fig

def get_lead_signal(ecg, lead_name):
    return ecg['Leads'][lead_name]['Signal']

def get_ecg_description(ecg):
    return ecg["TextDiagnosisDoc"]

def get_ecg_desaeces(ecg):
    return ""

def draw_ecgs_to_htlm(json_data, ecgs_ids, name_html, folder,
                      leads=['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
                      message=""):
    html = "<!DOCTYPE HTML><html><head> </head><body>"
    html += message

    os.makedirs(folder, exist_ok=True)
    for ecg_id in ecgs_ids:
        html +="<hr>"
        ecg = json_data[ecg_id]
        title = "<p>"+ get_ecg_description(ecg) + "</p>"
        html += title
        fig = plot_ecg_to_fig(ecg, leads_names=leads[:3] )
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + 'descrition.. <br>'
        plt.close(fig)

    # Save html
    html += "</body></html>"
    filename = folder + "/" + name_html
    with open(filename, 'w') as f:
        f.write(html)

