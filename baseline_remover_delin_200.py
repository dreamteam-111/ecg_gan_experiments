__author__ = "Sereda"

import os
import json
import BaselineWanderRemoval as bwr

def bwr_delin_dataset():
    print("Open json data, eliminate baseline wonder, than save all back to json file in new location")

    OLD_PATH = "C:\\!mywork\\datasets\\ecg_200_delineation\\"
    RESULT_PATH = "C:\\!mywork\\datasets\\BWR_ecg_200_delineation\\"
    LEADS_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    SIGNAL_LEN = 5000
    FREQUENCY_OF_DATASET = 500
    FILENAME = "ecg_data_200.json"

    os.makedirs(RESULT_PATH, exist_ok=True)
    problems = 0
    old_file_path = OLD_PATH + FILENAME
    result_file_path = RESULT_PATH + FILENAME

    print("Start processing of " + old_file_path + "...")
    with open(old_file_path, 'r') as f:
        data = json.load(f)

        for case_id in data.keys():
            for lead_name in LEADS_NAMES:
                signal = data[case_id]['Leads'][lead_name]['Signal']
                if (len(signal)) != SIGNAL_LEN:
                    print("problem with " + str(case_id))
                    problems += 1
                    continue
                fixed_signal = bwr.fix_baseline_wander(signal, FREQUENCY_OF_DATASET)
                data[case_id]['Leads'][lead_name]['Signal'] = fixed_signal
        print("saving to " + result_file_path +"...")
        with open(result_file_path, 'w') as outfile:
            json.dump(data, outfile)
            print("json was saved, " + str(problems) + " problems found")




if __name__ == "__main__":
    bwr_delin_dataset()