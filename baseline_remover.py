__author__ = "Sereda"

import os
import json
import BaselineWanderRemoval as bwr

def bwr_schiller_dataset():
    print("we will open raw json data, eliminate baseline wonder, than save all back to json files in new location")
    OLD_PATH = "C:\\!mywork\\datasets\\data_schiller\\"
    RESULT_PATH = "C:\\!mywork\\datasets\\BWR_data_schiller\\"
    LEADS_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    SIGNAL_LEN = 5000
    FREQUENCY_OF_DATASET = 500

    os.makedirs(RESULT_PATH, exist_ok=True)
    problems = 0
    for part_id in range(0, 20):
        part_path = OLD_PATH + "data_part_" + str(part_id) + ".json"
        result_part_path = RESULT_PATH + "data_part_" + str(part_id) + ".json"

        print("start processing of " + part_path + "...")
        with open(part_path, 'r') as f:
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
            print("saving to " + result_part_path +"...")
            with open(result_part_path, 'w') as outfile:
                json.dump(data, outfile)
                print("part was saved, " + str(problems) + " problems found")
    print("All done, " + str(problems) + " problems found!")



if __name__ == "__main__":
    bwr_schiller_dataset()