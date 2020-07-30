import json
__author__ = "Sereda"
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import torch
import numpy as np

from Dataset.visualise import draw_ecgs_to_htlm

class Query:
    """
    binary_features_names = ["isSinusRythm", "isRegularRhythm", "isNormalElectricalAxis"]
    diagnosys_names = ["regular_normosystole", "left_ventricular_hypertrophy",...]
    """
    def __init__(self, binary_features_names={},
                 diagnosys_names={},
                 HR_from=None, HR_to=None,
                 substrs=[]):
        self.binary_features_names = binary_features_names
        self.diagnosys_names = diagnosys_names
        self.heart_rate_from = HR_from
        self.heart_rate_to = HR_to
        self.substrs=substrs

    def is_query_ok(self, ecg_node):
        res = self.is_binarys_ok(ecg_node) \
              and self.is_diagnosis_ok(ecg_node) \
              and self.is_heart_rate_ok(ecg_node) \
              and self.contains_all_substrs(ecg_node)
        return res

    def contains_all_substrs(self, ecg_node):
        for substr in self.substrs:
            if substr not in ecg_node["TextDiagnosisDoc"]:
                return False
        return True

    def is_heart_rate_ok(self, ecg_node):
        HR = int(ecg_node["HeartRate"])
        if self.heart_rate_from is not None:
            if HR < self.heart_rate_from:
                return False
        if self.heart_rate_to is not None:
            if HR > self.heart_rate_to:
                return False
        return True

    def is_binarys_ok(self, ecg_node):
        for binary_feature_name in self.binary_features_names.keys():
            wanted_flag = self.binary_features_names[binary_feature_name]
            real_flag = ecg_node[binary_feature_name]
            if real_flag != wanted_flag:
                return False
        return True

    def is_diagnosis_ok(self, ecg_node):
        for diag_name in self.diagnosys_names.keys():
            wanted_flag = self.diagnosys_names[diag_name]
            real_flag = ecg_node["StructuredDiagnosisDoc"][diag_name]
            if real_flag != wanted_flag:
                return False
        return True

def get_num_ecgs_with_feature(json_data, binary_feature_name):
    """Смотрим, у скольки пациентов датасета в разделе
    докторсокого стуктурированного диагноза (т.е. в "StructuredDiagnosisDoc")
    находится True """
    counter = 0
    for case_id in data.keys():
        if(json_data[case_id]["StructuredDiagnosisDoc"][binary_feature_name] is True):
            counter += 1
    return counter

def print_diagnosis_distribution(json_data):
    ecgs_ids = json_data.keys()
    first_ecg_id = next(iter(ecgs_ids ))
    binary_features_list = list(json_data[first_ecg_id]["StructuredDiagnosisDoc"].keys())
    feature_num_trues = dict()
    for binary_feature_name in binary_features_list:
        num_trues = get_num_ecgs_with_feature(json_data, binary_feature_name)
        feature_num_trues[binary_feature_name] = num_trues

    for w in sorted(feature_num_trues, key=feature_num_trues.get, reverse=True):
        print(w, feature_num_trues[w])


def get_ecgs_by_query(json_data, query):
    ecgs_ids = []
    for case_id in json_data.keys():
        if query.is_query_ok(json_data[case_id]):
            ecgs_ids.append(case_id)
    return ecgs_ids




if __name__ == "__main__":
    PATH = "C:\\!mywork\\datasets\\BWR_ecg_200_delineation\\"
    FILENAME = "ecg_data_200.json"
    JSON_PATH = PATH + FILENAME

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        #print_diagnosis_distribution(data)
        diags = {"regular_normosystole":True,
                 "incomplete_right_bundle_branch_block":True}
        substrs = ["регулярный"]
        query = Query(diagnosys_names=diags,
                      substrs=substrs)
        ecgs = get_ecgs_by_query(data, query)
        message = str(query.diagnosys_names)
        draw_ecgs_to_htlm(data, ecgs, name_html="test.html", folder="results", message=message)
