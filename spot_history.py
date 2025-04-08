import re
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score


def spot_history_regex():
    """Use regex to spot history"""

    # params
    target_history = ["history", "past"]
    target_disease = ["MI"]

    # load data
    df = pd.read_csv("data/history_myocardial_infarction/deepseek.csv")

    # scan data
    y_pred = []
    y = list(df['LABEL'])
    for index, row in df.iterrows():

        # run detection
        text = row['TEXT']
        label = row['LABEL']
        match = False

        # check history targets
        match_history = False
        for target in target_history:
            if re.search(target, text):
                match_history = True
                
        # check disease targets
        match_disease = False
        for target in target_disease:
            if re.search(target, text):
                match_disease = True

        if match_disease and match_history:
            match = True

        # update y pred
        if match:
            y_pred.append(1)
        else:
            y_pred.append(0)

    # compute acc
    accuracy = accuracy_score(y, y_pred)

    # compute rappel
    recall = recall_score(y, y_pred)

    # compute f1 score
    f1 = f1_score(y, y_pred)

    # display results
    print(f"[REGEX][ACC] : {accuracy}")
    print(f"[REGEX][RECALL] : {recall}")
    print(f"[REGEX][F1-SCORE] : {f1}")

    
if __name__ == "__main__":

    spot_history_regex()

    
