import re
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score


def spot_thx_regex():
    """Use regex to spot thx"""

    # params
    targets = ["thrombosis", "DVT", "thrombolysis", "thrombo", "VTE", "thrombus"]

    # load data
    df = pd.read_csv("data/thx/deepseek.csv")

    # scan data
    y_pred = []
    y = list(df['LABEL'])
    for index, row in df.iterrows():

        # run detection
        text = row['TEXT']
        label = row['LABEL']
        match = False
        for target in targets:
            if re.search(target, text):
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

    spot_thx_regex()

    
