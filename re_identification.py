import json
import random
import re
from pathlib import Path

import pandas as pd

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

DATA_DIR = Path('data')


def prepare_awards():
    awards_file = DATA_DIR / 'intelliprocure_all_awards.json'
    with open(awards_file, 'r') as f:
        data = json.load(f)[2]['data']
    df = pd.DataFrame(data)
    df.projectID = df.projectID.astype(int)
    df.noticeNumber = df.noticeNumber.astype(int)
    df.price = df.price.astype(float)
    # df.cpvNumber = df.cpvNumber.astype(int) #this one can also be a list
    df.datePublication = pd.to_datetime(df.datePublication)
    return df


awards = prepare_awards()
print(f"Our database of awards contains {len(awards.index)} entries")


def prepare_decisions():
    decisions_file = DATA_DIR / 'de_bvger.csv'
    return pd.read_csv(decisions_file)


decisions = prepare_decisions()
num_decisions = len(decisions.index)
print(f"Found {num_decisions} decisions containing either the term 'Projekt-ID' or 'Meldungsnummer'")


def get_identifiers(decision, regex):
    return list(set([int(match) for match in re.findall(regex, decision, re.IGNORECASE)]))


decisions['projectIDs'] = decisions.apply(
    lambda decision: get_identifiers(decision.text, 'Projekt-ID.*?(\d+)'), axis=1)
decisions['noticeNumbers'] = decisions.apply(
    lambda decision: get_identifiers(decision.text, 'Meldungsnummer.*?(\d+)'), axis=1)

columns = ['bidder', 'price', 'projectTitle']


def clean_awarded_df(awarded):
    if awarded.empty:
        return None
    return awarded[columns].to_json()


def find_by_projectID(projectID: int):
    return clean_awarded_df(awards[awards.projectID == projectID])


def find_by_noticeNumber(noticeNumber: int):
    return clean_awarded_df(awards[awards.noticeNumber == noticeNumber])


decisions['awards_found_by_projectID'] = decisions.apply(
    lambda decision: find_by_projectID(decision.projectIDs[0]) if decision.projectIDs else None, axis=1)
decisions['awards_found_by_noticeNumber'] = decisions.apply(
    lambda decision: find_by_noticeNumber(decision.noticeNumbers[0]) if decision.noticeNumbers else None, axis=1)

re_identified_decisions = decisions.dropna(subset=['awards_found_by_projectID', 'awards_found_by_noticeNumber'],
                                           how='all')
num_re_identified_decisions = len(re_identified_decisions.index)
print(f"Could re-identify {num_re_identified_decisions} decisions")
print(f"This corresponds to a re-identification rate of {100 * num_re_identified_decisions / num_decisions:.2f}%")
print(f"Find a randomly chosen re-identified sample below:")
random_sample = re_identified_decisions.iloc[random.randint(0, num_re_identified_decisions)]
print(random_sample[['awards_found_by_projectID', 'awards_found_by_noticeNumber']])
