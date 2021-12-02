import json
import random
import re
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from pandas import DataFrame

import plotly.express as px

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

DATA_DIR = Path('data')

search_queries = {
    "de": {"projectID": r"projekt.?id|simap.?nr\.?| -id",
           "noticeNumber": r"meldungsnummer|ref.?nr\.? simap.?nr\.?"},
    "fr": {"projectID": r"id du projet",
           "noticeNumber": r"no\.? de la publication|n° de la publication simap"},
    "it": {"projectID": r"id del progetto",
           "noticeNumber": r"n\.? della pubblicazione|n\.? di notificazione|pubblicazione simap n\.?"},
}

# TODO contractor auch rausholen bei re-identifizierung (beschwerdegegner)

"""
Why are there so many more german relevant decisions than french ones?
vielleicht entfernen Gerichtsschreiber die Meldungsnummer, Im Kanton Zürich bspw. werden Meldungsnummern des Amtblattes erwähnt
Anteil von kantonalen Urteilen in deutschsprachigen Entscheiden signifikant höher als in französischen (1/31) und italienischen (0/25)
TODO ==> bei hürlimann nachfragen ob bei entscheidsuche französische beschaffungsrechtliche Urteile nicht erfasst sind
==> Wir gehen davon aus, dass relevante Genfer oder Waadtländische Urteile nicht publiziert worden sind
"""

"""
Anzahl Urteile mit "simap" (ignorecase) im Text:
de: 557
fr: 414
it: 46
"""


def prepare_awards():
    awards_file = DATA_DIR / 'intelliprocure_all_awards.json'
    with open(awards_file, 'r') as f:
        data = json.load(f)[2]['data']
    df = pd.DataFrame(data)
    df.projectID = df.projectID.astype(int)
    df.noticeNumber = df.noticeNumber.astype(int)
    df.price = df.price.astype(float)  # cannot convert to int because of NA and inf
    # df.cpvNumber = df.cpvNumber.astype(int) # this one can also be a list
    df.datePublication = pd.to_datetime(df.datePublication)
    return df


awards = prepare_awards()
print(f"Our database of awards contains {len(awards.index)} entries")
print(f"The median awarded price is {int(awards.price.median())}")
print(f"There are {len(awards.bidder.unique())} unique bidders")
print(f"There are {len(awards.contractor.unique())} unique contractors")


def prepare_decisions(language: str):
    decisions_file = DATA_DIR / f'{language}_simap.csv'
    interesting_cols = ['language', 'canton', 'court', 'chamber', 'date', 'pdf_url', 'html_url', 'text']
    return pd.read_csv(decisions_file, usecols=interesting_cols)


lang_decisions = []
for language in search_queries.keys():
    lang_decisions.append(prepare_decisions(language))
decisions = pd.concat(lang_decisions)
num_decisions = len(decisions.index)
terms = []
for queries in search_queries.values():
    terms.append(queries['projectID'])
    terms.append(queries['noticeNumber'])
print(f"Found {num_decisions} decisions containing at least one of the following terms {terms}")


def get_identifiers(decision, language, query):
    regex = rf"({search_queries[language][query]}).*?(\d+)"
    return list(set([int(match[1]) for match in re.findall(regex, decision, re.IGNORECASE)]))


# Get the actual projectIDs and noticeNumbers from the text
decisions['projectIDs'] = decisions.apply(
    lambda decision: get_identifiers(decision.text, decision.language, "projectID"), axis=1)
decisions['noticeNumbers'] = decisions.apply(
    lambda decision: get_identifiers(decision.text, decision.language, "noticeNumber"), axis=1)

decisions = decisions.drop(columns=["text"])  # drop text so we can look at the df more easily


def clean_awarded_df(awarded):
    if awarded.empty:
        return None
    assert len(awarded.projectTitle.unique()) == 1  # there should only be one projectTitle
    return {
        "projectTitle": awarded.projectTitle.unique()[0],
        "bidders": awarded.bidder.tolist(),
        "contractors": awarded.contractor.tolist(),
        "prices": awarded.price.tolist()
    }
    # columns = ['bidder', 'price', 'projectTitle']
    # return awarded[columns].to_json()


def find_by_projectID(projectID: int):
    return clean_awarded_df(awards[awards.projectID == projectID])


def find_by_noticeNumber(noticeNumber: int):
    return clean_awarded_df(awards[awards.noticeNumber == noticeNumber])


decisions["found_projectID"] = decisions.projectIDs.str.len() > 0
decisions["found_noticeNumber"] = decisions.noticeNumbers.str.len() > 0

# retrieve the awards from the IntelliProcure export file and link it (just take the first found projectID or noticeNumber for simplicity)
decisions['awards_found_by_projectID'] = decisions.apply(
    lambda decision: find_by_projectID(decision.projectIDs[0]) if decision.projectIDs else None, axis=1)
decisions['awards_found_by_noticeNumber'] = decisions.apply(
    lambda decision: find_by_noticeNumber(decision.noticeNumbers[0]) if decision.noticeNumbers else None, axis=1)

# split into non_re_identified and re_identified
non_re_identified = decisions[decisions.awards_found_by_projectID.isna()
                              & decisions.awards_found_by_noticeNumber.isna()]
re_identified = decisions.dropna(subset=['awards_found_by_projectID', 'awards_found_by_noticeNumber'], how='all')

# just combine all of the found awards (do not care for duplicate resolution for simplicity)
re_identified['awards_found'] = re_identified.apply(
    lambda x: x.awards_found_by_noticeNumber or x.awards_found_by_projectID, axis=1)

# aggregate award prices
re_identified['mean_price'] = re_identified.apply(
    lambda x: np.mean(x.awards_found['prices']) if x.awards_found else None, axis=1)

# just take the first bidder/contractor for simplicity
re_identified['bidder'] = re_identified.apply(
    lambda x: x.awards_found['bidders'][0] if x.awards_found else None, axis=1)
re_identified['contractor'] = re_identified.apply(
    lambda x: x.awards_found['contractors'][0] if x.awards_found else None, axis=1)
re_identified['projectTitle'] = re_identified.apply(
    lambda x: x.awards_found['projectTitle'] if x.awards_found else None, axis=1)


def make_report(re_identified: DataFrame, decisions: DataFrame, language: str):
    print("\n\n")
    print("=" * 50)
    print(f"This is a report for the language {language}")
    print("=" * 50)
    if language != 'all':
        re_identified_lang = re_identified[re_identified.language == language]
        decisions_lang = decisions[decisions.language == language]
    else:
        re_identified_lang = re_identified
        decisions_lang = decisions

    terms_found_lang = decisions_lang[decisions_lang.found_noticeNumber | decisions_lang.found_projectID]

    num_decisions = len(decisions_lang.index)
    num_terms_found = len(terms_found_lang.index)
    num_re_identified = len(re_identified_lang.index)
    re_identification = {
        "re_identified":
            {
                "noticeNumber": re_identified_lang.awards_found_by_noticeNumber.count(),
                "projectID": re_identified_lang.awards_found_by_projectID.count(),
                "total": num_re_identified,
            },
        "terms_found":
            {
                "noticeNumber": (terms_found_lang.noticeNumbers.str.len() > 0).sum(),
                "projectID": (terms_found_lang.projectIDs.str.len() > 0).sum(),
                "total": num_terms_found,
            },
    }

    print(f"Found {num_decisions} containing the term 'simap' (ignorecase)")
    print(f"Could re-identify {num_re_identified} out of {num_terms_found} decisions where we found a term "
          f"==> {100 * num_re_identified / num_terms_found:.2f}% re-identification rate")

    pprint(re_identification)

    print(f"We found decisions from the following courts: {decisions_lang.court.value_counts().to_json()}")
    print(f"We found terms in decisions from the following courts: {terms_found_lang.court.value_counts().to_json()}")
    print(f"We re-identified decisions from the following courts: {re_identified_lang.court.value_counts().to_json()}")
    bvge_percentage_total = len(decisions_lang[decisions_lang.court.str.contains("CH_BVGE")].index) / num_decisions
    bvge_percentage_terms_found = len(
        terms_found_lang[terms_found_lang.court.str.contains("CH_BVGE")].index) / num_terms_found
    bvge_percentage_re_identified = len(
        re_identified_lang[re_identified_lang.court.str.contains("CH_BVGE")].index) / num_re_identified
    print(f"CH_BVGE makes up {bvge_percentage_total * 100:2.2f}% of total decisions, "
          f"{bvge_percentage_terms_found * 100:2.2f}% of terms-found decisions "
          f"and {bvge_percentage_re_identified * 100:2.2f}% of re-identified decisions")

    print(f"Find a randomly chosen re-identified sample below:")
    random_sample = re_identified_lang.iloc[random.randint(0, num_re_identified)]
    print(random_sample[['awards_found_by_projectID', 'awards_found_by_noticeNumber']])

    # draw violin plot for prices
    # häufig rahmenverträge, müssen nicht alle Leistungen bezogen werden
    prices = re_identified_lang[re_identified_lang.mean_price > 0]
    fig = px.violin(prices, y="mean_price", box=True,  # draw box plot inside the violin
                    points='all',  # can be 'outliers', or False
                    )
    fig.write_image(f'results/{language}_price_distribution.png')

    # sort by mean_price
    re_identified_lang = re_identified_lang.sort_values(by='mean_price', ascending=False)

    # save all re-identifications
    re_identified_lang.to_csv(f"results/{language}_re_identifications.csv")

    # find bidders associated with highest prices
    high_bidder_threshold = 50000000  # this applies to many bidders in German decisions and to a few in the other langs
    high_bidders = re_identified_lang[re_identified_lang.mean_price > high_bidder_threshold]
    high_bidders.to_csv(f"results/{language}_high_bidders.csv")

    # calculate aggregates for the bidders (sum/mean of price and number of re-identifications
    bidder_aggregate = re_identified_lang.groupby(by="bidder").agg({'bidder': 'count', 'mean_price': ['sum', 'mean']})
    bidder_aggregate = bidder_aggregate.fillna(0)
    bidder_aggregate.mean_price = bidder_aggregate.mean_price / 1000000
    bidder_aggregate.mean_price = bidder_aggregate.mean_price.round(3)
    bidder_aggregate = bidder_aggregate.sort_values(by=[('mean_price', 'sum'), ('bidder', 'count')],
                                                    ascending=[False, False])
    bidder_aggregate = bidder_aggregate.rename(
        columns={'count': '# Re-Identifications', 'sum': 'Sum (M CHF)', 'mean': 'Mean (M CHF)'}, level=1)
    bidder_aggregate.to_csv(f"results/{language}_bidder_aggregate.csv")


make_report(re_identified, decisions, 'all')
make_report(re_identified, decisions, 'de')
make_report(re_identified, decisions, 'fr')
make_report(re_identified, decisions, 'it')
