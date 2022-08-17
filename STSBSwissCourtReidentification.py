import json
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
import datetime
import re 
import itertools
from pathlib import Path
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

DATA_DIR = Path('data')

"""
prepare_decisions function build a dataframe of decisions which are related to STSB events
64 decisions containing one of the following terms: SUST,STSB,SISI,SESE.
44 decisions in de.
17 decisions in fr.
3 decisions in it.
"""
def prepare_decisions():
    df = pd.read_csv(DATA_DIR / 'decisions.csv')
    interesting_cols = ['file_id','language', 'canton_name', 'court_name', 'chamber_string', 'date', 'pdf_url', 'html_url', 'text']
    df['text'] = df['pdf_raw'].astype(str) + df['html_raw'].astype(str)
    return df[interesting_cols]
    
decisions=prepare_decisions()
num_decisions = len(decisions.index)
print(f"Found {num_decisions} decisions containing at least one of the following terms SUST,STSB,SISI,SESE")

de_decisions= len(decisions[(decisions['language']=='de')])
print(f"Found {de_decisions} decisions in German languages")

fr_decisions= len(decisions[(decisions['language']=='fr')])
print(f"Found {fr_decisions} decisions in French languages")

it_decisions= len(decisions[(decisions['language']=='it')])
print(f"Found {it_decisions} decisions in Italian languages")

"""
stsb_events function build a dataframe of events related to aviation or trains and ships.
Our database of aviation events contains 1944 entries
Our database of trains and ships events contains 777 entries
"""
def stsb_events(file):
    stsb_file = DATA_DIR / file
    with open(stsb_file, 'r') as f:
         data = json.load(f)
    df = pd.DataFrame(data)
    return df

aviation =  stsb_events('aviatik.json')
trains_and_ships =  stsb_events('bahnen_und_schiffe.json')

print(f"Our database of aviation events contains {len(aviation.index)} entries") 
print(f"Our database of trains and ships events contains {len(trains_and_ships.index)} entries")


#Converts date format from 26. 05. 2016 to 26. Mai 2016 based on decision languages. we need this to search it in decision text. 
dic= {
    "de": {"January": "Januar",
           "February": "Februar",
           "March": "März",
           "April": "April",
           "May": "Mai",
           "June": "Juni",
           "July": "Juli",
           "August": "August",
           "September": "September",
           "October": "Oktober",
           "November": "November",
           "December": "Dezember",},
    "fr": {"January": "janvier",
           "February": "février",
           "March": "mars",
           "April": "avril",
           "May": "mai",
           "June": "juin",
           "July": "juillet",
           "August": "août",
           "September": "septembre",
           "October": "octobre",
           "November": "novembre",
           "December": "décembre",},
    "it": {"January": "gennaio",
           "February": "febbraio",
           "March": "marzo",
           "April": "aprile",
           "May": "maggio",
           "June": "giugno",
           "July": "luglio",
           "August": "agosto",
           "September": "settembre",
           "October": "ottobre",
           "November": "novembre",
           "December": "dicembre",},
}
def convert_date(e_date,lang):
    idx =pd.to_datetime(e_date,dayfirst=True)
    en_month = idx.strftime("%B")
    month = dic[lang][en_month]
    date1 = idx.strftime("%d")+" "+month+" "+idx.strftime("%Y")
    date2 = idx.strftime("%d")+". "+month+" "+idx.strftime("%Y")
    date = [str(date1),str(date2)]
    return date

"""
clean location and details information in stsb data and remove any words which not used as identifier.
for example: "HB-KLT ROBIN AIRCRAFT ROBIN DR 400/160FlugzeugBetriebsart: SchulungFlugregeln: VFR" I remove "SchulungFlugregeln" and "FlugzeugBetriebsart" and "VFR".
and use other uniqe words as direct identifier
"""
ineffective_words = ["schweizer","lange","FLUGZEUGBAU","flight","technische","aéronautique","Aérodrome","undefine","EUROCOPTER",
                "EcolightBetriebsart","HelikopterFlugregeln","PrivatFlugregeln"," VFR","HelikopterBetriebsart",
                "FlugzeugBetriebsart","Flugzeug","SchulungFlugregeln","KommerziellFlugregeln"," IFR","Motorsegel",
                "VerkehrsfliegereiFlugregeln","HeissluftBetriebsart","AIRPLANE","COMPANY","AVIATION","AIRCRAFT","III", 
                "CORPORATION","HELICOPTER","HÉLICOPTÈRES","BedarfsfliegereiFlugregeln","ArbeitsflugFlugregeln",
                "RettungseinsätzeFlugregeln","MilitärFlugregeln","Eigenbau","GMBH","Helikopter","flugregeln","flug"]

stopwords = stopwords.words('english')+ stopwords.words('dutch') + stopwords.words('german') +stopwords.words('italian')+stopwords.words('french')

def clean_data(data):
    d = re.sub("[.,():]","",data)
    big_regex = re.compile('|'.join(ineffective_words),re.IGNORECASE)
    d = big_regex.sub(" ", d)
    word_tokens = word_tokenize(d)
    filtered_words = [w for w in word_tokens if not w.lower() in stopwords and len(w)>2 and not w.isdigit()]
    return filtered_words

aviation['location_list'] = aviation.apply(
    lambda aviation: clean_data(aviation.location), axis=1)
aviation['details_list'] = aviation.apply(
    lambda aviation: clean_data(aviation.details), axis=1)
  
#Extraxt Time,pist and report numbers from content of events. I used these as indirect identifier. 
reg_query ={
    "time": r'[0-9]{2}:[0-9]{2}:?[0-9]*',
    "pist":  r'\sPiste\s[0-9]+',
    "report_number": r'Schlussbericht Nr.\s*[0-9]+|Rapport final n°\s[0-9]+',
}

def extraxt_content_identifier(content,reg):
    doc= ""
    for each in content:
        lang = each["lang"]
        if lang == "en" or lang == "no":
            pass
        else:
            doc = each["content"] + doc   
    match = re.findall(reg,doc)
    keywords=list(set(filter(None,match))) 
    if not keywords:
        return None
    else:
        return keywords
    
aviation['time_pattern'] = aviation.apply(
    lambda aviation: extraxt_content_identifier(aviation.content,reg_query["time"]), axis=1)

aviation['pist_pattern'] = aviation.apply(
    lambda aviation: extraxt_content_identifier(aviation.content,reg_query["pist"]), axis=1)

aviation['report_number_pattern'] = aviation.apply(
    lambda aviation: extraxt_content_identifier(aviation.content,reg_query["report_number"]), axis=1)

#Find list of identifiers in decision text
def find_match(search_list,text):
    pattern =re.compile(r'\b(?:%s)\b' % '|'.join(search_list),re.IGNORECASE)
    match = re.findall(pattern,text)
    if not match:
        return None
    else:
        return set(list(map(lambda x: x.lower(), match)))

"""
For each event we have list of identifiers like date,location, details of aircrafts type and event times,
pist and report number. so in each decision we search these identifier to finde coresponding event.
based on score of identifiers we can link more similar events to decisions.
"""
def get_identifiers(text,lang,file_id):
    find_list = []
    for row in aviation.iterrows():
        score = 0
        kw = []
        id = row[1]["id"]  
        
        date = row[1]["event_date"]
        date_key = convert_date(date,lang)
        loc = row[1]["location_list"]
        det = row[1]["details_list"]
        time = row[1]['time_pattern']
        pist = row[1]['pist_pattern']
        report_number = row[1]['report_number_pattern']
        
        matched_date =find_match(date_key,text)
        if  matched_date is not None:
            score = len(matched_date) * 3 + score
            kw.append(matched_date)
        
        matched_loc =find_match(loc,text)
        if  matched_loc is not None:
            score = len(matched_loc) * 1 + score
            kw.append(matched_loc)
        
        matched_details =find_match(det,text)
        if  matched_details is not None:
            score = len(matched_details) * 2  + score
            kw.append(matched_details)
            
        if  report_number is not None:
            matched_report_number =find_match(report_number,text)
            if  matched_report_number is not None:
                score = len(matched_report_number) * 3 + score
                kw.append(matched_report_number)
        else:
            pass
        
        if  time is not None:
            matched_time =find_match(time,text)
            if  matched_time is not None:
                score = len(matched_time) * 2 + score
                kw.append(matched_time)
        else:
            pass
            
        if  pist is not None:
            matched_pist =find_match(pist,text)
            if  matched_pist is not None:
                score = len(matched_pist) * 1 + score
                kw.append(matched_pist)
        else:
            pass
            
        find_list.append([id,score,kw])
    
    max_score = max(find_list, key=lambda x: x[1])
    if max_score[1] > 4:
        return max_score[0]
    else:
        return None

decisions['event_id'] = decisions.apply(
    lambda decision: get_identifiers(decision.text, decision.language,decision.file_id), axis=1)

#Delete text columns from decisions
decisions = decisions.drop(columns=["text"])


useless_words = ["Proprietario","Immatrikulation"," Besatzung","de l’accident","Luogo","Type d’aéronef ","Pays","Commandant de bord","Haupthalter","Prüfer","Hersteller","Eintragungsstaat","1.2.2.1.1","1.2.1.1.1","Allgemeines","Luftfahrzeug","Halter","Eigentümer","Pilot",
                "Passagiere","Ort","Datum und Zeit","Betriebsart","Fahrtphase","Startort","Zielort","Ziel","Koordinate",
                "Aéronef","Exploitant","Immatriculation","Indicatif d’appel","Constructeur","Propriétaire","Pilote","Lizenz",
                "Kommandant","Ausweis","Copilot","Examiner","Flugverkehrsleiter","Dienstbeginn","Coordonnées","Licence",
                "Lieu","Dommages aux personnes","Point de destination","Contrôleur de la circulation aérienne","Jours","muster","Fase di volo",
                "Luogo di destinazione","Luogo di decollo","Personne","Person","Date et heure",":"]

search_query = {
    "de":{"Aircraft":r"Luftfahrzeug.{1,100}Halter|Luftfahrzeug.{1,100}Eintragungsstaat|Luftfahrzeug.{1,100}Haupthalter",
         "Operator":r"Halter.{1,100}Eigentümer|Halter.{1,100}Hersteller|Haupthalter.{1,100}Haupteigentümer",
          "Owner":r"Eigentümer.{1,100}Halter|Haupteigentümer.{1,100}Prüfer|Eigentümer.{1,110}Pilot|Eigentümer.{1,100}Relevante|Eigentümer.{1,100}Besatzung\b",
          "Captain": r"Kommandant.{1,100}Lizenz",
          "Pilot": r"Pilot.{1,100}Ausweis",
          "Copilot": r"Pilot.{1,100}Ausweis|Pilot.{1,100}Passagiere",
          "Examiner": r"Prüfer.{1,100}Ausweis",
          "Staff of control services": r"Flugverkehrsleiter\s+[a-zA-Z,]*[0-9]{4}",
          "Aircraft_type": r"Betriebsart.{1,100}\s{2,50}Fahrtphase",
          "Point of departure": r"Startort.{1,100}\s{2,50}Ziel",
          "Point of landing": r"Ziel.{1,100}\s{2,50}",
        },
    "fr":{"Aircraft":r"Aéronef\s.{1,100}Exploitant|Immatriculation.{1,100}Indicatif d’appel|Immatriculation.{1,100}Lieu",
         "Operator":r"Exploitant.{1,110}Propriétaire|Exploitant.{1,100}Constructeur",
          "Owner":r"Propriétaire.{1,120}Pilote|Propriétaire.{1,100}Exploitant",
          "Captain": r"Commandant.{1,100}Licence",
          "Pilot": r"Pilote.{1,100}\s{2,50}Licence",
          "Copilot": r"Copilote.{1,100}\s{2,50}Licence",
          "Examiner": r"Prüfer.{1,100}\s{2,50}Ausweis",
          "Staff of control services": r"Contrôleur.{1,200}\s{2,50}Jours",
          "Aircraft_type": r"Type d’exploitation.{1,100}\s{2,50}Règles de vol",
          "Point of departure": r"Point de départ.{1,100}\s{2,50}Point de destination",
          "Point of landing": r"Point de destination\s.{1,100}\s{2,50}Dommages aux personnes",
        },
    "it":{"Aircraft":r"Aeromobile.{1,100}Esercente",
         "Operator":r"Esercente.{1,100}Proprietario",
          "Owner":r"Proprietario.{1,100}Pilota",
          "Captain": r"Kommandant.{1,100}Lizenz",
          "Pilot": r"Pilota.{1,100}\s{2,50}Licenza",
          "Copilot": r"Pilot.{1,100}\s{2,50}[Ausweis,Passagiere]$|Copilot\s.{1,100}\s{2,50}Lizenz",
          "Examiner": r"Prüfer\s.{1,100}\s{2,50}Ausweis",
          "Staff of control services": r"Contrôleur.{1,200}\s{2,50}Jours",
          "Aircraft_type": r"Tipo di volo.{1,100}Regole di volo",
          "Point of departure": r"Luogo di decollo.{1,100}\s{2,50}Luogo di destinazione",
          "Point of landing": r"Luogo di destinazione.{1,100}\s{2,50}Fase di volo",
        },
}

#retrieve more information in events report and assign to decisions
def re_identify(row):
    r = re.compile('|'.join(useless_words),re.IGNORECASE)
    id =int(row["event_id"])
    content= list(aviation.iloc[[id]]["content"])[0]
    for i in range(0,len(content)):
        lang =content[i]["lang"]
        if lang == "en":
            pass
        else:
            doc = content[i]["content"]
            row["Event date"] = list(aviation.iloc[[id]]["event_date"])
            row["Location"]= list(aviation.iloc[[id]]["location"])
            row["Event Url"] = list(aviation.iloc[[id]]["url"])
            row["Aircraft"]=r.sub("", str(re.findall(search_query[lang]["Aircraft"],doc)))  
            row["Operator"]=r.sub("", str(re.findall(search_query[lang]["Operator"],doc)))
            row["Owner"]=r.sub("", str(re.findall(search_query[lang]["Owner"],doc)))
            row["Captain"]= r.sub("", str(re.findall(search_query[lang]["Captain"],doc)))
            row["Pilot"]= r.sub("", str(re.findall(search_query[lang]["Pilot"],doc)))
            row["Copilot"]= r.sub("", str(re.findall(search_query[lang]["Copilot"],doc)))
            row["Examiner"]= r.sub("", str(re.findall(search_query[lang]["Examiner"],doc)))
            row["Staff of control services"]= r.sub("", str(re.findall(search_query[lang]["Staff of control services"],doc)))
            row["Aircraft_type"]= r.sub("", str(re.findall(search_query[lang]["Aircraft_type"],doc)))
            row["Point of departure"]= r.sub("", str(re.findall(search_query[lang]["Point of departure"],doc)))
            row["Point of landing"]= r.sub("", str(re.findall(search_query[lang]["Point of landing"],doc)))
    return row

non_linked = decisions[decisions.event_id.isna()]

linked = decisions[decisions.event_id.notna()]
re_identified = linked.apply(re_identify, axis=1)

#Save re_identified decisions in exels file
datatoexcel = pd.ExcelWriter('re-swisscourt.xlsx')
re_identified.to_excel(datatoexcel,header=True, index=True)
datatoexcel.save()