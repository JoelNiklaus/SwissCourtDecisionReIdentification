# SIMAP Re-Identification

This analysis makes use of awards scraped from the SIMAP database and Swiss Court Rulings obtained from the Association entscheidsuche.
Using the project ID and the project number we can match the two datasets for some decisions. 
In this way, we can reidentify certain anonymized companies from the court decisions.

The Paper draft is available on [Google Docs](https://docs.google.com/document/d/1S6G5be0qo6YofHxoo-zZBCIqUg7wFjtEE2kJrLHRaK0/edit#)

## Organization
- The input data should be put into the data folder.
- The output is saved into the results folder.
- The script for analyis is called re_identification.py

## Run
Run the analysis using 
```python
python re_identification.py
```
