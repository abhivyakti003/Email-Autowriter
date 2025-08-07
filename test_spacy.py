import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("My name is Abhivyakti. I study at Banasthali University. I can speak hindi, english and a little french.")

for ent in doc.ents:
    print(ent.text, "=>", ent.label_)
