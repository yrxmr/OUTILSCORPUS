import spacy

with open("/home/y/Documents/sequoia.txt","r") as f, open("/home/y/Documents/orfeo.txt","r") as f2, open("/home/y/Documents/bashung.txt") as f3:
    f = f.read()
    f2 = f2.read()
    f3 = f3.read()
    nlp = spacy.load("fr_core_news_md")
    doc = nlp(f)
    doc2 = nlp(f2)
    doc3= nlp(f3)

    for token in doc:
        print(token.text, token.pos_)

    for token in doc2:
        print(token.text, token.pos_)

    for token in doc3:
        print(token.text, token.pos_)
