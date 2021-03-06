import spacy
from spacy.util import minibatch, compounding
from pathlib import Path
import random


with open("articlejv.txt") as f1, open("articlejv2.txt") as f2:
    file1 = f1.read()
    file2 = f2.read()
    TRAIN_DATA=[(file1, {"entities": [(200, 206, "ORG"),(252,262,"MISC"),(307,313,"LOC"),(599,605,"ORG"),(838,851,"PER"),(871,884,"ORG"),
                                      (1067,1074,"ORG"),(1077,1083,"ORG"),(1085,1092,"ORG"),(1098,1104,"ORG"),(1260,1273,"PER"),(1397,1403,"MISC"),
                                      (1415,1417,"MISC"),(1477,1483,"ORG"),(1499,1505,"ORG"),(1611,1622,"MISC"),(1624,1630,"MISC"),
                                      (1700,1706,"MISC"),(1710,1719,"ORG"),(1732,1736,"ORG")]})]

    
    nlp = spacy.load('fr_core_news_sm')
    print("Original model :")
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    ner = nlp.get_pipe("ner")
    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(100):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    print("Testing model : ")
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    # save model to output directory
    output_dir = Path("./")
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

    # test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    for text, _ in TRAIN_DATA:
        doc = nlp2(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])


doc = nlp2(file2)
for ent in doc.ents:
    print(ent, ent.label_)

