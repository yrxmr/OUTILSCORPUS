import spacy, re

filepath = input("Le chemin du fichier à traiter ? ")
f = open( filepath ,"r")
file = f.read()

#création d'une liste qui sera dédiée aux tokens
tokens = []

#tokenisation grâce au module spacy    
nlp = spacy.load("en_core_web_sm")
doc = nlp(file)
for token in doc:
    #l'on ajoute chaque token dans la liste "tokens"
    tokens.append(token.text)


#l'on crée une nouvelle liste "types" en reprenant la liste "tokens" mais sans les doublons
types = list(dict.fromkeys(tokens))


#l'on définit V et N comme étant la longueur des listes de types et de tokens
V = len(types)
N = len(tokens)


#l'on calcule et imprime le résulat 
print("Le ratio type/token est:")
print(((V/N)*100),"%")
