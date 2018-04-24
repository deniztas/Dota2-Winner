import json
from IPython import embed

def fileReader(fileName):
    f = open("json/" + fileName + ".json", "r")
    lines = f.readlines()
    f.close()
    return [line.strip() for line in lines]

def heroReader():
    heroesFile = fileReader("heroes")
    heroesJoined = "".join(heroesFile)
    heroesObject = json.loads(heroesJoined)
    heroesList = heroesObject['heroes']
    heroesDict = {}
    for hero in heroesList:
        heroesDict[hero['id']] = hero
    return heroesDict

h = heroReader()
embed()