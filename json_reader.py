import json

def fileReader(fileName):
    f = open("json/" + fileName + ".json", "r")
    lines = f.readlines()
    f.close()
    return [line.strip() for line in lines]

def heroReader(fileName):
    heroesFile = fileReader(fileName)
    heroesJoined = "".join(heroesFile)
    heroesObject = json.loads(heroesJoined)
    heroesList = heroesObject[fileName]
    heroesDict = {}
    for hero in heroesList:
        heroesDict[hero['id']] = hero
    return heroesDict

heroes = heroReader("heroes")
lobbies = heroReader("lobbies")
mods = heroReader("mods")
regions = heroReader("regions")

print(mods)


