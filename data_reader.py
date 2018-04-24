import numpy as np
from json_reader import jsonReader
from IPython import embed

teams = {1: "Team 1", -1: "Team 2"}
heroes = jsonReader('heroes')
lobbies = jsonReader('lobbies')
mods = jsonReader('mods')
regions = jsonReader('regions')

def datasetFormatter(data):
    data = list(map(int, data))
    obj = {}
    obj["winner"] = teams[data[0]] if data[0] in teams else "Unknown"
    obj["region"] = regions[data[1]]['name'] if data[1] in regions else "Unknown"
    obj["mode"] = mods[data[2]]['name'] if data[2] in mods else "Unknown"
    obj["type"] = lobbies[data[3]]['name'] if data[3] in lobbies else "Unknown"
    obj["team1_heroes"] = []
    obj["team2_heroes"] = []
    for index, team in enumerate(data[4:]):
        if team == 1:
            obj["team2_heroes"].append(index)
        elif team == -1:
            obj["team1_heroes"].append(index)
    return obj

trainData = np.genfromtxt('dataset/dota2Train.csv', delimiter=',')
trainFormatted = list(map(datasetFormatter, trainData))
testData = np.genfromtxt('dataset/dota2Test.csv', delimiter=',')
testFormatted = list(map(datasetFormatter, trainData))

embed()