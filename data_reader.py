import numpy as np
from json_reader import jsonReader

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
            obj["team1_heroes"].append(heroes[index+1]['localized_name'])
        elif team == -1:
            obj["team2_heroes"].append(heroes[index+1]['localized_name'])
    return obj

def datasetChecker(data):
    data = list(map(int, data))
    data[0] = data[0] if data[0] in teams else np.nan
    data[1] = data[1] if data[1] in regions else np.nan
    data[2] = data[2] if data[2] in mods else np.nan
    data[3] = data[3] if data[3] in lobbies else np.nan
    team1, team2 = [], []
    for index, team in enumerate(data[4:]):
        if team == 1:
            team1.append(index)
        elif team == -1:
            team2.append(index)
    data[4] = np.nan if len(team1) != 5 or len(team2) != 5 else data[4]
    return data

def getDataset(func = datasetChecker):
    trainData = np.genfromtxt('dataset/dota2Train.csv', delimiter=',')
    testData = np.genfromtxt('dataset/dota2Test.csv', delimiter=',')
    
    trainChecked = list(map(func, trainData))
    testChecked = list(map(func, testData))

    return np.asarray(trainChecked), np.asarray(testChecked)