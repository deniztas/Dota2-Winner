import json

def fileReader(fileName):
    f = open("json/" + fileName + ".json", "r")
    lines = f.readlines()
    f.close()
    return [line.strip() for line in lines]

def jsonReader(fileName):
    itemsJoined = "".join(fileReader(fileName))
    itemList = json.loads(itemsJoined)[fileName]
    itemDict = {}
    for item in itemList:
        itemDict[item['id']] = item
    return itemDict


