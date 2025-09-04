from bs4 import BeautifulSoup
import requests


website="https://www.techwithtim.net"

"""
import json
def getpath(data, value, prepath=()):
    if type(data) is dict:
        for k, v in data.items():
            path=prepath+(k,)#without the comma it comes out as a string in brackets and cant concatenate
            if k == value: # found key
                yield [v,path[0]]
            yield from getpath(v, value,path) # recursive call
    elif type(data) is list:
        for i in range(len(data)):
            val=data[i]
            path=prepath + ("f{i}",) #comma here makes it a tuple
            yield from getpath(val,value,path)


with open("file_struct.json","r") as file:
    data=json.load(file) #gets all the data as dict
    #print(json.dumps(data,indent=4)) #prints all data
    print(*getpath(data,"href")) #generator, cuz it does yield of all values then unpacks them
    nodes=list(getpath(data,"href"))
"""
