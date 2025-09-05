from bs4 import BeautifulSoup
import requests
import json
from typing import List

website="https://www.techwithtim.net"
result = requests.get(website)
docs= BeautifulSoup(result.text,"html.parser")

bodys= docs.find("body")
content=list(bodys.children)
for i in content:
    if len(i.contents) > 0:
        print("ok")
        print(i.contents)
        print("\n\n")
    else: print("no")



def getmeta(docs:str) -> List[dict]:
    meta=docs.find_all("meta")
    useful=[]
    for i in meta:
        metadata={}
        for j,v in i.attrs.items():
            if j == "name" or j == "content" or j == "title":
                metadata[j]=v
        if metadata != {}:
            useful.append(metadata)
    return useful        


# with open("test.json","w") as f:
#     allvalues=getmeta(docs)
#     json.dump(allvalues,f,indent=4)



"""
start with docs and make it a feature, and inside it check for head and also main/body
in main, find title and stuff. in 

inside docs, if there is a child: make this parent a key, and make the child a value. if the child has children, recall the function


def get metadata (one time)
doc make dict, has meta data and elements
in meta data, make dict, for each child value, write its own k,v pair


def get elements(recursive)
in elements, make list
need prepath? unknown
for i in (children):
    check the child, see all attributes, make dict in the following method:
            tag name
            attributes: href and other parts, like source of image, alt etc. forget the sizing and stuff <- needed: href, src, alt name, id, name, type of form or type
            text attributes etc.
            children[]
    if children have children, recall function on that child

"""
