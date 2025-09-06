from bs4 import BeautifulSoup
import bs4
import requests
import json
from typing import List

website="https://www.techwithtim.net"
result = requests.get(website)
docs= BeautifulSoup(result.text,"html.parser")


def find(tag):
    #finding attributes
    attributes=["href","src","alt","name","type","id"]
    new_attrs={}
    if tag.attrs:
        for k,v in tag.attrs.items():
            if k in attributes:
                new_attrs[k]=v
    
    content= tag.find(text=True,recursive=False)
    if content:
        content=content.strip()
    else:
        content=""
    
    dictionary={ #the new dictionary we wanna return
        "tag":tag.name, #returns type of tag
        "attributes":new_attrs,
        "text":content,
        "children":[]
    }

    for i in tag.children:
        if isinstance(i, bs4.element.Tag):
            dictionary["children"].append(find(i))
    
    return dictionary

    # print(bodys.attrs)
    # if bodys.attrs == {}:
    #     final[search] = {
    #         "children":[]
    #     }
    # elif "text" in bodys.attrs:
    #     final[search]["text"] = bodys.attrs["text"]
    # elif "href" in bodys.attrs:
    #     final[search]["attributes"]["href"]=bodys.attrs["href"]
    # elif "src" in bodys.attrs:
    #     final[search]["attributes"]["src"]=bodys.attrs["src"]
    # elif "type" in bodys.attrs:
    #     final[search]["attributes"]["type"]=bodys.attrs["type"]
    # elif "name" in bodys.attrs:
    #     final[search]["attributes"]["name"]=bodys.attrs["name"]
    # elif "alt" in bodys.attrs:
    #     final[search]["attributes"]["alt"]=bodys.attrs["alt"]

    # if len(bodys.contents)>0:
    #     print(len(bodys.contents))
    #     for i in bodys.contents:
    #         if len(i.contents) >0:
    #             find(i.contents)
    #         else:
    #             final["body"]["children"].append(i)


bodys= docs.find("body")
final=find(bodys)
with open("test.json","w") as f:
    f.write(json.dumps(final,indent=4))


# if bodys.attrs
# content = bodys.content
# for i in content:
#     final
#     if len(i.contents)>0:
#         #has childrent
# pass

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
