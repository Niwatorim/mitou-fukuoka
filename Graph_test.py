import matplotlib.pyplot as plt
import networkx as nx
from typing import List
import json
from bs4 import BeautifulSoup
import bs4
import requests

def getpath(data, value,original, prepath=()):
    if type(data) is dict:
        if "attributes" in data and type(data["attributes"]) is dict and "href" in data["attributes"]: # found key
                href=data["attributes"]["href"]
                original_page=prepath[0]
                yield [href,original_page,data]
        for k, v in data.items():
            path=prepath+(k,)#without the comma it comes out as a string in brackets and cant concatenate
            yield from getpath(v, value,original,path) # recursive call
    elif type(data) is list:
        for i in range(len(data)):
            val=data[i]
            path=prepath + (f"{i}",) #comma here makes it a tuple
            yield from getpath(val,value,original,path)

def shortestPath(G,edge_labels) -> List[str]:
    source= str(input("From node..?"))
    end = str(input("To node.....?"))
    shortest=nx.shortest_path(G,source,end)
    links = list(zip(shortest,shortest[1:]))
    print(links)
    steps=[]
    for i in links:
        step="use "+edge_labels[(i[0],i[1],0)]+" to go from "+i[0]+" to "+i[1] #edge labels[0] is a problem, cuz it only shows one way to get there not all
        steps.append(step)
    return steps

def getmeta(docs:str) -> List[dict]: #function to find all the meta data and find the values we are looking for
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

def find(tag): #Find the elements
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

if False: #for adding new pages, idk need get meta
    link="/"
    website=f"https://www.techwithtim.net{link}"
    result = requests.get(website)
    docs= BeautifulSoup(result.text,"html.parser")

    bodys= docs.find("body")
    final=find(bodys)
    with open("test.json","r") as f:
        content=json.load(f)
    with open("test.json","w") as f:
        content[link]=final
        f.write(json.dumps(content,indent=4))


with open("test.json","r") as file:

    #---------Fixing the nodes

    data=json.load(file) #gets all the data as dict
    #print(json.dumps(data,indent=4)) #prints all data
    #print(*getpath(data,"href",data)) #generator, cuz it does yield of all values then unpacks them
    nodes_data=list(getpath(data,"href",data))
    """
    Quick note here:
    i[2] must remain i[2] below, but its hurting my eyes so imma change it to the attribute and text only
    its meant to contain the entire dictionary which holds the href so the LLM knows what to click, but it overloads the networkx and overlaps it
    """
    for i in nodes_data:
        i[2]=str(i[2]["tag"]+"-"+i[2]["text"]) #replace all this yap with i[2] = str(i[2])
    nodes=[]
    for i in nodes_data:
        nodes.append((i[1],i[0],{"label":i[2]}))
    print(nodes)

    #--------Making the graph

    G = nx.MultiDiGraph() #makes graph
    G.add_edges_from(nodes) #adds edges
    pos=nx.circular_layout(G) #layout of graph
    nx.draw(G,pos,with_labels=True)
    edge_labels=nx.get_edge_attributes(G,"label")
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)

    #-------Finding shortest path
    
    print(shortestPath(G,edge_labels))
    
    plt.axis("off")
    plt.show()





    