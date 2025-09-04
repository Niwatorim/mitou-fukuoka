import matplotlib.pyplot as plt
import networkx as nx
from typing import List
import json

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

with open("file_struct.json","r") as file:

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
    pos=nx.spring_layout(G)
    nx.draw(G,pos,with_labels=True)
    edge_labels=nx.get_edge_attributes(G,"label")
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)

    #-------Finding shortest path
    print(shortestPath(G,edge_labels))
    
    plt.axis("off")
    plt.show()





    