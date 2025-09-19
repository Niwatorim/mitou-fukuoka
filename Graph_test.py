import matplotlib.pyplot as plt
import networkx as nx
from typing import List
import json
from bs4 import BeautifulSoup
import bs4
import requests
from crawl4ai import AsyncWebCrawler,CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from bs4 import BeautifulSoup
import json

#--- for scraping
def path_generation(element):
    values=[]
    child=element
    for parent in child.parents: #for every parent object
        siblings=parent.find_all(child.name,recursive=False)
        if len(siblings) > 1: #if has siblings
            count = 1
            for sib in siblings:
                if sib is element:
                    values.append(f"{child.name}[{count}]")
                    break
                count+=1
        else:
            values.append(child.name)
        if parent.name == '[document]':
            break
        child=parent
    values.reverse()
    return "/" + "/".join(values)

def beautiful(data):
    soup = BeautifulSoup(data,"lxml")
    tags=["a", "button", "input", "select", "textarea", "form", "h1", "h2", "h3", "p", "img", "li"]
    interaction_map=[]
    for element in soup.find_all(tags):
        xpath=path_generation(element)

        attributes={}
        attributes_to_find=["id", "class", "name", "href", "src", "alt", "type", "value", "placeholder", "role", "aria-label"]
        for attribute in element.attrs:
            if attribute in attributes_to_find:
                attributes[attribute]=element.attrs[attribute]
        item={
            "tag": element.name,
            "text":element.get_text(strip=True),
            "locator":xpath,
            "attributes":attributes
        }
        interaction_map.append(item)
    return interaction_map

async def main_scraping():

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            include_external=False
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
    )

    site="https://www.techwithtim.net"
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(
            url=site,
            config=config
        )
        scraped=[]
        content={}
        for result in results:
            if result.url not in scraped:
                url=str(result.url).replace(site,"")
                content[url]=beautiful(result.html)
                scraped.append(result.url)
        print(content)
        with open("temp.json","w") as f:
            f.write(json.dumps(content,indent=2))

#-- for making graph
def getpath(data, value,original, prepath=()):#Find the path for hrefs
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

def shortestPath(G,edge_labels) -> List[str]: #Find shortest path
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

with open("temp.json","r") as file:

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
    
    #print(shortestPath(G,edge_labels))
    
    plt.axis("off")
    plt.show()





    