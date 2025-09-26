import matplotlib.pyplot as plt
import networkx as nx
from typing import List
import json
from bs4 import BeautifulSoup
import bs4
import asyncio
from crawl4ai import AsyncWebCrawler,CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
from browser_use import Agent, ChatGoogle

#-- for browser-use
async def browseruse(): #for browser use
        with open("instructions.txt","r") as f:
            task=str(f.readlines())
        agent = Agent(
            task=task,
            llm=ChatGoogle(model="gemini-2.5-flash"),
        )
        await agent.run()

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

async def main_scraping(site):
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            include_external=False,
            max_pages=3
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
    )

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
                if url == "":
                    url = "/"
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
                href= urljoin(original_page,href) 
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
        step="use "+edge_labels[(i[0],i[1],0)]+" to go from https://www.techwithtim.net"+i[0]+" to https://www.techwithtim.net"+i[1] #edge labels[0] is a problem, cuz it only shows one way to get there not all
        steps.append(step)
    return steps

#unused
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

user= int(input("1. Scrape \n2. Show graph and return instructions \n3. User browser use"))
if user == 1:
    user=str(input("Give site to scrape, else default to techwithtim"))
    if user== "":
        site="https://www.techwithtim.net"
    else:
        site=user
    asyncio.run(main_scraping(site))
elif user == 2:
    with open("temp.json","r") as file:

        #---------Fixing the nodes
        data=json.load(file) #gets all the data as dict
        nodes_data=list(getpath(data,"href",data))
        """
        Quick note here:
        i[2] must remain i[2] below, but its hurting my eyes so imma change it to the attribute and text only
        its meant to contain the entire dictionary which holds the href so the LLM knows what to click, but it overloads the networkx and overlaps it
        """

        for i in nodes_data:
            print("----- i -----")
            print(i)
            #i[2]=str("'"+i[2]["tag"]+"-"+i[2]["text"]+"' located in: '" +i[2]["locator"]+"' ") #replace all this yap with i[2] = str(i[2])
            i[2]=str("'"+i[2]["tag"]+"-"+i[2]["text"])
        nodes=[]
        for i in nodes_data:
            nodes.append((i[1],i[0],{"label":i[2]}))

        #--------Making the graph
        G = nx.MultiDiGraph() #makes graph
        G.add_edges_from(nodes) #adds edges
        pos=nx.circular_layout(G) #layout of graph
        nx.draw(G,pos,with_labels=True)
        edge_labels=nx.get_edge_attributes(G,"label")
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        #-------Finding shortest path
        
        user=str(input("Find shortest path?"))
        if user == "y":
            values=shortestPath(G,edge_labels)
            print(values)
            with open("instructions.txt","w") as file:
                for i in values:
                    file.write(i+"\n")  

        plt.axis("off")
        plt.show()
elif user == 3:
    asyncio.run(browseruse())