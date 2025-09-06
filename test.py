from bs4 import BeautifulSoup
import requests
import json
from typing import List

website="https://practice.expandtesting.com/"
result = requests.get(website)
docs= BeautifulSoup(result.text,"html.parser")

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

# need to turn beautifulsoup data into dict first so get use this function
# get the elements of the page
def get_elements(data, value, original, prepath=()):
    if type(data) is dict:
        if "attributes" in data and type(data["attributes"]) is dict and "href" in data["attributes"]: # found key
                href=data["attributes"]["href"]
                original_page=prepath[0]
                yield [href,original_page,data]
        for k, v in data.items():
            path=prepath+(k,)#without the comma it comes out as a string in brackets and cant concatenate
            yield from get_elements(v, value,original,path) # recursive call
    elif type(data) is list:
        for i in range(len(data)):
            val=data[i]
            path=prepath + (f"{i}",) #comma here makes it a tuple
            yield from get_elements(val,value,original,path)

# using the bautifulsoup way to search links
def get_path(docs):
    link_buttons = docs.find_all('a', href=True)
    final = []
    buttons_dict = {}
    for button in link_buttons:
        button_name = button.text
        button_href = button['href']
        buttons_dict[button_name] = button_href
    final.append(buttons_dict)
    return final

# using beautifulsoup way to get elements
def get_elements_bs(docs):
    elements = []
    bodys = docs.find("body")
    children = bodys.findChildren()
    for child in children:
        tag_name = child.name
        attributes = child.attrs
        elements.append({ # too many junks
            'tag': tag_name,
            'attributes': attributes, 
        })
    return elements

# with open("buttons.json","w") as f:
#     allbuttons=get_path(docs)
#     json.dump(allbuttons,f,indent=4)

# bodys = docs.find("body")
# children = bodys.findChildren()
# for child in children:
#     print(child.attrs)
    
with open("elements.json","w") as f:
    allelements=get_elements_bs(docs)
    json.dump(allelements,f,indent=4)

'''
get the buttons info or picture or text 
get the button name
for button in buttons:
find the href
store in dict[button]=href

what if there's multiple steps? recursive?
form located at: final web page --> need to requests.get if located in another webpage
'''

# bodys= docs.find("body")
# print(bodys)
# children = bodys.findChildren()
# for child in children:
#     print(child)

# all_a_tags=bodys.find_all('a', href=True)
# for link in all_a_tags:
#     href=link.get('href')
#     # print(link)
#     print(href)

# content=list(bodys.children)
# for i in content:
    # if len(i.contents) > 0:
    # print("ok")
    # data=i.contents
    # print(data)
    # print("\n\n")
    # else: print("no")

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
