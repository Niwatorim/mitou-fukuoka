from my_ast import no_del_graph_creation
import os

path="./victim-site"

def path_finder(path):
    for i in os.listdir(path):
        new_path=os.path.join(path,i)
        if os.path.isdir(new_path): path_finder(new_path)
        if os.path.isfile(new_path): print(i)
        # no_del_graph_creation(i)
path_finder(path)