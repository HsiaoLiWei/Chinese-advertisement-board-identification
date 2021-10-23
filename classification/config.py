import json, os
cwd = os.path.split(__file__)[0]

char_to_index = {}
with open(os.path.join(cwd, "configs",'char_to_index.json'),'r',encoding='utf-8') as f:
    char_to_index = json.load(f)

char_to_index_Only = {}
with open(os.path.join(cwd, "configs",'char_to_index_Only.json'),'r',encoding='utf-8') as f:
    char_to_index_Only = json.load(f)

char_to_index_New = {}
with open(os.path.join(cwd, "configs",'char_to_index_New.json'),'r',encoding='utf-8') as f:
    char_to_index_New = json.load(f)

index_to_char = {}
with open(os.path.join(cwd, "configs",'index_to_char.json'),'r',encoding='utf-8') as f:
    index_to_char = json.load(f)

index_to_char_Only = {}
with open(os.path.join(cwd, "configs",'index_to_char_Only.json'),'r',encoding='utf-8') as f:
    index_to_char_Only = json.load(f)

index_to_char_New = {}
with open(os.path.join(cwd, "configs",'index_to_char_New.json'),'r',encoding='utf-8') as f:
    index_to_char_New = json.load(f)
