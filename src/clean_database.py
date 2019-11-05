'''
Remove all the enrolled faces in the database
'''

import os
import json
import sys
import shutil


path = "../database/"
raw_path = "../database/raw_images/"
cropped_path = "../database/cropped_faces/"
test_path = "../database/test_data/"
embedding_path = "../database/embeddings/"

print("!!Initializing Database Cleanup!!")
print("")

print("Removing data from: {}".format(path))
print("")
for the_file in os.listdir(path):
    file_path = os.path.join(path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(e)
print("Data Removed from: {}".format(path))
print("\n!!! Database Cleaned !!!")
print("")

print("Initializing Fresh Database")

if not os.path.exists(raw_path):
    os.makedirs(raw_path)
if not os.path.exists(cropped_path):
    os.makedirs(cropped_path)
if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

emb_file = os.path.join(embedding_path, "face_embeddings.json")
with open(emb_file, 'w+') as wf:
    base_emb = {}
    json.dump(base_emb, wf)

print("Fresh Database Initialized")
