import os

def joinmakedir(a, b):
    newp = os.path.join(a, b)
    os.makedirs(newp, exist_ok = True)
