#coding=utf-8
import json


def inWeb():
    try:
        REQUEST
        return True
    except NameError:
        return False

def q(key):
    req = json.loads(REQUEST)
    return req['args'][key][0]
def inWeb():
    try:
        REQUEST
        return True
    except NameError:
        return False
def q(key):
    req = json.loads(REQUEST)
    return req['args'][key][0]

