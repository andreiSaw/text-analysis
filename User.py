import json


class User(object):
    def __init__(self):
        self.age="-1"
        self.id=-1
        self.texts=[]
    def from_dict(self, dicter):
        self.age=dicter['age']
        self.id=dicter['id']
        self.texts=dicter['texts']
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True)
