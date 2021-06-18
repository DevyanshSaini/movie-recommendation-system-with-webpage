#importing our model

import recommendSystem as model


class recommendTextManager:
    def __init__(self, text):
        self.text = text



def pas(tex):
    text = model.recommendGET(tex)
    return text['title'].values  

