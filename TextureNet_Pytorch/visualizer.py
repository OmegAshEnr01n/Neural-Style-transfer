from visdom import Visdom
import numpy as np

class VisdomImgUpload(object):
    def __init__(self, env_name = 'main'):
        self.viz = Visdom()
        self.env = env_name
    def add_img(self, img, window):
        self.viz.image(
            img,
            win = window,
            env = self.env,
            opts=dict(caption='img', store_history=True),
        )
class VisdomPlot(object):
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='iteration',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

class VisdomText(object):
    def __init__(self, env_name):
        self.viz = Visdom()
        self.env = env_name
    def add_txt(self, txt, windowname):
        self.viz.text(txt, win=windowname, append=False, env = self.env)
