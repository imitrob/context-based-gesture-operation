
from srcmodules.Scenes import Scene
from srcmodules.Actions import Actions

class CupToDrawer():
    ''' Static '''
    action_sequence = [
        ['open', 'drawer'], # open drawer
        ['pick_up', 'cup'],   # pick up cup1
        ['put', 'drawer']  # put to drawer
    ]
    def __init__(self):
        pass
    def decide(self, s):
        if s.cup in s.drawer.contains:
            return True
        if s.r.attached and s.cup == s.r.attached:
            return ('put', 'drawer')
        if s.drawer.opened:
            return ('pick_up', 'cup')
        return ('open', 'drawer')

    def get_start_target_scenes(self):
        s = Scene(init='cup,drawer',random=False)
        s2 = s.copy()
        Actions.do(s2, ('open', 'drawer'), ignore_location=True, out=False)
        Actions.do(s2, ('pick_up', 'cup'), ignore_location=True, out=False)
        Actions.do(s2, ('put', 'drawer'), ignore_location=True, out=False)
        return s, s2


if __name__ == '__main__':
    t = CupToDrawer()
    import sys; sys.path.append("..")
    from Scenes import Scene
    from Actions import Actions
    s = Scene(init='drawer,cup', random=False)

    while True:
        action = t.decide(s)
        if action is True: break
        Actions.do(s, action, ignore_location=True)
    print("Task done!")

    s,s2 = CupToDrawer().get_start_target_scenes()
    s.info
    s2.info
    s == s2
