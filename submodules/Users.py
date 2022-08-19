
class Users():
    ''' static attributes '''
    U = ['Jan', 'Mara']
    selected_id = 0

    def __init__(self):
        # preferable model
        #self.preference = ['pour', 'red']
        pass

    @property
    def selected(self):
        return self._users[self.selected_id]

    def __getattr__(self, attr):
        return self._users[attr]

    def demo__put_to_drawer__decide(self, s):
        if s.cup in s.drawer.contains:
            return True
        if s.r.attached and s.cup == s.r.attached:
            return ('put', 'drawer')
        if s.drawer.opened:
            return ('pick_up', 'cup')
        return ('open', 'drawer')

    def demo__put_to_drawer(self):
        action_sequence = [
            ['open', 'drawer'], # open drawer
            ['pick_up', 'cup1'],   # pick up cup1
            ['put', 'drawer']  # put to drawer
        ]
        return action_sequence




if __name__ == '__main__':
    u = Users()
    import sys; sys.path.append("..")
    from Scenes import Scene
    from Actions import Actions
    s = Scene(init='drawer,cup', random=False)

    while True:
        action = u.demo__put_to_drawer__decide(s)
        if action is True: break

        Actions.do(s, action, ignore_location=True)
    print("Task done!")
