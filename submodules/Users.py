
class Users():
    ''' static attributes '''
    U = ['Jan', 'Mara']

    def __init__(self, u=0):
        # preferable model
        #self.preference = ['pour', 'red']
        if isinstance(u, int):
            self.selected_id = u
        else: # u is string
            self.selected_id = Users.U.index(u)

    @property
    def name(self):
        return Users.U[self.selected_id]

    @property
    def selected(self):
        return self._users[self.selected_id]

    def __getattr__(self, attr):
        return self._users[attr]






if __name__ == '__main__':
    u = Users()
    u.name
