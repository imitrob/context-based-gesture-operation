
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






if __name__ == '__main__':
    pass
