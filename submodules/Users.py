
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

    def demo__put_to_drawer(self):
        action_sequence = [
            ['open', 'drawer'], # open drawer
            ['pick_up', 'cup1'],   # pick up cup1
            ['put', 'drawer']  # put to drawer
        ]
        return action_sequence


if __name__ == '__main__':
    u = Users()
    u.Jan
    u.users
