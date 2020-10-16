class Node(object):
    def __init__(self, id, cost=0.0, fname="", children = []):
        self.id = id
        self.cost = cost
        self.children = children
        self.fname = ""

    def insert(self, parentid, new_id, cost=0.0, fname="",):
        if self.id == parentid:
            #print(self.id, parentid)
            self.children.append(Node(new_id, cost=0.0, fname="", children=[]))
            return True

        for child in self.children:
            return child.insert(parentid,new_id)

    def find(self, id_to_search):
        if self.id == id_to_search:
            return True

        for child in self.children:
            return child.search(id_to_search)

    def print_tree(self):
        print((self.id,self.cost), end="  ")

        for child in self.children:
            child.print_tree()
        return True

    def __str__(self):
        return str(self.id)

class MultiStart():
    def __init__(self, k, max_iters):
        self.k = k
        self.max_iters = max_iters
        self.tree = Node(0)
        self.ids = []
        self.costs = []
        self.fnames = []

    def get_topk(self):
        costs, ids, fnames = zip(*sorted(zip(self.costs, self.ids, self.fnames)))
        return costs[:self.k], ids[:self.k], fnames[:self.k]

    def add(self, parentid, cost, fname):
        self.fnames.append(fname)
        self.costs.append(cost)
        self.ids.append(len(self.ids))
        self.tree.insert(parentid,len(self.ids), cost, fname)
