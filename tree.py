"""
Simple tree object
"""
class Node(object):
    def __init__(self, id, level=0, cost=0.0, fname="", children = []):
        self.id = id
        self.cost = cost
        self.children = children
        self.level = level
        self.fname = ""

    def insert(self, parentid, new_id, cost=0.0, fname="",):
        if self.id == parentid:
            self.children.append(Node(new_id, level=self.level+1, cost=cost, fname=fname, children=[]))
            return True
        for child in self.children:
            return child.insert(parentid, new_id, cost, fname)

    def find(self, id_to_search):
        if self.id == id_to_search:
            return self.level
        for child in self.children:
            return child.find(id_to_search)
        return False

    def print_tree(self):
        print((self.id, self.cost, self.level), end="  ")

        for child in self.children:
            child.print_tree()
        return True

    def __str__(self):
        return str(self.id)

class MultiStart():
    def __init__(self, k, max_iters):
        self.k = k # number of instances to keep
        self.max_iters = max_iters # maximum number of iterations - max depth of the tree
        self.tree = Node(0) # root node
        self.ids = [0] # list of ids
        self.costs = [1e10] # list of costs
        self.fnames = [''] # temp filenames

    """
    get top-k results by sorting on the cost
    """
    def get_topk(self):
        costs, ids, fnames = zip(*sorted(zip(self.costs, self.ids, self.fnames)))
        return costs[:self.k], ids[:self.k], fnames[:self.k]

    """
    add entry to the tree
    """
    def add(self, parentid, cost, fname):
        self.fnames.append(fname)
        self.costs.append(cost)
        newid = len(self.ids)
        self.ids.append(newid)
        self.tree.insert(parentid=parentid,new_id=newid,cost=cost, fname=fname)
