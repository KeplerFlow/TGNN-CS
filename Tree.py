from MLP_models import MLP, MLPNonleaf


class TreeNode:
    def __init__(self, timerange, layer_id):
        self.time_start = timerange[0]
        self.time_end = timerange[1]
        self.children = []
        self.parent = None
        self.model = None
        self.layer_id = layer_id
        self.vertex_core_number = None
        self.vertex_degree = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def set_model(self, model):
        self.model = model

