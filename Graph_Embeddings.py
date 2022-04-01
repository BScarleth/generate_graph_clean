import torch
import numpy as np


class GraphEmbeddings:
    def __init__(self, n, index, a = 4, m = 300, e = 20):
        self.n = n
        self.e = e
        self.index = index
        self.embedded_concepts = np.zeros((n, a, 1, m))
        self.embedded_relations = np.zeros((e, 1, m)) #2

    #for each object we do:
    #   multiply probabilities times de embeddings of the attributes (e.g colors)
    def shape_embedding(self, d_shape, corpus_embedding, p_shapes, indx):
        v = 0
        for s, p in zip(d_shape, p_shapes[indx]):
            v += torch.tensor(corpus_embedding[s], dtype= torch.float32) * p
        return [v.detach().numpy()]

    def color_embedding(self, d_color, corpus_embedding, p_colors, indx):
        v = 0
        for c, p in zip(d_color, p_colors[indx]):
            v += torch.tensor(corpus_embedding[c], dtype= torch.float32) * p#.to("cpu")
        return [v.detach().numpy()]

    def size_embedding(self, d_size, corpus_embedding, p_sizes, indx):
        v = 0
        for s, p in zip(d_size, p_sizes[indx]):
            v += torch.tensor(corpus_embedding[s], dtype= torch.float32) * p#.to("cpu")
        return [v.detach().numpy()]

    def material_embedding(self, d_material, corpus_embedding, p_material, indx):
        v = 0
        for m, p in zip(d_material, p_material[indx]):
            v += torch.tensor(corpus_embedding[m], dtype= torch.float32) * p#.to("cpu")
        return [v.detach().numpy()]

    def edge_embedding(self, d_relation, corpus_embedding, p_edges_0, p_edges_1):
        p_edges = torch.tensor(p_edges_0 + p_edges_1)

        for indx in range(self.e):
            v = 0
            for r, p, in zip(d_relation, p_edges[indx]):

                v += torch.tensor(corpus_embedding[r], dtype=torch.float32) * p#.to("cpu")
            self.embedded_relations[indx] = [v.detach().numpy()] #[] or np.stack(())


    def obj_embedding(self, D, corpus_embedding, p_shapes, p_colors, p_sizes, p_material ):
        for obj in range(self.n):
            s = self.shape_embedding(D["shape"], corpus_embedding, p_shapes, obj)
            c = self.color_embedding(D["color"], corpus_embedding,p_colors, obj )
            si = self.size_embedding(D["size"], corpus_embedding, p_sizes, obj)
            m = self.material_embedding(D["material"], corpus_embedding, p_material, obj)

            self.embedded_concepts[obj] = np.stack((s, c, si, m))




