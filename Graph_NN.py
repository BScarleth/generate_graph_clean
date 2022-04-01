import dgl
from dgl.nn import SAGEConv
from dgl.nn import GraphConv
from dgl.nn import RelGraphConv, GraphConv
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from datetime import datetime
import time, torch
from datetime import datetime
from sklearn.metrics import classification_report
from utils import timeSince
from sklearn.metrics import classification_report
from torchvision.models.resnet import resnet101
import numpy as np


class ScorePredictor(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.W = nn.Linear(in_features, num_classes)
        self.X = nn.Linear(in_features, num_classes)

    def forward(self, graph, features):
        with graph.local_scope():
          graph.ndata['h'] = features

          graph.apply_edges(lambda edges: {'score' : edges.src['h'] - edges.dst['h']})
          data = graph.edata['score']
          w = self.W(data)
          x = self.X(data)
          return w, x

class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = GraphConv(in_features, hidden_features, norm='both')
        self.conv2 = GraphConv(hidden_features, out_features, norm='both')

    def forward(self, graph, x):
        x = F.relu(self.conv1(graph, x))
        x = F.relu(self.conv2(graph, x))
        return x

class GraphModel(nn.Module):
    def __init__(self, in_features = 1000, hidden_features = 10 , out_features = 1000, num_classes = 2, pretrained= False, height= 440, width = 520, timestamp = None, device="cuda"):
        super().__init__()
        self.heigth = height
        self.width = width
        self.net = resnet18(pretrained=True)
        #self.net = resnet101(pretrained=True)
        self.gcn = StochasticTwoLayerGCN(in_features, hidden_features, out_features)
        self.predictor = ScorePredictor(num_classes, out_features)

        if pretrained:
            PATH_OUTPUT_MODEL = "/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/"+timestamp
            self.load_state_dict(torch.load(PATH_OUTPUT_MODEL, map_location=device))

    def forward(self, graph, x):
        new_x = x.view(5, 3, self.heigth, self.width)
        x = self.net(new_x)
        x = self.gcn(graph, x)
        return self.predictor(graph, x)

def train_graph_model(model, data, n_iters, print_every=1, plot_every=1):
    model.to("cuda")
    model.net.requires_grad = False

    PATH_OUTPUT_MODEL = "/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/"
    plot_losses = []
    plot_acc = []

    print_loss_total = 0  # Reset every print_every
    print_acc_total = [0, 0]  # Reset every print_every

    plot_loss_total = 0  # Reset every plot_every
    plot_acc_total = 0  # Reset every plot_every

    start = time.time()
    t = datetime.now().strftime("%Y%m%d_%H%M%S")

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    for iter in range(1, n_iters + 1):
        idx_g = 0

        epoch_loss = 0
        epoch_acc = [0,0]

        for dl in data:
            model.train()

            input_features = dl.ndata['feature']
            edge_labels = dl.edata['label']  # labels

            labels0 =   torch.tensor([x[0] for x in edge_labels]).to("cuda")
            labels1 = torch.tensor([ 0 if x[1] == 2 else 1 for x in edge_labels]).to("cuda")

            edge_predictions1, edge_predictions2 = model(dl, input_features)

            loss1 = F.cross_entropy(edge_predictions1, labels0)
            loss2 = F.cross_entropy(edge_predictions2, labels1)

            loss = loss1 + loss2
            opt.zero_grad()
            loss.backward()
            opt.step()

            acc = evaluate_training(model, data[:10])
            epoch_loss += loss
            epoch_acc += acc
            idx_g += 1

        print_loss_total += (epoch_loss / idx_g)
        plot_loss_total += (epoch_loss / idx_g)

        print_acc_total += (epoch_acc / idx_g)
        plot_acc_total += (epoch_acc / idx_g)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0

            print_acc_avg = print_acc_total / print_every
            print_acc_total = 0
            print('%s (%d %d%%) %.4f --  %.4f %.4f' % (
            timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg, print_acc_avg[0],  print_acc_avg[1]))

            torch.save(model.state_dict(), PATH_OUTPUT_MODEL + "gnn_model_{}".format(t))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            plot_acc_avg = plot_acc_total / plot_every
            plot_acc.append(plot_acc_avg)
            plot_acc_total = 0
    return plot_losses, plot_acc

def evaluate_training(model, graphs):
  model.eval()
  acc0 =0
  acc1 = 0

  for dl in graphs:
    input_features = dl.ndata['feature']
    edge_labels = dl.edata['label']
    labels0 = torch.tensor([x[0] for x in edge_labels]).to("cuda")
    labels1 = torch.tensor([ 0 if x[1] == 2 else 1 for x in edge_labels]).to("cuda")

    edge_predictions0, edge_predictions1 = model(dl, input_features)
    _, indices0 = torch.max(edge_predictions0, dim=1)
    _, indices1 = torch.max(edge_predictions1, dim=1)

    correct0 = torch.sum(indices0 == labels0)
    correct1 = torch.sum(indices1 == labels1)
    acc0 += (correct0.item() * 1.0 / len(labels0))
    acc1 += (correct1.item() * 1.0 / len(labels1))

  return np.array([acc0/len(graphs), acc1/len(graphs)])

def evaluate_graph_model(data, model):
    model.to("cuda")
    model.eval()
    cl_labels0 = ["0", "1"]
    cl_labels1 = ["2", "3"]
    total_labels0 = []
    total_labels1 = []
    total_pred0 = []
    total_pred1 = []

    for dl in data:
        input_features = dl.ndata['feature']
        edge_labels = dl.edata['label']
        labels0 = torch.tensor([x[0] for x in edge_labels])
        labels1 = torch.tensor([ 0 if x[1] == 2 else 1 for x in edge_labels])
        edge_predictions0, edge_predictions1 = model(dl, input_features)

        total_labels0.extend(labels0)
        total_labels1.extend(labels1)
        total_pred0.extend(torch.max(edge_predictions0, axis=1).indices.tolist())
        total_pred1.extend(torch.max(edge_predictions1, axis=1).indices.tolist())
    print(classification_report(total_labels0, total_pred0, target_names=cl_labels0))
    print(classification_report(total_labels1, total_pred1, target_names=cl_labels1))

def graphnn_predict(model, data_item, device="cuda"):

  model.to(device)
  model.eval()

  input_features = data_item.ndata['feature']

  edge_predictions0, edge_predictions1 = model(data_item, input_features)
  """
  "left": 0,
    "right": 1,
    "behind": 2,
    "front": 3,
  """

  relation0 = {
      0: "left",
      1: "right"
  }

  relation1 = {
      0: "behind",
      1: "front",
  }

  relations0 = []
  relations1 = []

  for m in list(torch.max(edge_predictions0, axis=1).indices):
      relations0.append(relation0[m.item()])

  for m in list(torch.max(edge_predictions1, axis=1).indices):
      relations1.append(relation1[m.item()])

  return relations0, edge_predictions0, relations1, edge_predictions1