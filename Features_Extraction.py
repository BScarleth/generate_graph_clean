from torchvision.models.resnet import resnet101, resnet18
import torch.nn as nn
import torch.nn.functional as F
import time, torch
from datetime import datetime
from sklearn.metrics import classification_report
from utils import timeSince
import numpy as np

class Model(nn.Module):
    def __init__(self, in_features = 1000, num_material= 2, num_color= 8 , num_size = 2, height= 440, width = 520, pretrained= False, device="cuda"):
        super().__init__()
        self.net = resnet18(pretrained=True)
        self.predictorSize = PredictorSize(num_size, in_features)
        self.predictorMaterial = PredictorMaterial(num_material, in_features)
        self.predictorColor = PredictorColor(num_color, in_features)
        self.heigth = height
        self.width = width


        if pretrained:
            PATH_OUTPUT_MODEL = "/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/fex_model_20220328_103143"
            self.load_state_dict(torch.load(PATH_OUTPUT_MODEL, map_location=device))

    def forward(self, x):
        #new_x = x.to("cuda")
        #print(type(new_x))
        new_x = x.view(5, 3, self.heigth, self.width)
        new_x = self.net(new_x)
        predictor_size = self.predictorSize(new_x)
        predictor_material = self.predictorMaterial(new_x)
        predictor_color = self.predictorColor(new_x)

        return predictor_color, predictor_size, predictor_material,

class PredictorSize(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.W = nn.Linear(in_features, num_classes)
        self.X = nn.Linear(num_classes, num_classes)

    def forward(self, data):
        w = self.W(data)
        x = self.X(w)
        return x

class PredictorMaterial(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.W = nn.Linear(in_features, num_classes)
        self.X = nn.Linear(num_classes, num_classes)

    def forward(self, data):
        w = self.W(data)
        x = self.X(w)
        return x

class PredictorColor(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.W = nn.Linear(in_features, 500)
        self.X = nn.Linear(500, num_classes)
        self.Y = nn.Linear(num_classes, num_classes)

    def forward(self, data):
        w = self.W(data)
        x = self.X(w)
        y = self.Y(x)
        return y

def train_conv_net(model, data, n_iters, print_every=1, plot_every=1, device = "cuda"):
    PATH_OUTPUT_MODEL = "/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/"

    model.to(device)
    model.net.requires_grad = False
    plot_losses = []
    plot_acc = []

    print_loss_total = 0  # Reset every print_every
    print_acc_total = [0, 0, 0]  # Reset every print_every

    plot_loss_total = 0  # Reset every plot_every
    plot_acc_total = 0  # Reset every plot_every

    start = time.time()
    t = datetime.now().strftime("%Y%m%d_%H%M%S")

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0001)

    print("--------- training started ---------")
    for iter in range(1, n_iters + 1):
        idx_g = 0
        epoch_loss = 0
        epoch_acc = [0,0,0]

        for d in data:
            for i in range(len(d)):
                model.train()
                input_features = d["pixel_mask"][i]
                color_labels = d["colors"][i]
                size_labels = d["sizes"][i]
                material_labels = d["materials"][i]
                #with torch.no_grad():
                color_predictions, size_predictions, material_predictions = model(input_features)

                loss_color = F.cross_entropy(color_predictions, color_labels)
                loss_size = F.cross_entropy(size_predictions, size_labels)
                loss_material = F.cross_entropy(material_predictions, material_labels)

                loss = loss_color + loss_size + loss_material

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += float(loss)
                #epoch_acc += evaluate(model, val_data)
                idx_g += 1
        print_loss_total += (epoch_loss / idx_g)
        plot_loss_total += (epoch_loss / idx_g)

        #print_acc_total += (epoch_acc / idx_g)
        #plot_acc_total += (epoch_acc / idx_g)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0

            #print_acc_avg = print_acc_total / print_every
            print_acc_total = 0
            #print('%s (%d %d%%) %.4f -- C: %.4f S: %.4f  M: %.4f' % (
            #timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg, print_acc_avg[0], print_acc_avg[1], print_acc_avg[2]))

            print('%s (%d %d%%) %.4f' % (
                timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

            #torch.save(model.state_dict(), PATH_OUTPUT_MODEL + "fex_model_{}".format(t))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            plot_acc_avg = plot_acc_total / plot_every
            plot_acc.append(plot_acc_avg)
            plot_acc_total = 0
    return plot_losses, plot_acc

def evaluate(model, data):
  model.eval()
  acc=0
  acc_sizes = 0
  acc_material = 0

  len_acc = 5 * len(data)

  for d in data:
      for i in range(len(d)):
        input_features = d["pixel_mask"][i]
        color_labels = d["colors"][i]
        size_labels = d["sizes"][i]
        material_labels = d["materials"][i]
        color_predictions, size_predictions, material_predictions = model(input_features)

        _, indices = torch.max(color_predictions, dim=1)
        _, indices_sizes = torch.max(size_predictions, dim=1)
        _, indices_materials = torch.max(material_predictions, dim=1)

        correct = torch.sum(indices == color_labels)
        correct_sizes = torch.sum(indices_sizes == size_labels)
        correct_materials = torch.sum(indices_materials == material_labels)

        acc += (correct.item() * 1.0 / len(color_predictions))
        acc_sizes += (correct_sizes.item() * 1.0 / len(size_predictions))
        acc_material += (correct_materials.item() * 1.0 / len(material_predictions))
  return np.array([acc/len_acc, acc_sizes/len_acc, acc_material/len_acc])

def covnet_predict(model, data_item, device="cuda"):

  model.to(device)
  model.eval()

  input_features = data_item
  color_predictions, size_predictions, material_predictions = model(input_features)

  material = {
      0: "metal",
      1: "rubber",
  }

  size = {
      0: "small",
      1: "large",
  }

  color = {
      0: "gray",
      1: "red",
      2: "blue",
      3: "green",
      4: "brown",
      5: "purple",
      6: "cyan",
      7: "yellow",
  }

  colors = []
  for c in list(torch.max(color_predictions, axis=1).indices):
    colors.append(color[c.item()])

  sizes = []
  for s in list(torch.max(size_predictions, axis=1).indices):
      sizes.append(size[s.item()])

  materials = []
  for m in list(torch.max(material_predictions, axis=1).indices):
      materials.append(material[m.item()])

  return colors, sizes, materials, color_predictions, size_predictions, material_predictions

def evaluate_model(model, data, device="cuda"):
    model.to(device)
    model.eval()

    clr_labels = ["0", "1", "2", "3", "4", "5", "6", "7"]
    sz_labels = ["0", "1"]
    mtl_labels = ["0", "1"]

    total_pred_c = []
    total_pred_s = []
    total_pred_m = []

    total_labels_c = []
    total_labels_s = []
    total_labels_m = []

    #for d in data:
    for d in data:
        for i in range(len(d)):
            input_features = d["pixel_mask"][i]
            color_labels = d["colors"][i]
            size_labels = d["sizes"][i]
            material_labels = d["materials"][i]
            color_predictions, size_predictions, material_predictions = model(input_features)

            total_labels_c.extend(color_labels.tolist())
            total_labels_s.extend(size_labels.tolist())
            total_labels_m.extend(material_labels.tolist())

            total_pred_c.extend(torch.max(color_predictions, axis=1).indices.tolist())
            total_pred_s.extend(torch.max(size_predictions, axis=1).indices.tolist())
            total_pred_m.extend(torch.max(material_predictions, axis=1).indices.tolist())
    print(classification_report(total_labels_c, total_pred_c, target_names=clr_labels))
    print(classification_report(total_labels_s, total_labels_s, target_names=sz_labels))
    print(classification_report(total_labels_m, total_pred_m, target_names=mtl_labels))