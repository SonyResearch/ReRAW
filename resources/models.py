import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models

class ReRAW(nn.Module):
    def __init__(
        self,
        in_size=3,
        out_size=4,
        target_size=(32, 32),
        hidden_size=128,
        n_layers=8,
        gammas=[],
    ):

        super().__init__()

        self.gammas = gammas

        self.hidden_size = hidden_size
        self.target_size = target_size

        self.color_reconstruction = ColorReconstructionModule(
            in_size=in_size,
            hidden_size=hidden_size,
            out_size=hidden_size,
            n_layers=n_layers,
        )

        self.global_context_encoder = GlobalContextModule(hidden_size)
        self.gamma_scaling_encoder = GlobalContextModule(len(self.gammas))
        self.softmax = nn.Softmax(dim=1)

        self.heads = nn.ModuleList()
        for _ in range(len(self.gammas)):
            self.heads.append(
                Head(
                    in_size=hidden_size,
                    hidden_size=hidden_size,
                    out_size=out_size,
                    n_layers=n_layers,
                )
            )

    def forward(self, x, global_img):
        b, c, h, w = global_img.size()

        global_features = self.global_context_encoder(global_img)
        global_features = global_features.view(b, self.hidden_size, 1, 1)
        global_features = global_features.repeat(1, 1, self.target_size[0], self.target_size[1])

        x = self.color_reconstruction(x)
        x = x * global_features

        scaling = self.gamma_scaling_encoder(global_img)
        scaling = self.softmax(scaling)

        y = 0
        outputs = []

        for head_no, head in enumerate(self.heads):
            scale = scaling[:, head_no : head_no + 1].view(b, 1, 1, 1).repeat(1, 4, 1, 1)
            y0 = torch.clip(head(x), 0, None)
            y += torch.pow(y0, 1 / self.gammas[head_no]) * scale
            outputs.append(y0)
        
        outputs = torch.cat(outputs, dim=1)

        return y, outputs, scaling


class ColorReconstructionModule(nn.Module):
    def __init__(
        self,
        in_size=3,
        initial_size=32,
        hidden_size=128,
        out_size=128,
        n_layers=4,
        bias=True,
        act=nn.LeakyReLU,
        val=0.1,
    ):

        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_size,
                    initial_size * in_size,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    bias=bias,
                    padding_mode="reflect",
                    groups=3,
                ),
                act(val),
            )
        )
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(initial_size * in_size, hidden_size, kernel_size=2, stride=2, bias=bias),
                act(val),
            )
        )
        for i in range(n_layers-1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, bias=bias),
                    act(val),
                )
            )
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_size, out_size, kernel_size=1, stride=1, bias=bias), act(val)
            )
        )

        for seq in self.layers:
            for layer in seq.children():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)

    def forward(self, x):
        x = self.layers[0](x)
        x = self.layers[1](x)
        for layer in self.layers[2:-1]:
            x = x + layer(x)
        x = self.layers[-1](x)
        return x


class GlobalContextModule(nn.Module):
    def __init__(self, out_size=128):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(512, out_size)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.act(x)
        return x


class Head(nn.Module):
    def __init__(
        self,
        in_size=128,
        hidden_size=128,
        out_size=4,
        n_layers=4,
        bias=True,
        act=nn.LeakyReLU,
        val=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_size, hidden_size, kernel_size=1, stride=1, padding=0, bias=bias),
                act(val),
            )
        )
        for i in range(n_layers - 2):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_size, hidden_size, kernel_size=1, stride=1, padding=0, bias=bias
                    ),
                    act(val),
                )
            )

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_size, out_size, kernel_size=1, stride=1, padding=0, bias=bias),
                act(val),
            )
        )

        for seq in self.layers:
            for layer in seq.children():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = x + layer(x)
        x = self.layers[-1](x)
        return x


class l1_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss = torch.abs(x - y).mean()
        return loss


class hard_log_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss = (-1 * torch.log(1 - torch.clamp(torch.abs(x - y),0,1) + 1e-6)).mean()
        return loss



