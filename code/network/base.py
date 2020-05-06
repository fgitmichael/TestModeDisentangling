import torch
import torch.nn as nn


def weights_init_xavier(m):
    if isinstance(m, nn.Linear)\
            or isinstance(m, nn.Conv2d)\
            or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def write_net_params(self, summary_writer, itr):
        '''
        params:
            summary_writer :    tensorboard writer attribute
            itr            :    iteration
        '''
        for name, weight in self.named_parameters():
            summary_writer.add_histogram(name, weight, itr)
            try:
                summary_writer.add_histogram(f'{name}.grad', weight.grad, itr)
            except:
                print('%s could not be written to tensorboard' % name)


def create_linear_network(input_dim, output_dim, hidden_units=[256, 256],
                          hidden_activation=nn.ReLU(), output_activation=None,
                          initializer=weights_init_xavier):
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units))
        model.append(hidden_activation)
        units = next_units

    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(output_activation)

    return nn.Sequential(*model).apply(initializer)
