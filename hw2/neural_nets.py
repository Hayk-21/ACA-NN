import torch.nn as nn

class fcn(nn.Module):
    def __init__(self, params):
        super(fcn, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(params['input_size'],params['hidden_size']),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(params['hidden_size'], params['output_size']),
            nn.Sigmoid()
        )



    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class cnn(nn.Module):
    def __init__(self, params):
        super(cnn, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(params['Conv1_in_channel'], params['Conv1_out_channel'], params['Conv_kernel_size'], params['Conv_stride'], params['Conv_padding']),
            nn.BatchNorm2d(params['Conv1_out_channel']),
            nn.ReLU(),
            nn.MaxPool2d(params['Pool_kernel_size'], params['Pool_stride']))
        self.layer2 = nn.Sequential(
            nn.Conv2d(params['Conv2_in_channel'], params['Conv2_out_channel'], params['Conv_kernel_size'], params['Conv_stride'], params['Conv_padding']),
            nn.BatchNorm2d(params['Conv2_out_channel']),
            nn.ReLU(),
            nn.MaxPool2d(params['Pool_kernel_size'], params['Pool_stride']))
        self.drop_out = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(params['Conv2_out_channel'] * params['img_size'] * params['img_size'], 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
