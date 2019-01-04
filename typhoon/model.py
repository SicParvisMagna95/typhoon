import torch
import torch.nn.functional as F

SEQUENCE_LENGTH = 8
TYPHOON_MIN_LENGTH = 20
PHYSICAL_NUM = 0
FEATURE_NUM = 2
INPUT_FEATURES = FEATURE_NUM+PHYSICAL_NUM
BATCH_SIZE = 200
HIDDEN_NUM = 100
NUM_LAYERS = 2
EPOCH = 100
USE_GPU = torch.cuda.is_available()

class CNN_LSTM(torch.nn.Module):
    def __init__(self):
        super(CNN_LSTM,self).__init__()
        '''cnn'''
        # (121,301)
        '''lstm'''
        self.lstm = torch.nn.LSTM(input_size=INPUT_FEATURES,
                                   hidden_size=HIDDEN_NUM,
                                   num_layers=NUM_LAYERS,
                                   batch_first=True,   # (batch, seq, feature)
                                   dropout=0)
        self.out = torch.nn.Linear(HIDDEN_NUM,2)

    def forward(self,x):
        r_out, h_state = self.lstm(x, self.initHidden(x))
        # r_out = F.relu(r_out)
        out = self.out(r_out)[:,-4:,:]  # 输出后四个点 6h，12h，18h，24h
        out = torch.sigmoid(out)
        return out

    def initHidden(self,x):
        return (torch.rand(NUM_LAYERS, x.size(0), HIDDEN_NUM).cuda(),
                    torch.rand(NUM_LAYERS, x.size(0), HIDDEN_NUM).cuda())