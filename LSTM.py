#---------------------------------------------------#
#   Baseline: LSTM
#---------------------------------------------------#
class LSTM(nn.Module):
    def __init__(self,input_size,window_size,step, hidden_size,output_size = 1,num_layers = 1):
        super(LSTM, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size = input_size, hidden_size = hidden_size,num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(window_size,step)
        self.fc2 = nn.Linear(hidden_size,output_size)
        
    def forward(self, x):
        x, _  = self.lstm1(x)
        x = self.fc1(x.permute(0,2,1))                  
        x = self.fc2(x.permute(0,2,1))                                      
        return x