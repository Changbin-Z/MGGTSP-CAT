
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_blocks, rate):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_blocks,norm=nn.LayerNorm(input_dim))
    def forward(self, x):
        src = x
        out = self.transformer_encoder(src)
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding) 
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   
            in_channels = num_inputs if i == 0 else num_channels[i-1]  
            out_channels = num_channels[i]  
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

        
#---------------------------------------------------#
#   MGGTSP-CAT
#---------------------------------------------------#
class Encoder(nn.Module):
    def __init__(self, input_size,window_size,step,hidden_size):
        super(Encoder, self).__init__()
        self.window_size = window_size
        self.step = step
        
        #Transformer
        self.encoder1 = TransformerEncoder(input_dim = window_size, num_heads= 16, ff_dim = 256, num_blocks = 2,rate = 0.3)
        self.encoder2 = TransformerEncoder(input_dim = input_size, num_heads= 10, ff_dim = 256, num_blocks = 2,rate = 0.3)
        #TCN
        self.tcn1 = TemporalConvNet(input_size, [hidden_size], kernel_size=3, dropout=0.3)
        self.tcn2 = TemporalConvNet(window_size//2, [hidden_size], kernel_size=3, dropout=0.3)
        # Bi-LSTM
        self.lstm1 = nn.LSTM(input_size=window_size//4, hidden_size = hidden_size, batch_first=True, dropout=0.3,num_layers=1,bidirectional = True)
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size = hidden_size, batch_first=True, dropout=0.3,num_layers=1,bidirectional = True)     

    def forward(self, x1,x2,x3):
    
        x1 = self.encoder1(x1.permute(0,2,1)) 
        x1 = self.encoder2(x1.permute(0,2,1))
        
        x2 = self.tcn1(x2.permute(0,2,1))  
        x2 = self.tcn2(x2.permute(0,2,1)) 
        
        x3,_ = self.lstm1(x3.permute(0,2,1))    
        x3,_ = self.lstm2(x3.permute(0,2,1))
        
        return x1,x2,x3
    
    
class Decoder(nn.Module):
    def __init__(self,input_size,window_size,step,hidden_size, output_size):
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.step = step
        
        # Cross Attention
        self.cross_attn = CrossAttention(query_dim = input_size, context_dim=hidden_size)
        self.prenorm_CA = PreNorm(input_size, self.cross_attn, context_dim=hidden_size)
        self.ff = FeedForward(input_size, dropout=0.1)
        self.prenorm_ff = PreNorm(input_size, self.ff)
        
        self.cross_attn1 = CrossAttention(query_dim = input_size, context_dim=hidden_size*2)
        self.prenorm_CA1 = PreNorm(input_size, self.cross_attn1, context_dim=hidden_size*2)
        self.ff1 = FeedForward(input_size, dropout=0.1)
        self.prenorm_ff1 = PreNorm(input_size, self.ff1)
        
        # Linear
        self.fc1 = nn.Linear(window_size,step)
        self.fc2 = nn.Linear(input_size,output_size)

    def forward(self, x1,x2,x3): 
        
        # Cross Attention
        cross_attn_out1 = self.prenorm_CA(x1, context=x2)
        # Feed-Forward
        ff_out1 = self.prenorm_ff(cross_attn_out1) 
        # Cross Attention
        cross_attn_out2 = self.prenorm_CA1(ff_out1, context=x3)
        # Feed-Forward
        ff_out2 = self.prenorm_ff1(cross_attn_out2) 
        # Linear
        x = self.fc1(ff_out2.permute(0,2,1)) 
        x = self.fc2(x.permute(0,2,1))
        
        return x 
    

class MGGTSP_CAT(nn.Module):
    def __init__(self, input_size, window_size, step, hidden_size, output_size=1):
        super(MGGTSP_CAT, self).__init__()
        self.encoder = Encoder(input_size, window_size, step, hidden_size)
        self.decoder = Decoder(input_size, window_size, step, hidden_size, output_size)
                
    def forward(self, x1,x2,x3): 
        x1,x2,x3 = self.encoder(x1,x2,x3)
        dec_output = self.decoder(x1,x2,x3)  
        
        return dec_output