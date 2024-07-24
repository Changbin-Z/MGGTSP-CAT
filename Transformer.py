#---------------------------------------------------#
#   Baseline: Transformer
#---------------------------------------------------#
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_blocks, rate):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_blocks,norm=nn.LayerNorm(input_dim))
        
    def forward(self, x):
        src = x
        out = self.transformer_encoder(src)
        return out
     
class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_blocks, rate):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads,dim_feedforward=ff_dim,dropout=rate)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_blocks,norm=nn.LayerNorm(input_dim))
        
    def forward(self, x):
        tgt = x
        memory = x
        out = self.transformer_decoder(tgt,memory)
        return out
    
class Transformer(nn.Module):
    def __init__(self, input_size,window_size,step,hidden_size,output_size=1, num_heads=2 , ff_dim=256, num_blocks = 1, rate=0.1):
        super(Transformer, self).__init__()
        self.embedding1 = nn.Linear(window_size, hidden_size)
        self.embedding2 = nn.Linear(input_size, hidden_size)
        self.encoder2 = TransformerEncoder(input_dim=hidden_size, num_heads=2, ff_dim=256, num_blocks=1,rate=0.1)
        self.decoder = TransformerDecoder(input_dim=hidden_size, num_heads=2, ff_dim=256, num_blocks=1,rate=0.1)
        self.fc1 = nn.Linear(hidden_size,step)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x = self.embedding1(x.permute(0,2,1))
        x = self.embedding2(x.permute(0,2,1))
        x = self.encoder2(x)
        x = self.decoder(x)
        x = self.fc1(x.permute(0,2,1))             
        x = self.fc2(x.permute(0,2,1))   
        return x
    