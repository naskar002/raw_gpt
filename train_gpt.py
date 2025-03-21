from dataclasses import dataclass
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import inspect

@dataclass
class GPTConfig:
    block_size:int = 1024
    n_layers:int = 12
    vocab_size:int = 50257
    n_heads: int = 12
    n_embeds: int = 768
    

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embeds,3*config.n_embeds)
        self.c_proj = nn.Linear(config.n_embeds,config.n_embeds)
        self.c_proj.NANOGPT_FLAGS = 1
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self,x):
        #x: (B,T,C)
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.chunk(3, dim=-1)
        k = k.view(B,T,self.config.n_heads,C//self.config.n_heads).transpose(1,2) #(B,nh,T,hs)
        q = q.view(B,T,self.config.n_heads,C//self.config.n_heads).transpose(1,2) #(B,nh,T,hs)
        v = v.view(B,T,self.config.n_heads,C//self.config.n_heads).transpose(1,2) #(B,nh,T,hs)

        # att = (q @ k.transpose(-2,-1)) * (1/math.sqrt(k.size(-1))) #(B,nh,T,T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] ==0 ,float("-inf"))
        # att = F.softmax(att,dim = -1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        '''use flash attention instead'''
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embeds, 4*config.n_embeds)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embeds, config.n_embeds)
        self.c_proj.NANOGPT_FLAGS = 1

    def forward(self,x):
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embeds)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embeds)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(self.config.vocab_size,self.config.n_embeds),
            'wpe':nn.Embedding(self.config.block_size,self.config.n_embeds),
            'h' : nn.ModuleList([Block(config) for _ in range(self.config.n_layers)]),
            'ln_f': nn.LayerNorm(config.n_embeds)
        })
        self.lm_head = nn.Linear(config.n_embeds,config.vocab_size,bias = False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight  # share the same weight matrix for lm_head

        self.apply(self.init_weights) # the self.apply function of nn.Moudle iteratively goes through all the parameters of the model

    def init_weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'NANOGPT_FLAGS'):
                std *= (2*self.config.n_layers)**-0.5
            torch.nn.init.normal_(module.weight,mean=0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        if isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)

    def forward(self,x,target = None):
        B,T = x.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, block size {self.config.block_size}"
        pos_emb = self.transformer.wpe(torch.arange(0,T,dtype=torch.long,device=x.device))
        token_emb = self.transformer.wte(x)
        x = token_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)#(B,T,vocab_size)
        if target is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
        else:
            loss = None
        return logits,loss
    
    def configure_optimizer(self,weight_decay,learning_rate,device_type):
        param_dict = {pn:p for pn,p in self.named_parameters()}
        #only accept parameters which has gradeint = True
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}

        # devide the parameters in 2 part which needs to be weight decayed and which does not need
        decay_params = [p for pn,p in param_dict.items() if p.dim()>=2]
        nondecay_params = [p for pn,p in param_dict.items() if p.dim()<2]

        optim_grouped_params = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)
        is_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = is_fused and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_grouped_params,lr = learning_rate,betas= (0.9,0.95),eps=1e-8,fused=use_fused)
        return optimizer



    @classmethod
    def from_pretrained(cls,model_type):
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            'gpt2':         dict(n_layers=12, n_heads=12, n_embeds=768),  # 124M params
            'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embeds=1024), # 350M params
            'gpt2-large':   dict(n_layers=36, n_heads=20, n_embeds=1280), # 774M params
            'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embeds=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPTModel(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        #init hugging face model 
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()
        

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
        assert len(sd_keys) == len(sd_keys_hf) , f"mismatched keys: {len(sd_keys)} ! = {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
 
# ----------------------------------------------------------
import tiktoken
import time
import math
num_return_sequences = 5
max_length = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

warm_up_iter = 10
max_iter = 50
max_lr = 6e-4
min_lr = max_lr * 0.1
def get_lr(iter):
    if iter< warm_up_iter:
        return max_lr *(iter+1)/warm_up_iter
    
    if iter>max_iter:
        return min_lr
    
    decay_ratio = (iter - warm_up_iter)/(max_iter- warm_up_iter)
    assert 0 <= decay_ratio <= 1
    coefficient = 1.0 + math.cos(decay_ratio * math.pi)
    result = min_lr + 0.5 * (max_lr - min_lr) * coefficient
    return result


# making dataloader
class DataLoaderLite:
    
    def __init__(self,B,T):
        self.B = B
        self.T = T
        # tokenizing the input data 
        with open('input.txt','r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = enc.encode(text)
        print(f"no of tokens : {len(self.tokens)}")
        print(f"no of batch per epoch : {len(self.tokens)//(self.B*self.T)}")

        self.init_point = 0
    def next_batch(self):

        self.buf = torch.tensor(self.tokens[self.init_point:self.init_point+((self.B*self.T)+1)],dtype=torch.long)
        x = self.buf[:-1].view(self.B,self.T)
        y = self.buf[1:].view(self.T,self.B)
        self.init_point += (self.B*self.T)

        if self.init_point+(self.B*self.T +1)>len(self.tokens):
            self.init_point = 0
        return x,y
    

#training the model----------------------------------------------------------------

#whats need to be added :
'''
1. flash attention to optimize and it's a algorithimic change
2. Hyper paramters ,adamw and grad clippinng
'''
B,T = 2,1024

total_batch_size = 524288
grad_accum_steps = total_batch_size//(B*T)

assert total_batch_size % (B*T) == 0
print(f"Total batch size : {total_batch_size}")
print(f"grad_accum_steps: {grad_accum_steps}")

torch.cuda.empty_cache()
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# torch.set_float32_matmul_precision('high') #this line does not work for nvidia geforce gtx 1650

from torch.amp import GradScaler
scaler = GradScaler() # as we are using float16 so due to precision issues 


model = GPTModel(GPTConfig).to(device)
model = torch.compile(model) # this gives faster result.

train_loader = DataLoaderLite(B,T)
# logits,loss = model(x,y)

# optimizer = torch.optim.AdamW(model.parameters(),lr = 6e-4,betas=(0.9,0.95),eps=1e-8)
# configure_optimizer(self,weight_decay,learning_rate,device_type)
optimizer = model.configure_optimizer(weight_decay = 0.1,learning_rate = max_lr,device_type = device)

for i in range(10):
    t0 = time.time()
    cumulative_loss = 0
    for baby_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device),y.to(device)
        optimizer.zero_grad()
        # with torch.amp.autocast(device_type = device,dtype=torch.float16):
        logits,loss = model(x,y)
        loss = loss / grad_accum_steps
        cumulative_loss += loss.detach()
        # import code;code.interact(local=locals())
        # scaler.scale(loss).backward()
        loss.backward() # when grad scaler is not used 
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr = get_lr(i)
    for paramgroup in optimizer.param_groups:
        paramgroup['lr'] = lr
    optimizer.step()
    '''use grad norm'''
    
    # scaler.step(optimizer)
    # scaler.update()
    torch.cuda.synchronize()
    t1 = time.time()
    time_per_token = (B*T*grad_accum_steps) // (t1-t0)
    print(f"Iter {i+1}| Loss: {cumulative_loss.item()} |total time: {(t1-t0)*1000} milisec| norm: {norm}| token/sec : {time_per_token}")
import sys; sys.exit(0)

#model eval using pretrained weights----------------------------------------------------------------
model.eval()
model.to('cuda')


torch.cuda.empty_cache()
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    logits = model(x)
    logits = logits[:,-1,:]
    probs = F.softmax(logits,dim = -1)
    topk_probs,topk_indices = torch.topk(probs,50,dim = -1)
    ix = torch.multinomial(topk_probs,1) #(B,1)
    xcol = torch.gather(topk_indices,-1,ix) #(B,1)
    x = torch.cat((x,xcol),dim = 1)

for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decode = enc.decode(tokens)
    print(">",decode)