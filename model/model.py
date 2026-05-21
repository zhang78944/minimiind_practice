from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional,Tuple
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

#归一化，防止梯度爆炸或梯度消失
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)       #rsqrt求的是平方根倒数

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            #输入波长b，输出对应的维度索引i.用波长找到高低频的分界线
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin

#执行代码
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    #rotate_half 是为了求旋转半向量，前后交换，后半部分取反
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)               #unsqueeze是为了形状对齐
    )  
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed

#GQA中，KV需要复制
def repeat_kv(x:torch.Tensor,n_rep:int) -> torch.Tensor:
    bs,slen,num_key_values_heads,head_dim =x.shape
    if n_rep==1:
        return x
    
    return (x[:,:,:,None,:].expands(bs,slen,num_key_values_heads,n_rep,head_dim)       #增加了一个新的维度，复制 n_rep 次
            .reshape(bs,slen,num_key_values_heads*n_rep,head_dim))     #合并

class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads
        assert args.num_attention_heads %  self.num_key_value_heads ==0,"num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads=args.num_attention_heads     #Q的头数
        #self.num_key_value_heads = args.num_key_value_heads
        self.n_rep=self.n_local_heads //self.num_key_value_heads
        self.head_dim=args.hidden_size // args.num_attention_heads   #每个头的维度

        #投影层
        self.q_proj=nn.Linear(args.hidden_size,args.num_attention_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(args.hidden_size,args.num_key_value_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(args.hidden_size,args.num_key_value_heads*self.head_dim,bias=False)
        self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.attn_dropout=nn.Dropout(args.dropout)  #防止过拟合
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout
        # Flash Attention 加速      分块计算，在线softmax，重计算
        self.flash_attention=hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash_attention

        #投影，计算q k v
        #把输入拆分成多个头，用view
        #q k使用rope
        # k v 使用repeat 注意 kv cache,把前一部分的kv缓存起来，下次用的时候直接拿出来用
        #进行Attention的计算  q k^t/sqrt(d)    
        #最后拼接头，输出投影，返回
        def forward(self,x:torch.Tensor,position_embedding:Tuple[torch.Tensor,torch.Tensor],
                    past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,use_cache=False,
                    attention_mask:Optional[torch.Tensor]=None) ->torch.Tensor:
            bsz,seq_len,_=x.shape
            xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
            #拆分成多个头
            xq=xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
            xk=xk.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
            xv=xv.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
            # q k使用rope
            cos,sin =position_embedding
            xq,xk=apply_rotary_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])
            # k v 使用repeat 注意 kv cache,把前一部分的kv缓存起来，下次用的时候直接拿出来用
            if past_key_value is not None:
                xk=torch.cat([past_key_value[0],xk],dim=1)
                xv=torch.cat([past_key_value[1],xv],dim=1)
            past_kv=(xk,xv) if use_cache else None

            xq,xk,xv=(
                xq.transpose(1,2),  #要对每个序列计算，所以交换一下，把头放在前面
                #bsz,n_local_heads,seq_len,head_dim
                repeat_kv(xk,self.n_rep).transpose(1,2),
                repeat_kv(xv,self.n_rep).transpose(1,2),)
            
            #Attention计算
            if self.flash and seq_len>1 and (attention_mask is None or torch.all(attention_mask==1)):
               attn_mask=(
                   None
                   if attention_mask is None
                   else attention_mask.view(bsz,1,1,-1).expand(bsz,self.n_local_heads,seq_len,-1).bool()               
               )

               output=F.scaled_dot_product_attention(xq,xk,xv,attn_mask=attn_mask,dropout_p=self.dropout if self.training else 0.0,
                                                     is_causal=True)
            else:
                scores=(xq@xk.transpose(-2,-1)/math.sqrt(self.head_dim))
                scores=scores+torch.triu(
                    torch.full((seq_len,seq_len),float('-inf'),device=scores.device),
                ).unsqueeze(0).unsqueeze(0)

                #padding掩码
                if attention_mask is not None:
                    extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
                    extended_attention_mask=(1-extended_attention_maask)*-1e9
                    scores=scores+extended_attention_mask

            scores=F.softmax(scores,float(),dim=-1).type_as(xq)
            scores=self.attn_dropout(scores)
            output=scores@xv

        output=output.transpose(1,2).reshape(bsz,seq_len,-1)  #把头合并回来,[batch, seq_len, hidden_size]
        ouput=self.resid_dropout(self.o_proj(output))
        return output,past_kv
    
class FeedForward(nn.Moudule):
    #初始化
    #升维
    #降维
    #门控
    #dropout
    #激活函数
    def __init__(self,args:MokioMindConfig):
        super().__init__()
        #SwiGLU
        if args.intermediate_size is None:
            intermediate_size =int(args.hidden_size*8/3)    #把维度升到2.66左右
            args.intermediate_size=64*((intermediate_size+64-1)//64)  #向上取整64的倍数

        self.up_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.down_proj=nn.Linear(args.intermediate_size,args.hidden_size,bias=False)
        self.gate_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.dropout=nn.Dropout(args.dropout)
        self.act_fn=ACT2FN[args.hidden_act]  #激活函数
    def forward(self,x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)*self.up_proj(x))))   #gate * up 自适应特征筛选

class MokioMindBlock(nn.Module):
    def __init__(self,layer_id:int,config:MokioMindConfig):
        super().__init__()
        self.num_attention_heads=config.num_attention_heads
        self.hidden_size=config.hidden_size
        self.head_dim=self.hidden_size//self.num_attention_heads
        self.self_attn=Attention(config)

        self.layer_id=layer_id
        self.input_layernorm=RMSNorm(self.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.mlp=FeedForward(config)
    def forward(self,hidden_sates,position_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
        residual=hidden_states
        hidden_states,present_key_value=self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states=residual+hidden_states
        hidden_states=hidden_states+self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states,present_key_value

class MokioMindModel(nn.Module):
    def __init__(self,config:MokioMindConfig):
        super().__init__()
        self.vpcab_size,self.num_hidden_layeres=(
            config.vocab_size,
            config.num_hidden_layers
        )

        self.embed_tokens=nn.Embedding(config.vocab_size,comfig,hidden_size)  #变成稠密向量
        self.dropout=nn.Dropout(config.dropout)

        self.layers=nn.ModuleList(      #列表，把多个Layer放在一起
            [MokioMindBlock(i,config) for i in range(self.num_hidden_layers)]    #隐藏层有几个维度，在中间需要插入K个前面编写的Block
            #MokioMindBlock=一层注意力（Attention + FFN）
        )

        self.norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)

        #Rope 预计算，保证Rope旋转值是固定的，避免重复计算
        freqs_cos,freqs_sin=precompute_freqs(
            dim=config.hidden_size//config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.register_buffer("freqs_cos",freqs_cos,presistent=False)    #注册缓冲区，模型参数不更新，不参与优化器，不保存到checkpoint，每次加载模型时都会重新计算frequencies
        self.register_buffer("freqs_sin",freqs_sin,presistent=False)

    def forward(
            self,
            input_ids:Optional[torch.Tensor]=None,                  # 输入的 token 序列
            attention_mask:Optional[torch.Tensor]=None,             # 注意力掩码
            past_key_values:Optional[Tuple[torch.Tensor]]=None,     # KV 缓存
            use_cache:bool=False,                                   # 是否使用缓存
            **kwargs,
    ):
        batch_size,seq_len=input_ids.shape  #解包输入的张量

        if hasattr(past_key_values,'layers'):   #检查对象是否有属性
            past_key_values=None  #防御性清空，防止传入错误

        past_key_values=past_key_values or [None]*len(self.layers)  #如果没有缓存，创建一个空缓存位，每一组准备一个位置


        #past_key_values[0] 第0层的缓存（key,value）
        #past_key_values[0][0] 第0层的key缓存
        #past_key_values[0][0].shape[1] 已经缓存的token长度
        start_pos=(                                # start_pos指已经缓存的token长度
            past_key_values[0][0].shape[1] 
            if past_key_values[0] is not None 
            else 0
        )

        hidden_states=self.dropout(self.embed_tokens(input_ids)) #输入嵌入
        position_embeddings=(
            self.freqs_cos[start_pos:start_pos+seq_len],
            self.freqs_sin[start_pos:start_pos+seq_len],
        )

        presents=[]

        for layer_idx,(layer,past_key_values) in enumerate(
            zip(self.layers,past_key_values)
        ):
            hidden_states,present=layer(    #循环K次的那个Layer,self.layer是一堆层的列表
                hidden_states,
                position_embeddings,
                past_key_value=past_key_values,
                attention_mask=attention_mask,
            )

            presents.append(present)

        hidden_states=self.norm(hidden_states)
        
        return hidden_states,presents







    

    

    




