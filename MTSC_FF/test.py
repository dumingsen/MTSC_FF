# class WindowAttention(nn.Module):##
#     def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, wa_dim, wa, prob, mask,
#         dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.
#                  ):
#         super().__init__()
#         inner_dim = head_dim * heads
#
#         self.heads = heads
#         self.scale = head_dim ** -0.5
#         self.window_size = window_size
#         self.relative_pos_embedding = relative_pos_embedding
#         self.shifted = shifted
#         self.prob = prob
#         self.dropout = nn.Dropout(0.1)
#
#         if self.shifted:##如果shifted
#             displacement = window_size // 2
#             self.cyclic_shift = CyclicShift(-displacement)
#             self.cyclic_back_shift = CyclicShift(displacement)
#             self.left_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, MASK=mask),
#                                           requires_grad=False)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#
#         if self.relative_pos_embedding:
#             self.relative_indices = get_relative_distances(window_size) + window_size - 1
#             self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1))
#         else:
#
#             self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))
#
#         self.to_out = nn.Linear(inner_dim, dim)
#
#         self.window_attention = nn.Linear(wa_dim, wa_dim)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.activation = nn.ReLU()
#         self.wa = wa
#
# ##=================================================================================
#         assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'
#
#         self.seq_len = seq_len
#         self.k = k
#
#         self.heads = heads
#
#         dim_head = default(dim_head, dim // heads)
#         self.dim_head = dim_head
#
#         self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
#
#         kv_dim = dim_head if one_kv_head else (dim_head * heads)
#         self.to_k = nn.Linear(dim, kv_dim, bias=False)
#         self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))
#
#         self.share_kv = share_kv
#         if not share_kv:
#             self.to_v = nn.Linear(dim, kv_dim, bias=False)
#             self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))
#
#         self.dropout = nn.Dropout(dropout)
#         self.to_out = nn.Linear(dim_head * heads, dim)
#     def forward(self, x):
#         if self.shifted:####
#             x = self.cyclic_shift(x)
#
#         # b, n, t,c, h = *x.shape, self.heads
#         b, p, f, h = *x.shape, self.heads  # 32,128,96, 3     #128代表分了多少段，96是特征
# #1
#         #b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k
#
# # qkv
# #         kv_len = n if context is None else context.shape[1]
# #         assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'
# #
# #         queries = self.to_q(x)
# #         proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)
# #         kv_input = x if context is None else context
# #         keys = self.to_k(kv_input)
# #         values = self.to_v(kv_input) if not self.share_kv else keys
#
#         if p <= self.window_size:
#             self.window_size = p
#         # b batch_size   p : patch number f: feature
#
#         if self.wa:
#             y = self.activation(self.window_attention(self.avgpool(x).reshape(b, p)))
#             x = x * y.unsqueeze(2).repeat(1, 1, f)
#
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         print('qkv',qkv.shape)
#         new_p = p // self.window_size
#         # q, k, v = map(
#         #     lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
#         #                         h=h, w_h=self.window_size, w_w=self.window_size), qkv)
#
#         ##qkv矩阵
#         q, k, v = map(
#             lambda t: rearrange(t, 'b (new_w new_p) (head dim) -> b head new_p new_w dim',head=h, new_w=self.window_size), qkv)
#         # q  {batch_size,head,patch_num,window_size,feature}
#         start_time = time.time()
#
#
#         if not self.prob:###不稀疏
#             dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale#爱因斯坦求和
#
#             if visual_feature_map:##特征图
#                 import matplotlib.pyplot as plt
#                 import seaborn as sns
#                 for i in range(int(dots.shape[1])):
#                     sns.heatmap(torch.softmax(dots[0][i][0], dim=-1).cpu().detach().numpy())
#                     # plt.savefig('heatmap_{0}.pdf'.format(i), format='pdf')
#                     plt.show()
#
#             if self.relative_pos_embedding:##相对位置bias
#                 dots += self.pos_embedding[self.relative_indices[:, :].to(torch.long)]
#             else:
#                 dots += self.pos_embedding
#
#             if self.shifted:###
#                 dots[:, :, -new_p:] += self.left_mask
#
#             attn = self.dropout(dots.softmax(dim=-1))
#             out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
#
#         else:#应用稀疏策略
#
# # # project
# #             kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)  ###需要投影的k的值
# #
# #             # 投影k、v长度维度为k
# #             # project keys and values along the sequence length dimension to k
# #             keys, values = map(proj_seq_len, zip((keys, values), kv_projs))
# #
# #             # 融合
# #             # merge head into batch for queries and key / values
# #             queries = queries.reshape(b, n, h, -1).transpose(1, 2)
# #
# #             merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
# #             keys, values = map(merge_key_values, (keys, values))
# #
# #             # 注意力
# # # attention qk
# #             dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
# #             attn = dots.softmax(dim=-1)
# #             attn = self.dropout(attn)
# # # attn
# #             out = torch.einsum('bhnk,bhkd->bhnd', attn, values)
# #
# #             # 分割head
# #             # split heads
# #             out = out.transpose(1, 2).reshape(b, n, -1)
#
#             ##1
#             scores_top, index = self._compute_q_k(q, k)##qk矩阵和topk
#             scores_top = scores_top * self.scale
#             ##2 qk增加偏置
#             if self.relative_pos_embedding:
#                 scores_top += self.pos_embedding[self.relative_indices[index].to(torch.long)]
#             else:
#                 scores_top += self.pos_embedding
#
#             if self.shifted:###shifted
#                 scores_top[:, :, -new_p:] += self.left_mask[index]
#             #3 获得context
#             context = self._get_initial_context(v, self.window_size)
#             #4 根据topk query更新context
#             out = self._update_context(context, v, scores_top, index)
#
#         out = rearrange(out, 'b head patch window_size dim -> b (patch window_size) (head dim)',
#                         head=h)
#         out = self.to_out(out)
#
#         if self.shifted:##shifted
#             out = self.cyclic_back_shift(out)
#
#         return out
#
#     def _compute_q_k(self, q, k):#C
#         B, Head, patch, L_Q, D = q.shape
#         _, _, _, L_K, _ = k.shape
#
#         U_part = 5 * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
#         u = 5 * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)   #u是采样频率
#
#         U_part = U_part if U_part < L_K else L_K
#         u = u if u < L_Q else L_Q
#
#         scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)
#
#         return scores_top, index
#
#     def _prob_QK(self, Q, K, sample_k, n_top):##copy
#         B, H, p, L_K, D = K.shape
#         _, _, p, L_Q, _ = Q.shape
#
#         #计算sampled qk
#         K_expand = K.unsqueeze(-3).expand(B, H, p, L_Q, L_K, D)  # B ,H ,P ,L_Q, L_K
#         index_sample = torch.randint(L_K, (
#             L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q             #从0~96中选出随机L_Q*sample_k个数
#         K_sample = K_expand[:, :, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # 32 8 96 96 64  -> 32 8 96 25 64
#         Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
#         # 32 8 96 25  = 32 8 96 1 64  * 32 8 96 64 25
#
#         #找topk稀疏
#         # find the Top_k query with sparisty measurement
#         M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
#         M_top = M.topk(n_top, sorted=False)[1]
#
#         ##使用reduced q计算qk
#         Q_reduce = Q[torch.arange(B)[:, None, None, None],
#                    torch.arange(H)[None, :, None, None],
#                    torch.arange(p)[None, None, :, None],
#                    M_top, :]  # factor*ln(L_q)
#         Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
#
#         return Q_K, M_top
#
#     def _get_initial_context(self, V, L_Q):###copy
#         B, H, p, L_V, D = V.shape
#
#         V_sum = V.mean(dim=-2)
#         contex = V_sum.unsqueeze(-2).expand(B, H, p, L_Q, V_sum.shape[-1]).clone()
#
#         return contex
#
#     def _update_context(self, context_in, V, scores, index):###copy
#         B, H, P, L_V, D = V.shape
#
#         attn = self.dropout(torch.softmax(scores, dim=-1))  # nn.Softmax(dim=-1)(scores)
#
#         context_in[torch.arange(B)[:, None, None, None],
#         torch.arange(H)[None, :, None, None],
#         torch.arange(P)[None, None, :, None],
#         index, :] = torch.matmul(attn, V).type_as(context_in)
#
#         return context_in


# class WindowAttention(nn.Module):  ##
#     def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, wa_dim, wa, prob, mask,
#                  # down_dim=length,  # length = 1536 * 2，降维维度
#                  # hidden_dim=(96, 192, 62),                 # layers=(2, 2, 6, 2),                 # heads=(3, 6, 12, 24),
#                  # channels=in_channel,                 # num_classes=num_class,                 # head_dim=32,
#                  # window_size=args.window, 64                # downscaling_factors=(4, 2, 2, 2),  # 代表多长的时间作为一个特征
#                  # relative_pos_embedding=True,
#                  # dim, ##每一层的维度
#                  seq_len,  ##序列长度
#                  k=256,  ##投影的长度
#                  # heads = 8,##头部数量
#                  dim_head=None,  ##头部维度
#                  one_kv_head=False, share_kv=False, dropout=0.
#                  #     dim = 512,                 #     seq_len = 4096,
#                  #     heads = 8,                 #     k = 256,
#                  #     one_kv_head = True,                 #     share_kv = True
#                  ):
#         super().__init__()
#         inner_dim = head_dim * heads
#
#         self.heads = heads
#         self.scale = head_dim ** -0.5
#         self.window_size = window_size
#         self.relative_pos_embedding = relative_pos_embedding
#         self.shifted = shifted
#         self.prob = prob
#         self.dropout = nn.Dropout(0.1)
#
#         if self.shifted:  ##如果shifted
#             displacement = window_size // 2
#             self.cyclic_shift = CyclicShift(-displacement)
#             self.cyclic_back_shift = CyclicShift(displacement)
#             self.left_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, MASK=mask),
#                                           requires_grad=False)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  ##inner_dim = head_dim * heads
#
#         if self.relative_pos_embedding:
#             self.relative_indices = get_relative_distances(window_size) + window_size - 1  ##相对位置索引，首先加上m-1
#             self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1))  # 然后行标乘以2m-1，返回一个正态分布
#         else:
#
#             # 可以参与梯度，进行forward
#             self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))  # 次方
#
#         self.to_out = nn.Linear(inner_dim, dim)
#
#         self.window_attention = nn.Linear(wa_dim, wa_dim)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.activation = nn.ReLU()
#         self.wa = wa
#
#         ##=================================================================================
#         assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'
#
#         self.seq_len = seq_len
#         self.k = k
#
#         self.heads = heads
#         dim_head = head_dim
#         dim_head = default(dim_head, dim // heads)  # 头部维度
#         self.dim_head = dim_head
#
#         self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
#
#         kv_dim = dim_head if one_kv_head else (dim_head * heads)
#         self.to_k = nn.Linear(dim, kv_dim, bias=False)
#         self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))
#
#         self.share_kv = share_kv
#         if not share_kv:
#             self.to_v = nn.Linear(dim, kv_dim, bias=False)
#             self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))
#
#         self.dropout = nn.Dropout(dropout)
#         self.to_out = nn.Linear(dim_head * heads, dim)
#
#     def forward(self, x, context=None, **kwargs):
#         if self.shifted:  ####
#             x = self.cyclic_shift(x)
#
#         # b, n, t,c, h = *x.shape, self.heads
#         # b, p, f, h = *x.shape, self.heads  # 32,128,96, 3     #128代表分了多少段，96是特征
#         # 1
#         b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k
#
#         # qkv
#         kv_len = n if context is None else context.shape[1]
#         assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'
#
#         queries = self.to_q(x)  # self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
#         proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)
#         kv_input = x if context is None else context
#         keys = self.to_k(kv_input)
#         values = self.to_v(kv_input) if not self.share_kv else keys
#         # =====================================================================================================================
#         if n <= self.window_size:
#             self.window_size = n
#         # b batch_size   p : patch number f: feature
#
#         if self.wa:
#             y = self.activation(self.window_attention(self.avgpool(x).reshape(b, n)))
#             x = x * y.unsqueeze(2).repeat(1, 1, d)
#
#         qkv = self.to_qkv(x).chunk(3, dim=-1)  # 分块
#         print('qkv', qkv.shape)
#         new_p = n // self.window_size
#         # q, k1, v = map(
#         #     lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
#         #                         h=h, w_h=self.window_size, w_w=self.window_size), qkv)
#
#         ##qkv矩阵
#         # map函数的原型是map(function, iterable, …)，将function应用于iterable的每一个元素，结果以列表的形式返回
#         q, k1, v = map(
#             lambda t: rearrange(t, 'b (new_w new_p) (head dim) -> b head new_p new_w dim', head=h,
#                                 new_w=self.window_size)
#             , qkv)  # def rearrange(inputs, pattern, **axes_lengths) ⟶ \longrightarrow ⟶ transform_inputs
#         # q  {batch_size,head,patch_num,window_size,feature}
#         start_time = time.time()
#
#         if not self.prob:  ###不稀疏
#             dots = einsum('b h w i d, b h w j d -> b h w i j', q, k1) * self.scale  # 爱因斯坦求和,q，k
#
#             if visual_feature_map:  ##特征图
#                 import matplotlib.pyplot as plt
#                 import seaborn as sns
#                 for i in range(int(dots.shape[1])):
#                     sns.heatmap(torch.softmax(dots[0][i][0], dim=-1).cpu().detach().numpy())
#                     # plt.savefig('heatmap_{0}.pdf'.format(i), format='pdf')
#                     plt.show()
#
#             if self.relative_pos_embedding:  ##相对位置bias
#                 dots += self.pos_embedding[self.relative_indices[:, :].to(torch.long)]
#             else:
#                 dots += self.pos_embedding
#
#             if self.shifted:  ###
#                 dots[:, :, -new_p:] += self.left_mask
#
#             attn = self.dropout(dots.softmax(dim=-1))
#             out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)  # v
#
#         else:  # 应用稀疏
#
#             # project
#             kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)  ###需要投影的k的值
#
#             # 投影k、v长度维度为k
#             # project keys and values along the sequence length dimension to k
#             keys, values = map(proj_seq_len, zip((keys, values), kv_projs))
#
#             # 融合
#             # merge head into batch for queries and key / values
#             queries = queries.reshape(b, n, h, -1).transpose(1, 2)
#
#             merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
#             keys, values = map(merge_key_values, (keys, values))
#
#             # 注意力
#             # attention qk
#             dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
#             attn = dots.softmax(dim=-1)
#             attn = self.dropout(attn)
#
#             # ##1
#             # scores_top, index = self._compute_q_k(q, k)##qk矩阵和topk
#             # scores_top = scores_top * self.scale
#             ##2 qk增加偏置
#             if self.relative_pos_embedding:  ##true
#                 # scores_top += self.pos_embedding[self.relative_indices[index].to(torch.long)]###找对应的index
#                 attn += self.pos_embedding[self.relative_indices[:, :].to(torch.long)]
#             else:
#                 attn += self.pos_embedding
#
#             if self.shifted:  ###shifted
#                 attn[:, :, -new_p:] += self.left_mask
#
#             # attention attn
#             out = torch.einsum('bhnk,bhkd->bhnd', attn, values)
#             # 分割head
#             # split heads
#             out = out.transpose(1, 2).reshape(b, n, -1)
#
#             # #3 获得context
#             # context = self._get_initial_context(v, self.window_size)
#             # #4 根据topk query更新context
#             # out = self._update_context(context, v, scores_top, index)
#
#         out = rearrange(out, 'b head patch window_size dim -> b (patch window_size) (head dim)',
#                         head=h)
#         out = self.to_out(out)
#
#         if self.shifted:  ##shifted
#             out = self.cyclic_back_shift(out)
#
#         return out
#
#     def _compute_q_k(self, q, k):  # C
#         B, Head, patch, L_Q, D = q.shape
#         _, _, _, L_K, _ = k.shape
#
#         U_part = 5 * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
#         u = 5 * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)   #u是采样频率
#
#         U_part = U_part if U_part < L_K else L_K
#         u = u if u < L_Q else L_Q
#
#         scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)
#
#         return scores_top, index
#
#     def _prob_QK(self, Q, K, sample_k, n_top):  ##copy
#         B, H, p, L_K, D = K.shape
#         _, _, p, L_Q, _ = Q.shape
#
#         # 计算sampled qk
#         K_expand = K.unsqueeze(-3).expand(B, H, p, L_Q, L_K, D)  # B ,H ,P ,L_Q, L_K
#         index_sample = torch.randint(L_K, (
#             L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q             #从0~96中选出随机L_Q*sample_k个数
#         K_sample = K_expand[:, :, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # 32 8 96 96 64  -> 32 8 96 25 64
#         Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
#         # 32 8 96 25  = 32 8 96 1 64  * 32 8 96 64 25
#
#         # 找topk稀疏
#         # find the Top_k query with sparisty measurement
#         M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
#         M_top = M.topk(n_top, sorted=False)[1]
#
#         ##使用reduced q计算qk
#         Q_reduce = Q[torch.arange(B)[:, None, None, None],
#                    torch.arange(H)[None, :, None, None],
#                    torch.arange(p)[None, None, :, None],
#                    M_top, :]  # factor*ln(L_q)
#         Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
#
#         return Q_K, M_top
#
#     def _get_initial_context(self, V, L_Q):  ###copy
#         B, H, p, L_V, D = V.shape
#
#         V_sum = V.mean(dim=-2)
#         contex = V_sum.unsqueeze(-2).expand(B, H, p, L_Q, V_sum.shape[-1]).clone()
#
#         return contex
#
#     def _update_context(self, context_in, V, scores, index):  ###copy
#         B, H, P, L_V, D = V.shape
#
#         attn = self.dropout(torch.softmax(scores, dim=-1))  # nn.Softmax(dim=-1)(scores)
#
#         context_in[torch.arange(B)[:, None, None, None],
#         torch.arange(H)[None, :, None, None],
#         torch.arange(P)[None, None, :, None],
#         index, :] = torch.matmul(attn, V).type_as(context_in)
#
#         return context_in


# attn = LinformerSelfAttention(
#     dim = 512,
#     seq_len = 4096,
#     heads = 8,
#     k = 256,
#     one_kv_head = True,
#     share_kv = True
# )
# x = torch.randn(1, 4096, 512)
# attn(x) # (1, 4096, 512)

# model = LinformerLM( numtokens = 20000, dim = 512, seqlen = 4096, depth = 12, heads = 8, dimhead = 128,
# be able to set the dimension of each head in multi-head attention k = 256, # this is the k that the key/values
# are projected to along the sequence dimension onekvhead = True, # share one key/value head across all heads sharekv = False,
# share the same projection for keys and values ,reversible = True # make network reversible, like Reformer )

# 多头维度 ；同时也是kv映射的维度，one_kv_head=ture；share one key/value head across all heads share_kv = False,分享相同映射对于k v
#

# class LinformerSelfAttention(nn.Module):
#     def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
#         super().__init__()
#         assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'
#
#         self.seq_len = seq_len
#         self.k = k
#
#         self.heads = heads
#
#         dim_head = default(dim_head, dim // heads)
#         self.dim_head = dim_head
#
#         self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
#
#         kv_dim = dim_head if one_kv_head else (dim_head * heads)
#         self.to_k = nn.Linear(dim, kv_dim, bias=False)
#         self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))
#
#         self.share_kv = share_kv
#         if not share_kv:
#             self.to_v = nn.Linear(dim, kv_dim, bias=False)
#             self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))
#
#         self.dropout = nn.Dropout(dropout)
#         self.to_out = nn.Linear(dim_head * heads, dim)
#
#     def forward(self, x, context=None, **kwargs):
#         b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k
#
#         # qkv
#         kv_len = n if context is None else context.shape[1]
#         assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'
#
#         queries = self.to_q(x)
#         proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)
#         kv_input = x if context is None else context
#         keys = self.to_k(kv_input)
#         values = self.to_v(kv_input) if not self.share_kv else keys
#
#         # project
#         kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)  ###需要投影的k的值
#
#         # 投影k、v长度维度为k
#         # project keys and values along the sequence length dimension to k
#         keys, values = map(proj_seq_len, zip((keys, values), kv_projs))
#
#         # 融合
#         # merge head into batch for queries and key / values
#         queries = queries.reshape(b, n, h, -1).transpose(1, 2)
#
#         merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
#         keys, values = map(merge_key_values, (keys, values))
#
#         # 注意力
#         # attention qk
#         dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
#         attn = dots.softmax(dim=-1)
#         attn = self.dropout(attn)
#
#         # attn
#         out = torch.einsum('bhnk,bhkd->bhnd', attn, values)
#
#         # 分割head
#         # split heads
#         out = out.transpose(1, 2).reshape(b, n, -1)
#         return self.to_out1(out)