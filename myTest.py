import torch
from unixcoder import UniXcoder
import transformers

print(torch.__version__)
print("gpu:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)
print("---------------------------")

# Encode maximum function
func = "def f(a,b): if a>b: return a else return b"
tokens_ids = model.tokenize([func],max_length=512,mode="<encoder-only>")
print('tokens_ids:', tokens_ids)  # 先tokenization
source_ids = torch.tensor(tokens_ids).to(device)
print('source_ids:', source_ids)  # 再转为torch张量，并转到gpu上
tokens_embeddings1, max_func_embedding = model(source_ids) # 输入模型，得到多个token分别的embedding、整个函数的embedding

# Encode minimum function
func = "def f(a,b): if a<b: return a else return b"
tokens_ids = model.tokenize([func],max_length=512,mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings2, min_func_embedding = model(source_ids)

# Encode NL
nl = "return maximum value"
tokens_ids = model.tokenize([nl],max_length=512,mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings3, nl_embedding = model(source_ids)

print("---------------------------")
print("tokens_embeddings1.shape:", tokens_embeddings1.shape)
# print(tokens_embeddings1)
print("max_func_embedding.shape:", max_func_embedding.shape)
# print(max_func_embedding)
print("---------------------------")


# Normalize embedding
norm_max_func_embedding = torch.nn.functional.normalize(max_func_embedding, p=2, dim=1)  # shape: [1, 768]
norm_min_func_embedding = torch.nn.functional.normalize(min_func_embedding, p=2, dim=1)  # 先归一化再求内积，实际上就是余弦相似度
norm_nl_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)

max_func_nl_similarity = torch.einsum("ac,bc->ab",norm_max_func_embedding,norm_nl_embedding)  # 爱因斯坦求和约定，此处为求内积
min_func_nl_similarity = torch.einsum("ac,bc->ab",norm_min_func_embedding,norm_nl_embedding)

print(max_func_nl_similarity)
print(min_func_nl_similarity)