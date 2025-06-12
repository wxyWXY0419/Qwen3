# from transformers import GPT2Tokenizer, GPT2Model
# import torch# Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
# # Move model to GPU if available
# for param in model.parameters():
#     param.requires_grad = False
# model.to(device)

# def model_llm(encoded_input):
#     output = model(**encoded_input)

#     return output



# text = "Give me 3 tips to win the fottball game."
# encoded_input = tokenizer(text, return_tensors='pt').to(device)
# output = model(**encoded_input)
# print(len(output))
# x=output[0][0]
# print(x)
# print(len(x))
# print(len(x[0]))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import faiss
#模型下载

model_dir = "/extern2/zmy/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# model_name = "Qwen/Qwen3-0.6B"
# print(model_name)

# load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float32,
#     device_map="cpu"
# )

#获取模型的嵌入层权重
embedding_weights = model.get_input_embeddings().weight
print(embedding_weights)
print(len(embedding_weights))
print(len(embedding_weights[0]))

#把embedding_weights转换为numpy数组，并创建faiss索引
embedding_weights_np = embedding_weights.detach().cpu().numpy().astype('float32')
index = faiss.IndexFlatIP(embedding_weights_np.shape[1])  # 余弦相似度用内积，需归一化
faiss.normalize_L2(embedding_weights_np)
index.add(embedding_weights_np)

def model_llm(prompt=None, token_list=None):
    # prepare the model input
    if token_list is not None:
        #把输入的token_list也归一化，然后用faiss查找最近的token id
        if not isinstance(token_list, torch.Tensor):
            token_list_tensor = torch.tensor(token_list, dtype=torch.float32).to(model.device)
        else:
            token_list_tensor = token_list.to(model.device)
        token_list_np = token_list_tensor.detach().cpu().numpy().astype('float32')
        if token_list_np.ndim == 3 and token_list_np.shape[0] == 1:
            token_list_np = token_list_np.squeeze(0)
        faiss.normalize_L2(token_list_np)

        # faiss 查找最近的 token id
        D, I = index.search(token_list_np, 1)  # I.shape = [num_tokens, 1]
        input_ids = torch.tensor(I.T, dtype=torch.long).to(model.device)  # [1, num_tokens]
    elif prompt is not None:
        #原始的基于文本提示的处理方式
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            #enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    else:
        raise ValueError("Either prompt or token_list must be provided.")

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)