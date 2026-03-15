import torch
import torch.nn.functional as F

from model import Transformer
from tokenizer import CharTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

chars = torch.load("chars.pt")

tokenizer = CharTokenizer(chars)

model = Transformer(
    tgt_vocab_size=len(chars)
).to(device)

model.load_state_dict(torch.load("model.pt",map_location=device))
model.eval()


def generate(start,max_new_tokens=200):

    context = torch.tensor(
        tokenizer.encode(start),
        dtype=torch.long
    ).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):

        logits = model(context)

        logits = logits[:,-1,:]

        probs = F.softmax(logits,dim=-1)

        next_token = torch.multinomial(probs,1)

        context = torch.cat([context,next_token],dim=1)

    return tokenizer.decode(context[0].tolist())


print(generate("ROMEO:"))