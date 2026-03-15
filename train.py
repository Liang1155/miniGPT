import torch
from model import Transformer
from dataset import get_batch
from tokenizer import CharTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("Shakespeare.txt","r") as f:
    text = f.read()

chars = sorted(list(set(text)))
torch.save(chars,"chars.pt")

tokenizer = CharTokenizer(chars)

data = torch.tensor(tokenizer.encode(text))

model = Transformer(
    tgt_vocab_size=len(chars)
).to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

block_size = 128
batch_size = 32

for step in range(10000):

    x,y = get_batch(data,batch_size,block_size)

    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    loss = criterion(
        logits.view(-1,logits.size(-1)),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(step,loss.item())

torch.save(model.state_dict(),"model.pt")