import os, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch, torch.nn as nn
from tokenizers import Tokenizer
import telebot

torch.set_num_threads(1)
torch.set_grad_enabled(False)

# --- модель (те же размеры, что при обучении) ---
V, D, L, H, T = 6000, 256, 4, 4, 64
tok = Tokenizer.from_file("kenti_tokenizer.json")
emb = nn.Embedding(V, D)
pos = nn.Embedding(T, D)
enc = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(D, H, 4 * D, 0.1, batch_first=True, norm_first=True),
    L, norm=nn.LayerNorm(D), enable_nested_tensor=False)
head = nn.Linear(D, V, bias=False)
head.weight = emb.weight

c = torch.load("kenti_lucky.pt", map_location="cpu")
emb.load_state_dict(c["emb"])
pos.load_state_dict(c["pos"])
enc.load_state_dict(c["enc"])
enc.eval()

mask = nn.Transformer.generate_square_subsequent_mask(T)
eos = tok.token_to_id("<eos>")


def fwd(x):
    n = x.shape[1]
    h = emb(x) + pos(torch.arange(n))
    return head(enc(h, mask=mask[:n, :n], is_causal=True))


def say(q, max_new=30, temp=0.7, k=40):
    start = tok.encode(q.lower().strip() + "<sep>").ids[-40:]
    ids = list(start)
    for _ in range(max_new):
        lg = fwd(torch.tensor([ids[-T:]]))[0, -1] / temp
        lg = lg.masked_fill(lg < torch.topk(lg, k).values[-1], -1e9)
        nxt = torch.multinomial(torch.softmax(lg, -1), 1).item()
        if nxt == eos:
            break
        ids.append(nxt)
    return tok.decode(ids[len(start):], skip_special_tokens=True).strip()


# --- мини-сервер, чтобы хостинг видел, что сервис жив ---
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"KentiAi 1 Lucky OK")

    def log_message(self, *args):
        pass


port = int(os.environ.get("PORT", 8000))
threading.Thread(
    target=lambda: HTTPServer(("0.0.0.0", port), Health).serve_forever(),
    daemon=True).start()

# --- Telegram ---
bot = telebot.TeleBot(os.environ["BOT_TOKEN"])


@bot.message_handler(commands=["start"])
def start(m):
    bot.reply_to(m, "Привет! Я KentiAi 1 Lucky 🍀 Напиши мне что-нибудь, я отвечу (иногда странно).")


@bot.message_handler(content_types=["text"])
def chat(m):
    bot.reply_to(m, say(m.text[:200]) or "...")


print("KentiAi 1 Lucky в сети")
bot.infinity_polling(skip_pending=True)
