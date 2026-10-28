import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import io, glob, json, hashlib, threading, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from tokenizers import Tokenizer

# --- модель KentiAi 2 Lucky: чистый numpy (размеры как при обучении) ---
V, D, L, H, T = 8000, 512, 6, 8, 64
tok = Tokenizer.from_file("kenti2_tokenizer.json")
raw = b"".join(open(p, "rb").read() for p in sorted(glob.glob("kenti2_part*.bin")))
W = {k: v.astype(np.float32) for k, v in np.load(io.BytesIO(raw)).items()}
del raw
eos = tok.token_to_id("<eos>")
rng = np.random.default_rng()
LOCK = threading.Lock()


def layer_norm(x, w, b):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b


def forward(new_ids, cache):
    """Прогоняет новые токены, дописывает K/V в cache, возвращает логиты последнего токена."""
    n = len(new_ids)
    past = 0 if cache[0] is None else cache[0][0].shape[1]
    hd = D // H
    x = W["emb.weight"][new_ids] + W["pos.weight"][past:past + n]
    mask = np.triu(np.full((n, past + n), -1e9, dtype=np.float32), past + 1)
    for i in range(L):
        p = f"enc.layers.{i}."
        h = layer_norm(x, W[p + "norm1.weight"], W[p + "norm1.bias"])
        qkv = h @ W[p + "self_attn.in_proj_weight"].T + W[p + "self_attn.in_proj_bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(n, H, hd).transpose(1, 0, 2)
        k = k.reshape(n, H, hd).transpose(1, 0, 2)
        v = v.reshape(n, H, hd).transpose(1, 0, 2)
        if cache[i] is not None:
            k = np.concatenate([cache[i][0], k], axis=1)
            v = np.concatenate([cache[i][1], v], axis=1)
        cache[i] = (k, v)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        a = (s @ v).transpose(1, 0, 2).reshape(n, D)
        x = x + a @ W[p + "self_attn.out_proj.weight"].T + W[p + "self_attn.out_proj.bias"]
        h = layer_norm(x, W[p + "norm2.weight"], W[p + "norm2.bias"])
        h = np.maximum(h @ W[p + "linear1.weight"].T + W[p + "linear1.bias"], 0)
        x = x + h @ W[p + "linear2.weight"].T + W[p + "linear2.bias"]
    x = layer_norm(x[-1], W["enc.norm.weight"], W["enc.norm.bias"])
    return x @ W["emb.weight"].T


def say(q, max_new=30, temp=0.7, k=40):
    start = tok.encode(q.lower().strip() + "<sep>").ids[-40:]
    cache = [None] * L
    lg = forward(start, cache)
    out = []
    for _ in range(min(max_new, T - len(start))):
        lg = lg / temp
        top = np.argpartition(lg, -k)[-k:]
        p = np.exp(lg[top] - lg[top].max())
        p /= p.sum()
        nxt = int(rng.choice(top, p=p))
        if nxt == eos:
            break
        out.append(nxt)
        lg = forward([nxt], cache)
    return tok.decode(out, skip_special_tokens=True).strip()


# --- Telegram через webhook ---
TOKEN = os.environ["BOT_TOKEN"]
SECRET = hashlib.sha256(TOKEN.encode()).hexdigest()[:32]


def api(method, **data):
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TOKEN}/{method}",
        json.dumps(data).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=20).read()


def answer(msg):
    try:
        text = msg["text"]
        if text.startswith("/start"):
            reply = "Привет! Я KentiAi 2 Lucky 🍀 Напиши что-нибудь, я отвечу (иногда странно)."
        else:
            with LOCK:
                reply = say(text[:200]) or "..."
        api("sendMessage", chat_id=msg["chat"]["id"], text=reply,
            reply_to_message_id=msg["message_id"])
    except Exception as e:
        print("ошибка ответа:", repr(e))


class Handler(BaseHTTPRequestHandler):
    def _ok(self, body=b"KentiAi 2 Lucky OK"):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._ok()

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(size)
        if self.path == "/hook" and self.headers.get("X-Telegram-Bot-Api-Secret-Token") == SECRET:
            try:
                msg = json.loads(body).get("message") or {}
                if msg.get("text"):
                    threading.Thread(target=answer, args=(msg,), daemon=True).start()
            except Exception as e:
                print("ошибка разбора:", repr(e))
        self._ok(b"ok")  # отвечаем Telegram сразу, чтобы он не повторял запрос

    def log_message(self, *args):
        pass


def set_webhook():
    url = os.environ.get("PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        print("нет адреса сервиса (RENDER_EXTERNAL_URL), webhook не установлен")
        return
    try:
        api("setWebhook", url=url.rstrip("/") + "/hook", secret_token=SECRET)
        print("webhook установлен:", url)
    except Exception as e:
        print("не удалось поставить webhook:", repr(e))


threading.Thread(target=set_webhook, daemon=True).start()
print("KentiAi 2 Lucky в сети")
ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", 10000))), Handler).serve_forever()
