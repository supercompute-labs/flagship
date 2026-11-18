import json, random, re, sys, time, uuid, urllib.request
sys.path.insert(0, "."); import tune
Q = [("A shop sells pens at $3 each. Tom buys 7 pens and pays with a $50 bill. How much change does he get?", "29"),
     ("What is 17 * 23?", "391"),
     ("A train travels 240 km in 3 hours. What is its average speed in km/h?", "80"),
     ("If x + 2x + 3x = 48, what is x?", "8"),
     ("Sarah has 3 boxes with 12 apples each. She gives away 15 apples. How many apples remain?", "21"),
     ("What is the capital of Australia?", "canberra"),
     ("Which planet has the most moons as of 2023, Saturn or Jupiter? Answer one word.", "saturn"),
     ("What is the chemical symbol for tungsten?", "w"),
     ("A rectangle is 9 cm by 14 cm. What is its perimeter in cm?", "46"),
     ("What is 2 to the power of 12?", "4096"),
     ("How many prime numbers are there between 1 and 30?", "10"),
     ("What year did the Berlin Wall fall?", "1989"),
     ("A recipe needs 250 g flour per loaf. How many grams for 6 loaves?", "1500"),
     ("What is the square root of 1764?", "42"),
     ("Who wrote 'Pride and Prejudice'? Give the surname only.", "austen"),
     ("An item costs $80 and is discounted 15%. What is the sale price in dollars?", "68"),
     ("How many sides does a dodecagon have?", "12"),
     ("What is 1000 - 387?", "613"),
     ("If a car uses 6 litres per 100 km, how many litres for 350 km?", "21"),
     ("What gas do plants absorb from the air for photosynthesis? Give the chemical formula.", "co2")]
def ask(content, out=600, temp=0.0):
    body = {"model": tune.MODEL, "max_tokens": out, "temperature": temp, "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": False}}
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), tune.HDR), timeout=900))
    return d["choices"][0]["message"].get("content") or ""
def norm(s): return re.sub(r"[^a-z0-9]", " ", s.lower().replace(",", ""))
def needle(depth_tokens, frac):
    random.seed(depth_tokens + int(frac*100)); code = f"{random.randint(100000,999999)}"
    n = int(depth_tokens / 19.3); pos = int(n * frac)
    lines = [f"record{i:06d}: {tune.REC}" for i in range(1, n + 1)]
    lines.insert(pos, f"IMPORTANT NOTE: the vault access code is {code}.")
    a = ask(f"[{uuid.uuid4().hex}]\n" + "\n".join(lines) + "\nWhat is the vault access code mentioned in the notes above? Reply with just the number.", out=40)
    return code in a, a.strip()[:40]
res = {}
for name, extra in json.load(open(sys.argv[1])):
    tune.stop(); tune.start(extra); ask("hi", out=8)
    score = 0; misses = []
    for q, a in Q:
        r = ask(q + " Give the final answer clearly at the end.")
        tail = norm(r)[-160:]
        ok = re.search(r"\b" + a + r"\b", tail) is not None
        score += ok
        if not ok: misses.append((q[:40], r.strip()[-80:]))
    nd = [needle(d, f) for d, f in [(32768, 0.3), (65536, 0.5), (110000, 0.2), (110000, 0.8)]]
    story = ask("Write a vivid 250-word story about a fox who learns to sail.", out=450, temp=0.7)
    res[name] = {"qa": f"{score}/{len(Q)}", "needles": nd, "misses": misses, "story": story}
    print(json.dumps({"name": name, "qa": res[name]["qa"], "needles": [x[0] for x in nd]}), flush=True)
json.dump(res, open("./results/qgate.json", "w"), indent=1)
print("GATE DONE")
