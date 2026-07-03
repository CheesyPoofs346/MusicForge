"""Smoke test: full API flow with the GPU generator mocked out. Run: python test_smoke.py"""
import os
import time
import wave

os.environ["MIX"] = "off"  # keep tests fast — no Demucs load
import server
from fastapi.testclient import TestClient


def fake_audio(prompt, seconds, out_path):
    with wave.open(str(out_path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(32000)
        w.writeframes(b"\x00\x00" * 32000)  # 1s silence


server.generate_song = lambda prompt, lyrics, seconds, out_path, ref_path=None: fake_audio(prompt, seconds, out_path)
server.generate_song_heartmula = lambda prompt, lyrics, seconds, out_path: fake_audio(prompt, seconds, out_path)
server.enhance_and_write = lambda p, g, m: ("mock, tags", "[Verse 1]\nla la la" if m == "vocal" else None, "Mock Title")

client = TestClient(server.app)

# auth: tunnel requests (cf header) need the key; localhost doesn't
cf = {"cf-connecting-ip": "1.2.3.4"}
assert client.get("/api/tracks", headers=cf).status_code == 401
r = client.get(f"/?key={server.APP_KEY}", headers=cf, follow_redirects=False)
assert r.status_code == 307 and "mf_key" in r.headers.get("set-cookie", "")
client.cookies.set("mf_key", server.APP_KEY)
assert client.get("/api/tracks", headers=cf).status_code == 200

r = client.post("/api/generate", json={"prompt": "test song", "genre": "Pop", "mode": "vocal"})
assert r.status_code == 200, r.text
tid = r.json()["id"]

for _ in range(50):
    t = client.get(f"/api/tracks/{tid}").json()
    if t["status"] in ("done", "failed"):
        break
    time.sleep(0.1)
assert t["status"] == "done", t
assert t["lyrics"].startswith("[Verse 1]")

assert client.get(f"/api/tracks/{tid}/audio").status_code == 200
peaks = client.get(f"/api/tracks/{tid}/peaks").json()
assert len(peaks["peaks"]) == 160 and peaks["duration"] > 0
assert "attachment" in client.get(f"/api/tracks/{tid}/audio?download=1").headers["content-disposition"]
assert client.get(f"/share/{tid}").status_code == 200
assert client.post("/api/generate", json={"prompt": "", "genre": "Pop", "mode": "vocal"}).status_code == 400
assert client.post("/api/generate", json={"prompt": "x", "genre": "Pop", "mode": "vocal", "engine": "bogus"}).status_code == 400
assert client.delete(f"/api/tracks/{tid}").json()["ok"]
assert client.get(f"/api/tracks/{tid}").status_code == 404

# Pro engine + custom length flow
r = client.post("/api/generate", json={"prompt": "pro test", "genre": "Pop", "mode": "instrumental",
                                        "engine": "pro", "seconds": 120})
assert r.status_code == 200, r.text
ptid = r.json()["id"]
for _ in range(50):
    t = client.get(f"/api/tracks/{ptid}").json()
    if t["status"] in ("done", "failed"):
        break
    time.sleep(0.1)
assert t["status"] == "done", t
assert t["engine"] == "pro", t
assert client.get(f"/api/tracks/{ptid}/audio").status_code == 200
client.delete(f"/api/tracks/{ptid}")

assert client.get("/").status_code == 200

print("smoke test: all passed")
