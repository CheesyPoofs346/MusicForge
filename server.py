"""MusicForge — local AI music generation (MusicGen) + lyrics (Claude)."""
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / "audio"
AUDIO_DIR.mkdir(exist_ok=True)
REF_DIR = ROOT / "reference"
REF_DIR.mkdir(exist_ok=True)
DB_PATH = ROOT / "tracks.db"

TRACK_SECONDS = int(os.environ.get("TRACK_SECONDS", "60"))
LYRICS_MODEL = os.environ.get("LYRICS_MODEL", "claude-sonnet-5")


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


with db() as c:
    c.execute(
        """CREATE TABLE IF NOT EXISTS tracks(
        id TEXT PRIMARY KEY, prompt TEXT, genre TEXT, mode TEXT,
        lyrics TEXT, status TEXT, error TEXT, created REAL)"""
    )
    for mig in ("ALTER TABLE tracks ADD COLUMN title TEXT",
                "ALTER TABLE tracks ADD COLUMN started REAL",
                "ALTER TABLE tracks ADD COLUMN engine TEXT",
                "ALTER TABLE tracks ADD COLUMN seconds INTEGER"):
        try:
            c.execute(mig)
        except sqlite3.OperationalError:
            pass  # already migrated
    # jobs die with the process — anything still "generating" at boot is an orphan
    c.execute(
        "UPDATE tracks SET status='failed', error='Interrupted by a server restart — generate it again' "
        "WHERE status IN ('queued','generating')"
    )


def set_track(tid, **fields):
    cols = ", ".join(f"{k}=?" for k in fields)
    with db() as c:
        c.execute(f"UPDATE tracks SET {cols} WHERE id=?", (*fields.values(), tid))


# ---------------------------------------------------------------- model
# ponytail: everything routes through ACE-Step (48kHz stereo) — the old
# MusicGen instrumental path was 32kHz and noticeably rougher, deleted.
_load_lock = threading.Lock()


LYRICS_PROMPT = (
    "You are a hit songwriter. Write complete, original lyrics for a {genre} song about: {prompt}\n"
    "Craft rules — these matter:\n"
    "- Concrete images and specific details, never abstractions. Show a scene, don't announce a feeling.\n"
    "- BANNED cliches: 'shining bright', 'against all odds', 'rise above', 'reach for the stars', "
    "'heart on fire', 'break these chains', or anything a motivational poster would say.\n"
    "- Short singable lines, 4-9 syllables. Lines should scan when spoken with a beat.\n"
    "- The chorus hook must be hummable after one listen: short, rhythmic, repeated.\n"
    "- Match the genre's flow: rap/trap = dense internal rhyme and ad-libs in parentheses; "
    "pop = tight repeatable hook; rock = a chorus that opens up.\n"
    "- For sung genres (anything but rap), end lines on long open vowels the singer can hold "
    "(-ay, -oh, -ee, -I) — held notes are what make it singing instead of talking.\n"
    "- Layer (doubles), (echoes) and (crowd: ...) response lines in choruses and the bridge.\n"
    "Structure tags lowercase in square brackets: [intro], [verse], [pre-chorus], [chorus], "
    "[verse], [chorus], [bridge], [outro].\n"
    "Output lyrics only, no title, no commentary."
)
# Formula distilled from prompts that produce dense, pro-sounding tracks:
# per-section arrangement + ear candy between lines + explicit mix language.
ENHANCE_PROMPT = (
    "Rewrite this rough music idea into a dense music-generation prompt following this "
    "formula: genre + exact BPM + drum character; then per-section arrangement (what the "
    "verse rides on, what the pre-chorus strips down to, what the chorus slams in with); "
    "then ear candy filling every gap (risers, fills, reverse crashes, glitch hits, "
    "percussion layers){vocal_tags}; then mix character (loud, bright, punchy, wide). "
    "Name real instruments where the genre calls for them (live drums, funk guitar, brass "
    "stabs, piano, strings, upright bass) — not everything is a synth. "
    "One flowing comma-separated prompt like these examples:\n"
    "EXAMPLE 1: Hardstyle EDM trap with frantic 150 BPM kicks, syncopated snare rolls, and "
    "chest-rattling drop impacts; verse rides fast hype-rap over clipped percussion, "
    "pre-chorus strips to chanting voices and rising synth pressure, chorus slams with "
    "giant reverse bass, crowd shouts, and call-and-response hooks, bright brutal festival mix\n"
    "EXAMPLE 2: Soul-funk anthem with live drums at 104 BPM, greasy bassline, and wah guitar; "
    "verse rides tight pocket groove with horn stabs, pre-chorus strips to rimshots and "
    "electric piano, chorus explodes with full brass section, gospel stack vocals, and "
    "tambourine, warm analog mix with punchy low end\n"
    "Max 60 words. Output only the prompt, nothing else.\n"
    "Genre: {genre}\nIdea: {prompt}"
)


def llm(tasks: list) -> list:
    """Answer each task with Claude if a key is set, else a small local LLM
    (loaded once for the batch, then freed so the music model gets the VRAM)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        return [
            client.messages.create(
                model=LYRICS_MODEL, max_tokens=1200,
                messages=[{"role": "user", "content": t}],
            ).content[0].text
            for t in tasks
        ]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 3B writes markedly less chatbot-flavored lyrics than 1.5B; still loads/unloads fast
    model_id = os.environ.get("LOCAL_LYRICS_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_id)
    lm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    lm.to("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for task in tasks:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": task}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(lm.device)
        with torch.inference_mode():
            out = lm.generate(ids, max_new_tokens=800, do_sample=True, temperature=0.8)
        results.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())
    del lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def enhance_and_write(prompt: str, genre: str, mode: str):
    """Auto-enhance: turn any prompt (however rough) into a rich tag prompt,
    plus full lyrics in vocal mode. Enhancement failure falls back to the raw prompt."""
    vocal_tags = ", vocal style, delivery and how the vocals sit in the mix" if mode == "vocal" else ""
    tasks = [
        ENHANCE_PROMPT.format(vocal_tags=vocal_tags, genre=genre, prompt=prompt),
        f"Give a {genre} song about this a memorable 2-4 word title: {prompt}\n"
        "No quotes, no explanation — output only the title.",
    ]
    if mode == "vocal":
        tasks.append(LYRICS_PROMPT.format(genre=genre, prompt=prompt))
    try:
        results = llm(tasks)
    except Exception:
        if mode == "vocal":
            raise  # vocal needs lyrics; instrumental can survive without enhancement
        return f"{genre}, {prompt}", None, None
    enhanced = results[0].strip().strip('"')
    # small local models sometimes ramble — keep it a plausible tag line
    if not enhanced or len(enhanced) > 600 or "\n\n" in enhanced:
        enhanced = f"{genre}, {prompt}"
    title = results[1].strip().strip('"').splitlines()[0][:60] or None
    return enhanced, (results[2] if mode == "vocal" else None), title


def master_audio(path: Path):
    """Mastering pass: clean sub-rumble, normalize to -14 LUFS, keep peaks safe."""
    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf
    from scipy.signal import butter, sosfiltfilt

    data, sr = sf.read(path)
    loudness = pyln.Meter(sr).integrated_loudness(data)
    if loudness == float("-inf"):  # silence — nothing to master
        return
    # 30Hz high-pass: kills inaudible sub-rumble that eats headroom and muddies the low end
    sos = butter(2, 30, "highpass", fs=sr, output="sos")
    data = sosfiltfilt(sos, data, axis=0)
    data = pyln.normalize.loudness(data, loudness, -14.0)
    # soft-knee limiter: only the loudest transients get squeezed (caps ~0.99),
    # so the track holds -14 LUFS instead of being scaled quieter
    over = np.abs(data) > 0.95
    data[over] = np.sign(data[over]) * (0.95 + np.tanh((np.abs(data[over]) - 0.95) * 8) * 0.04)
    sf.write(path, data, sr)


# ---------------------------------------------------------------- vocal model
_ace = None


def load_ace():
    global _ace
    with _load_lock:
        if _ace is None:
            # ponytail: torchaudio 2.9 delegates load/save to torchcodec (broken on
            # Windows without ffmpeg DLLs); route both through soundfile
            import torch
            import torchaudio

            def _sf_save(uri, src, sample_rate, **kw):
                import soundfile as sf

                sf.write(str(uri), src.detach().cpu().float().numpy().T, sample_rate)

            def _sf_load(path, **kw):
                import soundfile as sf

                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                return torch.from_numpy(data.T), sr

            torchaudio.save = _sf_save
            torchaudio.load = _sf_load
            from acestep.pipeline_ace_step import ACEStepPipeline

            _ace = ACEStepPipeline(dtype="bfloat16")  # auto-downloads 3.5B checkpoint
    return _ace


def unload_ace():
    """Free ACE-Step from VRAM — the Pro (HeartMuLa) engine needs nearly the whole
    card, and both can't be resident at once on 16GB."""
    global _ace
    with _load_lock:
        if _ace is not None:
            _ace = None
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Suno Competition (Pro engine). Lives in its own subfolder — separate from
# Basic's LoRA checkpoint — so nothing gets mixed up between the two.
HEARTLIB_DIR = Path(os.environ.get("HEARTLIB_DIR", r"C:\musicforge\pro"))
HEARTLIB_PY = HEARTLIB_DIR / ".venv" / "Scripts" / "python.exe"


def generate_song_heartmula(prompt: str, lyrics: str, seconds: int, out_path: Path):
    """Pro engine: Suno Competition model, run in its own venv as a subprocess
    (needs its own transformers version, incompatible with ACE-Step's env). Slow on Windows.
    Lives in C:\\musicforge\\pro — separate from Basic's LoRA folder."""
    import subprocess
    import tempfile

    if not HEARTLIB_PY.exists():
        raise RuntimeError("Pro engine not installed — Suno Competition venv missing at " + str(HEARTLIB_PY))
    unload_ace()  # free VRAM for the Pro model
    with tempfile.TemporaryDirectory() as td:
        tp, lp = Path(td) / "tags.txt", Path(td) / "lyrics.txt"
        tp.write_text(prompt, encoding="utf-8")
        lp.write_text(lyrics, encoding="utf-8")
        cmd = [
            str(HEARTLIB_PY), str(HEARTLIB_DIR / "heartlib_gen.py"),
            "--tags-file", str(tp), "--lyrics-file", str(lp),
            "--seconds", str(seconds), "--out", str(out_path),
        ]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5400, env=env)
        if r.returncode != 0 or not out_path.exists():
            raise RuntimeError("HeartMuLa failed: " + (r.stderr or r.stdout or "")[-400:])


def write_mp3(src: Path, dst: Path):
    import soundfile as sf

    data, sr = sf.read(src)
    sf.write(dst, data, sr, format="MP3")


ACESTEP15_DIR = Path(os.environ.get("ACESTEP15_DIR", r"C:\acestep15"))
ACESTEP15_PY = ACESTEP15_DIR / ".venv" / "Scripts" / "python.exe"
ACESTEP15_API_PORT = int(os.environ.get("ACESTEP15_API_PORT", "8001"))
ACESTEP15_API_URL = f"http://127.0.0.1:{ACESTEP15_API_PORT}"

_acestep15_api_proc = None
_acestep15_api_lock = threading.Lock()


def _acestep15_api_get(path: str, timeout: float = 5):
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(ACESTEP15_API_URL + path, timeout=timeout) as r:
            return r.status, r.read()
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return None, None


def _acestep15_api_post(path: str, payload: dict, timeout: float = 30):
    import json as _json
    import urllib.request

    body = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ACESTEP15_API_URL + path, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return _json.loads(r.read())


def _ensure_acestep15_api():
    """Launch ACE-Step 1.5's own persistent API server (keeps the model
    resident in VRAM/loaded across requests) once, lazily, and reuse it for
    every Ultra generation. Cold model load only happens on the first call —
    that's most of Ultra's ~50s/song, so keeping it warm is the real speedup."""
    import subprocess

    global _acestep15_api_proc
    status, _ = _acestep15_api_get("/health", timeout=2)
    if status == 200:
        return
    with _acestep15_api_lock:
        status, _ = _acestep15_api_get("/health", timeout=2)
        if status == 200:
            return
        if _acestep15_api_proc is None or _acestep15_api_proc.poll() is not None:
            if not ACESTEP15_PY.exists():
                raise RuntimeError("Ultra engine not installed — ACE-Step 1.5 venv missing at " + str(ACESTEP15_PY))
            env = {**os.environ, "ACESTEP_API_PORT": str(ACESTEP15_API_PORT), "ACESTEP_OFFLOAD_TO_CPU": "true"}
            _acestep15_api_proc = subprocess.Popen(
                [str(ACESTEP15_PY), "-m", "acestep.api_server"],
                cwd=str(ACESTEP15_DIR), env=env,
            )
        for _ in range(180):  # first-ever start includes model load, give it up to 3 min
            time.sleep(1)
            status, _ = _acestep15_api_get("/health", timeout=2)
            if status == 200:
                return
        raise RuntimeError("Ultra engine API server didn't come up in time")


def generate_song_acestep15(prompt: str, lyrics: str, seconds: int, out_path: Path, instrumental: bool = False):
    """Ultra engine: ACE-Step 1.5 turbo (8-step) via its own persistent API
    server (own venv, own torch version — incompatible with ACE-Step v1's
    env). Model stays loaded between requests instead of a cold subprocess
    per song, which is where nearly all of the old ~50s/song went.

    Requests 2 candidate takes and keeps the one CLAP scores closer to the
    prompt — the old CLI wrapper just grabbed whichever file glob() listed
    first (effectively random), which is why quality felt like a coin flip
    take to take even though the model itself is solid."""
    import json as _json
    import shutil
    import tempfile

    unload_ace()  # free VRAM for Ultra's own resident model
    _ensure_acestep15_api()
    resp = _acestep15_api_post("/release_task", {
        "prompt": prompt,
        "lyrics": "" if instrumental else lyrics,
        "audio_duration": seconds,
        "audio_format": "mp3",
        "thinking": True,
        "batch_size": 2,
    })
    task_id = resp["data"]["task_id"]

    for _ in range(600):  # up to 10 min
        time.sleep(1)
        result = _acestep15_api_post("/query_result", {"task_id_list": [task_id]})
        entry = result["data"][0]
        if entry["status"] == 1:
            files = _json.loads(entry["result"])
            with tempfile.TemporaryDirectory() as td:
                candidates = []
                for i, f in enumerate(files):
                    status, data = _acestep15_api_get(f["file"], timeout=60)
                    if status != 200 or not data:
                        continue
                    p = Path(td) / f"take{i}.wav"
                    p.write_bytes(data)
                    candidates.append(p)
                if not candidates:
                    raise RuntimeError("Ultra engine: failed to download generated audio")
                best = candidates[0] if len(candidates) == 1 else max(candidates, key=lambda p: judge_take(p, prompt))
                shutil.copy(best, out_path)
            return
        if entry["status"] == 2:
            raise RuntimeError("ACE-Step 1.5 failed: " + str(entry.get("result"))[:400])
    raise RuntimeError("ACE-Step 1.5 timed out waiting for generation")


STABLEAUDIO_DIR = Path(os.environ.get("STABLEAUDIO_DIR", r"C:\stableaudio"))
STABLEAUDIO_PY = STABLEAUDIO_DIR / ".venv" / "Scripts" / "python.exe"
STABLEAUDIO_API_PORT = int(os.environ.get("STABLEAUDIO_API_PORT", "8002"))
STABLEAUDIO_API_URL = f"http://127.0.0.1:{STABLEAUDIO_API_PORT}"

_stableaudio_api_proc = None
_stableaudio_api_lock = threading.Lock()


def _stableaudio_api_get(path: str, timeout: float = 5):
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(STABLEAUDIO_API_URL + path, timeout=timeout) as r:
            return r.status, r.read()
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return None, None


def _stableaudio_api_post(path: str, payload: dict, timeout: float = 300):
    import json as _json
    import urllib.request

    body = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        STABLEAUDIO_API_URL + path, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return _json.loads(r.read())


def _ensure_stableaudio_api():
    """Launch Max Test's persistent API server once, lazily, and reuse it —
    same rationale as Ultra: cold model load is ~60-100s once, then a few
    seconds per song after that. See stableaudio_server.py."""
    import subprocess

    global _stableaudio_api_proc
    status, _ = _stableaudio_api_get("/health", timeout=2)
    if status == 200:
        return
    with _stableaudio_api_lock:
        status, _ = _stableaudio_api_get("/health", timeout=2)
        if status == 200:
            return
        if _stableaudio_api_proc is None or _stableaudio_api_proc.poll() is not None:
            if not STABLEAUDIO_PY.exists():
                raise RuntimeError("Max Test engine not installed — Stable Audio 3 venv missing at " + str(STABLEAUDIO_PY))
            env = {**os.environ, "STABLEAUDIO_API_PORT": str(STABLEAUDIO_API_PORT)}
            _stableaudio_api_proc = subprocess.Popen(
                [str(STABLEAUDIO_PY), str(STABLEAUDIO_DIR / "stableaudio_server.py")],
                cwd=str(STABLEAUDIO_DIR), env=env,
            )
        for _ in range(180):
            time.sleep(1)
            status, _ = _stableaudio_api_get("/health", timeout=2)
            if status == 200:
                return
        raise RuntimeError("Max Test engine API server didn't come up in time")


def generate_song_stableaudio(prompt: str, seconds: int, out_path: Path):
    """Max Test engine: Stable Audio 3 Medium, a fast latent-diffusion
    text-to-audio model from Stability AI — via its own persistent API
    server (own venv, own torch version). Instrumental only: this model has
    no lyrics/vocal conditioning at all, unlike Basic/Pro/Ultra."""
    unload_ace()  # free VRAM for Max Test's own resident model
    _ensure_stableaudio_api()
    resp = _stableaudio_api_post("/generate", {
        "prompt": prompt,
        "seconds": seconds,
        "out_path": str(out_path),
    })
    if resp.get("status") != "ok":
        raise RuntimeError("Max Test engine failed: " + str(resp)[:400])


def _active_lora():
    """ACE_LORA env wins; else lora_active.txt (written when a fine-tune finishes)."""
    if os.environ.get("ACE_LORA"):
        return os.environ["ACE_LORA"]
    f = ROOT / "lora_active.txt"
    if f.exists():
        p = f.read_text(encoding="utf-8").strip()
        if p and Path(p).exists():
            return p
    return "none"


_clap = None


def judge_take(path: Path, text: str) -> float:
    """Director's ear: CLAP text-audio alignment — how much does this take sound
    like what was asked for, with clear melodic singing?"""
    global _clap
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    if _clap is None:
        m = ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda" if torch.cuda.is_available() else "cpu").eval()
        p = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        _clap = (m, p)
    m, p = _clap
    y, _ = librosa.load(str(path), sr=48000, mono=True, duration=30)
    ins = p(text=[f"{text}, clear melodic singing with a full band"], audios=[y],
            sampling_rate=48000, return_tensors="pt", padding=True).to(m.device)
    with torch.no_grad():
        a = torch.nn.functional.normalize(m.get_audio_features(input_features=ins.input_features), dim=-1)
        t = torch.nn.functional.normalize(m.get_text_features(input_ids=ins.input_ids, attention_mask=ins.attention_mask), dim=-1)
    return float((a @ t.T).item())


def generate_song(prompt: str, lyrics: str, seconds: int, out_path: Path, ref_path: Path = None, seed=None):
    """ACE-Step: full song with actually-sung lyrics, 48kHz. Optional reference
    track steers the sound toward it (audio2audio)."""
    import re

    # the model expects lowercase structure tags; LLMs love [VERSE 1]
    lyrics = re.sub(r"\[([^\]]+)\]", lambda m: f"[{m.group(1).lower()}]", lyrics or "")
    pipe = load_ace()
    ref = {}
    if ref_path and ref_path.exists():
        ref = {
            "audio2audio_enable": True,
            "ref_audio_input": str(ref_path),
            "ref_audio_strength": float(os.environ.get("REF_STRENGTH", "0.5")),
        }
    pipe(
        format="wav",
        audio_duration=float(seconds),
        prompt=prompt,
        lyrics=lyrics,
        infer_step=int(os.environ.get("ACE_STEPS", "120")),  # 2x default steps: cleaner detail
        # guidance 20 won the overnight sweep vs chart-track profiles (base-g20: 62.9)
        guidance_scale=float(os.environ.get("ACE_GUIDANCE", "20")),
        scheduler_type=os.environ.get("ACE_SCHEDULER", "euler"),
        guidance_scale_text=float(os.environ.get("ACE_G_TEXT", "0")),
        guidance_scale_lyric=float(os.environ.get("ACE_G_LYRIC", "0")),
        manual_seeds=[seed] if seed is not None else None,
        lora_name_or_path=_active_lora(),  # your fine-tunes plug in here
        save_path=str(out_path),
        **ref,
    )
    # drop the params-json sidecar ACE-Step writes next to the wav
    out_path.with_name(out_path.stem + "_input_params.json").unlink(missing_ok=True)


# ---------------------------------------------------------------- jobs
pool = ThreadPoolExecutor(max_workers=1)  # ponytail: one GPU, one job; queue the rest


def run_job(tid: str, prompt: str, genre: str, mode: str, custom_lyrics: str = "", ref_id: str = "",
            takes: int = 1, engine: str = "basic", seconds: int = 60):
    try:
        set_track(tid, status="generating", started=time.time())
        if engine == "max":
            mode = "instrumental"  # Stable Audio 3 has no vocal/lyrics conditioning at all
        out = AUDIO_DIR / f"{tid}.wav"
        mp3_out = AUDIO_DIR / f"{tid}.mp3"
        if custom_lyrics and mode == "vocal":
            # user brought their own lyrics — enhance the music prompt only
            music_prompt, _, song_title = enhance_and_write(prompt, genre, "instrumental")
            lyrics = custom_lyrics
        else:
            music_prompt, lyrics, song_title = enhance_and_write(prompt, genre, mode)
        if song_title:
            set_track(tid, title=song_title)
        if mode == "vocal":
            set_track(tid, lyrics=lyrics)

        if engine == "pro":
            # HeartMuLa, raw — no style-ref, no director's cut, no mix/master ("just their model")
            generate_song_heartmula(music_prompt, lyrics or "[instrumental]", seconds, out)
            write_mp3(out, mp3_out)
            out.unlink(missing_ok=True)
            set_track(tid, status="done")
            return

        if engine == "ultra":
            # ACE-Step 1.5 turbo, raw — same "just their model" philosophy as Pro
            generate_song_acestep15(music_prompt, lyrics or "", seconds, mp3_out, instrumental=(mode != "vocal"))
            set_track(tid, status="done")
            return

        if engine == "max":
            # Stable Audio 3 Medium, raw — instrumental only, this model has no vocal conditioning
            generate_song_stableaudio(music_prompt, seconds, out)
            write_mp3(out, mp3_out)
            out.unlink(missing_ok=True)
            set_track(tid, status="done")
            return

        # --- basic: ACE-Step + our mix/master chain ---
        ref_path = (REF_DIR / ref_id) if ref_id and ref_id.replace(".", "").isalnum() else None
        if mode == "vocal":
            music_prompt += ", sustained melodic singing throughout — the vocalist sings, never speaks, while the full band plays under the voice for the whole song"
        if takes <= 1:
            generate_song(music_prompt, lyrics or "[instrumental]", seconds, out, ref_path=ref_path)
        else:
            # director's cut: run the takes, keep the one that sounds most like the brief
            import random

            best, best_score = None, -1.0
            for i in range(takes):
                take = out.with_name(f"{tid}_take{i}.wav")
                generate_song(music_prompt, lyrics or "[instrumental]", seconds, take,
                              ref_path=ref_path, seed=random.randint(0, 2**31))
                s = judge_take(take, music_prompt)
                if s > best_score:
                    best, best_score = take, s
            best.replace(out)
            for i in range(takes):
                out.with_name(f"{tid}_take{i}.wav").unlink(missing_ok=True)
        if os.environ.get("MIX", "on") != "off":
            try:
                import mix

                mix.mix_track(out)  # stem-split + per-stem treatment + rebalance
            except Exception:
                pass  # ponytail: mixing is polish — never fail a finished track over it
        master_audio(out)
        # pipeline stays lossless wav; only the final file is mp3 (~10x smaller)
        write_mp3(out, mp3_out)
        out.unlink(missing_ok=True)
        set_track(tid, status="done")
    except Exception as e:  # surfaced in the UI, don't kill the worker
        set_track(tid, status="failed", error=str(e)[:500])


# ---------------------------------------------------------------- api
app = FastAPI(title="MusicForge")

from fastapi.middleware.cors import CORSMiddleware

# allows the UI to work even when index.html is opened as a file (origin "null")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

KEY_FILE = ROOT / "access_key.txt"
APP_KEY = os.environ.get("APP_KEY") or (
    KEY_FILE.read_text().strip() if KEY_FILE.exists() else ""
)
if not APP_KEY:
    APP_KEY = uuid.uuid4().hex
    KEY_FILE.write_text(APP_KEY)


@app.middleware("http")
async def require_key(request, call_next):
    from fastapi.responses import PlainTextResponse, RedirectResponse

    path = request.url.path
    # share pages + their audio are capability URLs (random ids) — public by intent;
    # direct localhost use (no Cloudflare header) needs no key
    if (
        path.startswith("/share/")
        or (path.startswith("/api/tracks/") and path.endswith("/audio"))
        or "cf-connecting-ip" not in request.headers
    ):
        return await call_next(request)
    if request.query_params.get("key") == APP_KEY:
        resp = RedirectResponse(path or "/")
        resp.set_cookie("mf_key", APP_KEY, max_age=31536000, httponly=True)
        return resp
    if request.cookies.get("mf_key") == APP_KEY:
        return await call_next(request)
    return PlainTextResponse(
        "MusicForge locked - open ?key=<access key> once (key is in access_key.txt on the PC)",
        status_code=401,
    )


class GenerateReq(BaseModel):
    prompt: str
    genre: str = "Electronic"
    mode: str = "instrumental"
    lyrics: str = ""     # optional: bring your own, sung as-is
    ref_id: str = ""     # optional: style-reference upload id
    takes: int = 1       # director's cut: generate N takes, keep the judged best
    engine: str = "basic"  # "basic" = ACE-Step + our mix/master; "pro" = HeartMuLa raw; "ultra" = ACE-Step 1.5 raw; "max" = Stable Audio 3 raw, instrumental only
    seconds: int = 60      # track length, 30-240s


def _training_active() -> bool:
    """True while a training run owns the GPU — generating would OOM both."""
    pid_file = ROOT / "training" / "train_pid.txt"
    try:
        import psutil

        return psutil.pid_exists(int(pid_file.read_text().strip()))
    except Exception:
        return False


@app.post("/api/generate")
def generate(req: GenerateReq):
    if _training_active():
        raise HTTPException(503, "The GPU is busy training a new model right now — try again when training finishes")
    prompt = req.prompt.strip()
    if not prompt or len(prompt) > 500:
        raise HTTPException(400, "Prompt required (max 500 chars)")
    if req.mode not in ("instrumental", "vocal"):
        raise HTTPException(400, "mode must be instrumental or vocal")
    if req.engine not in ("basic", "pro", "ultra", "max"):
        raise HTTPException(400, "engine must be basic, pro, ultra, or max")
    if len(req.lyrics) > 5000:
        raise HTTPException(400, "Lyrics too long (max 5000 chars)")
    seconds = max(30, min(240, req.seconds))
    tid = uuid.uuid4().hex[:12]
    with db() as c:
        c.execute(
            "INSERT INTO tracks(id,prompt,genre,mode,lyrics,status,error,created,engine,seconds) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (tid, prompt, req.genre, req.mode, None, "queued", None, time.time(), req.engine, seconds),
        )
    pool.submit(run_job, tid, prompt, req.genre, req.mode, req.lyrics.strip(), req.ref_id,
                max(1, min(3, req.takes)), req.engine, seconds)
    return {"id": tid}


@app.post("/api/reference")
async def upload_reference(file: UploadFile):
    ext = Path(file.filename or "ref.mp3").suffix.lower()
    if ext not in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
        raise HTTPException(400, "Unsupported audio format")
    rid = f"{uuid.uuid4().hex[:12]}{ext}"
    data = await file.read()
    if len(data) > 30_000_000:
        raise HTTPException(400, "Reference too large (max 30MB)")
    (REF_DIR / rid).write_bytes(data)
    return {"id": rid}


STALL_SECONDS = int(os.environ.get("STALL_SECONDS", "600"))            # basic: ~2min real, 10min = dead
PRO_STALL_SECONDS = int(os.environ.get("PRO_STALL_SECONDS", "5400"))    # pro (HeartMuLa) is much slower
ULTRA_STALL_SECONDS = int(os.environ.get("ULTRA_STALL_SECONDS", "1800"))  # ultra (ACE-Step 1.5) should be fast,
                                                                           # but give cold model-load real headroom
ENGINE_STALL = {"basic": STALL_SECONDS, "pro": PRO_STALL_SECONDS, "ultra": ULTRA_STALL_SECONDS}


@app.get("/api/tracks")
def tracks():
    now = time.time()
    stuck = "Yo — this one got stuck and was cut loose. Run it again."
    with db() as c:
        for eng, window in ENGINE_STALL.items():
            cond = "COALESCE(engine,'basic')=?" if eng == "basic" else "engine=?"
            c.execute(f"UPDATE tracks SET status='failed', error=? "
                      f"WHERE status='generating' AND started < ? AND {cond}",
                      (stuck, now - window, eng))
        # queued timeout generous, since a slow job ahead can hold the queue a while
        c.execute("UPDATE tracks SET status='failed', error=? WHERE status='queued' AND created < ?",
                  (stuck, now - 2 * max(ENGINE_STALL.values())))
        rows = c.execute("SELECT * FROM tracks ORDER BY created DESC").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/tracks/{tid}")
def track(tid: str):
    with db() as c:
        row = c.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()
    if not row:
        raise HTTPException(404)
    return dict(row)


def _audio_path(tid: str) -> Path:
    mp3 = AUDIO_DIR / f"{tid}.mp3"
    return mp3 if mp3.exists() else AUDIO_DIR / f"{tid}.wav"  # older tracks are wav


@app.get("/api/tracks/{tid}/audio")
def audio(tid: str, download: bool = False):
    path = _audio_path(tid)
    if not tid.isalnum() or not path.exists():
        raise HTTPException(404)
    if download and path.suffix != ".mp3":
        mp3 = AUDIO_DIR / f"{tid}.mp3"
        write_mp3(path, mp3)
        path = mp3
    mime = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
    kw = {"filename": f"musicforge-{tid}{path.suffix}"} if download else {}
    return FileResponse(path, media_type=mime, **kw)


@app.get("/api/tracks/{tid}/peaks")
def peaks(tid: str, n: int = 160):
    """Waveform peaks for the player — tiny JSON instead of the full wav."""
    import numpy as np
    import soundfile as sf

    path = _audio_path(tid)
    if not tid.isalnum() or not path.exists():
        raise HTTPException(404)
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration = len(data) / sr
    data = np.abs(data[: len(data) // n * n]).reshape(n, -1).max(axis=1)
    top = data.max() or 1.0
    return {"peaks": [round(float(x / top), 3) for x in data], "duration": round(duration, 1)}


@app.delete("/api/tracks/{tid}")
def delete(tid: str):
    with db() as c:
        c.execute("DELETE FROM tracks WHERE id=?", (tid,))
    if tid.isalnum():
        (AUDIO_DIR / f"{tid}.wav").unlink(missing_ok=True)
        (AUDIO_DIR / f"{tid}.mp3").unlink(missing_ok=True)
    return {"ok": True}


@app.get("/share/{tid}", response_class=HTMLResponse)
def share(tid: str):
    with db() as c:
        row = c.execute("SELECT * FROM tracks WHERE id=? AND status='done'", (tid,)).fetchone()
    if not row:
        raise HTTPException(404)
    import html
    title = html.escape(row["title"] or row["prompt"][:80])
    genre = html.escape(row["genre"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — MusicForge</title>
<style>body{{background:#0B0812;color:#FAFAF9;font-family:'DM Sans',system-ui,sans-serif;
display:grid;place-items:center;min-height:100vh;margin:0;overflow:hidden}}
body::before{{content:"";position:fixed;inset:-20%;z-index:-1;filter:blur(90px);opacity:.7;
background:radial-gradient(circle at 20% 25%,#FF3D81,transparent 45%),
radial-gradient(circle at 80% 30%,#7C3AED,transparent 45%),
radial-gradient(circle at 50% 85%,#F97316,transparent 45%)}}
.card{{background:rgba(255,255,255,.1);backdrop-filter:blur(22px) saturate(1.5);
border:1px solid rgba(255,255,255,.28);padding:2.5rem;border-radius:1.4rem;max-width:32rem;width:90%;
box-shadow:0 12px 40px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.28)}}
h1{{font-size:1.25rem;margin:0 0 .25rem}}p{{color:#FDE68A;margin:0 0 1.5rem}}
audio{{width:100%;color-scheme:dark}}</style></head>
<body><div class="card"><h1>{title}</h1><p>{genre} · MusicForge</p>
<audio controls src="/api/tracks/{tid}/audio"></audio></div></body></html>"""


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8137)
