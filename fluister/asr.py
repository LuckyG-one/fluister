"""Speech-to-text backends.

Local:
- parakeet       parakeet-mlx, Apple Silicon, streaming: transcribes while
                 you speak so text is nearly ready when the key is released
- mlx            mlx-whisper, fastest batch option on Apple Silicon (Metal)
- fasterwhisper  faster-whisper / CTranslate2, cross-platform

Cloud (cheap):
- groq           whisper-large-v3-turbo, roughly $0.04 per audio hour
- openai         OpenAI transcription endpoint
"""
import io
import os
import queue
import threading
import unicodedata

import numpy as np

SAMPLE_RATE = 16000
PARAKEET_DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
WHISPER_MLX_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
# depth=99 clamps to "all encoder layers": streaming output is then exactly
# equivalent to a batch forward pass. Measured 2026-07-15: the default
# depth=1 mangles Dutch badly, depth>=2 matches batch quality, and full
# depth costs nothing extra (~150 ms per 0.5 s chunk on M-series).
PARAKEET_STREAM_KWARGS = {"context_size": (256, 256), "depth": 99}
# The TDT decoder commits tokens on every add_audio, so tiny chunks give it
# too little lookahead and wreck accuracy. Feed at least this many samples
# per call (0.75 s); the final flush may be smaller.
PARAKEET_MIN_CHUNK_SAMPLES = 12000


def has_foreign_script(text: str) -> bool:
    """True when the transcript contains letters outside the Latin script
    (e.g. Cyrillic). Parakeet v3 detects the language itself and sometimes
    decodes short Dutch/English clips as Russian."""
    for ch in text:
        if ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN"):
            return True
    return False


class _ParakeetEngine:
    """Owns the parakeet model and the single thread it lives on.

    MLX binds model state to the thread that loaded it (using it from any
    other thread raises "There is no Stream(gpu, 0) in current thread"),
    so loading, batch jobs and streaming sessions all run as jobs on this
    one daemon thread. Only one dictation happens at a time, so jobs never
    contend.
    """

    def __init__(self, model_name: str):
        self.model = None
        self.load_error = None
        self._jobs = queue.SimpleQueue()
        self._ready = threading.Event()
        self._model_name = model_name
        threading.Thread(target=self._loop, daemon=True, name="parakeet").start()

    def wait_ready(self, timeout: float | None = None) -> bool:
        self._ready.wait(timeout)
        return self.model is not None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def submit(self, job):
        self._jobs.put(job)

    def _loop(self):
        try:
            from parakeet_mlx import from_pretrained

            self.model = from_pretrained(self._model_name)
        except Exception as exc:
            self.load_error = exc
        finally:
            self._ready.set()
        if self.model is None:
            return
        while True:
            job = self._jobs.get()
            try:
                job()
            except Exception as exc:
                print(f"fluister: parakeet job failed ({exc})")


class ASR:
    def __init__(self, cfg: dict):
        self.cfg = cfg["asr"]
        self.backend = self.cfg["backend"]
        self._model = None
        self._engine = None

    @property
    def streaming(self) -> bool:
        """True when the configured backend can transcribe while recording."""
        return self.backend == "parakeet" and not self.cfg.get("language")

    def warm(self):
        """Load local models up front so the first dictation is fast."""
        if self.backend == "parakeet":
            engine = self._ensure_engine()
            if not engine.wait_ready():
                raise engine.load_error or RuntimeError("parakeet failed to load")
        if self.backend == "fasterwhisper" and self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.cfg.get("model", "large-v3-turbo"),
                device="auto",
                compute_type=self.cfg.get("compute_type", "int8"),
            )

    def _ensure_engine(self) -> _ParakeetEngine:
        if self._engine is None:
            self._engine = _ParakeetEngine(
                self.cfg.get("model") or PARAKEET_DEFAULT_MODEL
            )
        return self._engine

    def transcribe(self, audio: np.ndarray) -> str:
        language = self.cfg.get("language") or None

        if self.backend == "parakeet" and language:
            # Parakeet cannot be forced into a language; a pinned language
            # (menu bar choice) goes through whisper instead.
            text = self.rescue(audio, [language])
            if text:
                return text
            print("fluister: whisper failed, falling back to parakeet auto-detect")

        if self.backend == "parakeet":
            try:
                return self._transcribe_parakeet_batch(audio)
            except Exception as exc:
                print(f"fluister: parakeet failed ({exc}), falling back to mlx-whisper")
                return self._transcribe_mlx(audio, language, model=WHISPER_MLX_DEFAULT_MODEL)

        if self.backend == "mlx":
            return self._transcribe_mlx(audio, language)

        if self.backend == "fasterwhisper":
            self.warm()
            segments, _info = self._model.transcribe(
                audio, language=language, vad_filter=True
            )
            return " ".join(s.text.strip() for s in segments).strip()

        if self.backend in ("groq", "openai"):
            return self._cloud(audio, language)

        raise ValueError(f"Unknown ASR backend: {self.backend}")

    def rescue(self, audio: np.ndarray, languages: list | None = None) -> str | None:
        """Re-transcribe with mlx-whisper, forced to each language in
        asr.rescue_languages, and keep the most confident result. Used when
        Parakeet returned a script the user never dictates in. Returns None
        when whisper is unavailable or fails. Runs on the engine thread
        because MLX state is bound to the thread that created it."""
        languages = languages or self.cfg.get("rescue_languages") or ["nl", "en"]
        model = self.cfg.get("rescue_model") or WHISPER_MLX_DEFAULT_MODEL
        done = threading.Event()
        out = {}

        def job():
            try:
                import mlx_whisper

                best = None
                for lang in languages:
                    result = mlx_whisper.transcribe(
                        audio, path_or_hf_repo=model, language=lang
                    )
                    text = result["text"].strip()
                    segs = result.get("segments") or []
                    score = (
                        sum(s.get("avg_logprob", -10.0) for s in segs) / len(segs)
                        if segs
                        else -10.0
                    )
                    if text and not has_foreign_script(text):
                        if best is None or score > best[0]:
                            best = (score, text)
                out["text"] = best[1] if best else None
            except Exception as exc:
                out["error"] = exc
            finally:
                done.set()

        engine = self._ensure_engine()
        if not engine.wait_ready(timeout=0.5):
            return None
        engine.submit(job)
        # First use downloads the whisper model, so allow a long wait.
        if not done.wait(timeout=300):
            return None
        if "error" in out:
            print(f"fluister: whisper rescue failed ({out['error']})")
            return None
        return out["text"]

    def _transcribe_mlx(self, audio: np.ndarray, language, model: str | None = None) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model or self.cfg.get("model", WHISPER_MLX_DEFAULT_MODEL),
            language=language,
        )
        return result["text"].strip()

    def _transcribe_parakeet_batch(self, audio: np.ndarray) -> str:
        """Non-streaming parakeet path, used when the streaming session was
        never started (model still warming) or returned nothing. Runs as a
        job on the engine thread; fail fast when the model is not ready so
        the caller's whisper fallback handles the dictation."""
        engine = self._ensure_engine()
        if not engine.wait_ready(timeout=0.5):
            raise RuntimeError("parakeet model not ready")

        done = threading.Event()
        out = {}

        def job():
            import mlx.core as mx

            try:
                with engine.model.transcribe_stream(
                    **PARAKEET_STREAM_KWARGS
                ) as transcriber:
                    transcriber.add_audio(mx.array(audio))
                    # .result must be read inside the with-block; treat
                    # post-exit access as undefined.
                    out["text"] = transcriber.result.text.strip()
            except Exception as exc:
                out["error"] = exc
            finally:
                done.set()

        engine.submit(job)
        if not done.wait(timeout=60):
            raise RuntimeError("parakeet transcription timed out")
        if "error" in out:
            raise out["error"]
        return out["text"]

    def _cloud(self, audio: np.ndarray, language) -> str:
        import soundfile as sf
        from openai import OpenAI

        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        buffer.name = "dictation.wav"

        if self.backend == "groq":
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY", ""),
            )
            model = self.cfg.get("model", "whisper-large-v3-turbo")
        else:
            client = OpenAI()
            model = self.cfg.get("model", "gpt-4o-mini-transcribe")

        kwargs = {"model": model, "file": buffer}
        if language:
            kwargs["language"] = language
        result = client.audio.transcriptions.create(**kwargs)
        return result.text.strip()


class StreamingTranscriber:
    """One streaming session for a single dictation (parakeet backend).

    The mic callback feeds chunks in; the session runs as a job on the
    parakeet engine thread so the transcript is (nearly) ready the moment
    the key is released. Never reused across recordings.

    Threading: feed() runs on the sounddevice callback thread, start() and
    abort() on the Qt main thread, finish() on the pipeline thread, _run()
    on the engine thread. The Event in finish() orders the read of _text
    after the engine thread's write, so no lock is needed.
    """

    def __init__(self, asr: ASR, on_draft=None):
        self._asr = asr
        self._on_draft = on_draft
        self._queue = queue.SimpleQueue()
        self._failed = threading.Event()
        self._done = threading.Event()
        self._text = None
        self._started = False

    def start(self) -> bool:
        """Queue the session on the engine thread. Returns False when the
        model is not warm yet (first launch, download in progress); the
        caller then simply lets the batch path handle this dictation."""
        engine = self._asr._engine
        if engine is None or not engine.ready:
            return False
        self._started = True
        engine.submit(self._run)
        return True

    def feed(self, chunk: np.ndarray):
        if not self._failed.is_set():
            self._queue.put(chunk)

    def finish(self, timeout: float = 10.0):
        """Flush, wait for the session, return the final text or None when
        streaming failed (caller falls back to batch transcription)."""
        if not self._started:
            return None
        self._queue.put(None)
        if not self._done.wait(timeout) or self._failed.is_set():
            return None
        return self._text

    def abort(self):
        """Esc-cancel or too-short recording: discard everything."""
        self._failed.set()
        self._queue.put(None)

    def _run(self):
        try:
            import mlx.core as mx

            with self._asr._engine.model.transcribe_stream(
                **PARAKEET_STREAM_KWARGS
            ) as transcriber:
                pending = []
                pending_samples = 0
                while True:
                    item = self._queue.get()
                    done = item is None
                    if not done:
                        pending.append(item)
                        pending_samples += len(item)
                    # Drain the backlog of mic blocks that queued up.
                    while True:
                        try:
                            extra = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if extra is None:
                            done = True
                            break
                        pending.append(extra)
                        pending_samples += len(extra)
                    # Accumulate before inferring: the decoder commits
                    # tokens per add_audio, and tiny chunks give it too
                    # little lookahead (badly garbled Dutch in testing).
                    if not done and pending_samples < PARAKEET_MIN_CHUNK_SAMPLES:
                        continue
                    if pending:
                        transcriber.add_audio(mx.array(np.concatenate(pending)))
                        pending = []
                        pending_samples = 0
                    if done:
                        if not self._failed.is_set():
                            self._text = transcriber.result.text.strip()
                        return
                    if self._on_draft is not None:
                        self._on_draft(transcriber.result.text)
        except Exception as exc:
            print(f"fluister: streaming ASR failed ({exc}), batch fallback takes over")
            self._failed.set()
        finally:
            self._done.set()
