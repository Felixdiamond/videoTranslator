"""
tts_engine.py — Unified TTS interface (Phase 2, Issues 4 & 9)

Supports:
  - MeloTTS    (fast, preset voices, multi-speaker)
  - Qwen3-TTS  (voice cloning, high-quality, 10 languages)
  - gTTS       (network-based fallback; no local GPU needed)

Usage (context-manager — auto-unloads VRAM):

    with TTSEngine(mode="qwen3", device="cuda", model_size="1.7B") as tts:
        tts.extract_voice_embedding("voice_reference.wav", ref_text)
        tts.synthesize("Hello", "en", "/tmp/out.wav")

Usage (manual lifecycle):

    tts = TTSEngine(mode="melo", device="cuda")
    tts.load_melo("EN")
    tts.synthesize("Hello", "en", "/tmp/out.wav")
    tts.unload()
"""

import gc
import logging
import os
from typing import Dict, Optional

import torch
from gtts import gTTS
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Optional heavy-dependency imports — fail gracefully so the rest of the
# system can run even when a specific TTS backend isn't installed.
# ---------------------------------------------------------------------------
try:
    from melo.api import TTS as MeloTTS_API
    _MELO_AVAILABLE = True
except ImportError:
    MeloTTS_API = None
    _MELO_AVAILABLE = False
    logging.warning("tts_engine: MeloTTS not installed — MeloTTS mode disabled.")

try:
    from qwen3_tts import Qwen3TTSModel          # pip install git+https://github.com/QwenLM/Qwen3-TTS.git
    _QWEN3_AVAILABLE = True
except ImportError:
    Qwen3TTSModel = None
    _QWEN3_AVAILABLE = False
    logging.warning("tts_engine: Qwen3-TTS not installed — Qwen3 mode disabled.")

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    sf = None
    _SF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Speaker registry (Issue 4a)
# ---------------------------------------------------------------------------

#: All known MeloTTS speaker IDs per language code.
MELO_ALL_SPEAKERS: Dict[str, list] = {
    "en": ["EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-Default"],
    "es": ["ES"],
    "fr": ["FR"],
    "zh": ["ZH"],
    "ja": ["JP"],
    "ko": ["KR"],
    "de": ["DE"],
    "pt": ["PT"],
}


def get_all_speaker_ids(melo_instance) -> Dict[str, int]:
    """Return all speaker IDs available in a loaded MeloTTS instance.

    Args:
        melo_instance: A loaded ``MeloTTS_API`` object, or ``None``.

    Returns:
        ``dict`` mapping speaker-id string → integer index,
        e.g. ``{"EN-US": 0, "EN-BR": 1, ...}``.
        Returns an empty dict when *melo_instance* is ``None``.
    """
    if melo_instance is None:
        return {}
    try:
        return dict(melo_instance.hps.data.spk2id)
    except AttributeError:
        return {}


# ---------------------------------------------------------------------------
# TTSEngine (Issues 4b & 9)
# ---------------------------------------------------------------------------

class TTSEngine:
    """Unified TTS interface supporting MeloTTS and Qwen3-TTS with gTTS fallback.

    Parameters
    ----------
    mode : str
        Primary synthesis backend: ``"melo"`` | ``"qwen3"`` | ``"gtts"``.
    device : str
        PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
    model_size : str
        Qwen3-TTS model size when ``mode="qwen3"``: ``"0.6B"`` or ``"1.7B"``.
    language_model_map : dict, optional
        The ``LANGUAGE_MODEL_MAP`` dict from ``translator.py``, used to look up
        default MeloTTS speaker IDs and gTTS language codes.
    """

    def __init__(
        self,
        mode: str = "melo",
        device: str = "cpu",
        model_size: str = "1.7B",
        language_model_map: Optional[Dict] = None,
    ) -> None:
        self.mode = mode
        self.device = device
        self.model_size = model_size
        self._language_model_map: Dict = language_model_map or {}

        self._melo = None
        self._current_melo_lang: Optional[str] = None
        self._qwen = None
        self._reference_embedding = None   # pre-computed voice-clone embedding

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_melo(self, language: str) -> None:
        """Load (or hot-reload) MeloTTS for *language* (e.g. ``"EN"``, ``"FR"``).

        No-op when the requested language is already loaded. Re-loads and frees
        the old model when the language changes (Issue 9 — prevents wrong-language
        phonemes when target language changes between calls).
        """
        if not _MELO_AVAILABLE:
            logging.warning("TTSEngine: MeloTTS unavailable — skipping load.")
            return
        if self._melo is not None and self._current_melo_lang == language:
            return  # already correct
        if self._melo is not None:
            logging.info(
                f"TTSEngine: reloading MeloTTS {self._current_melo_lang!r} → {language!r}"
            )
            del self._melo
            torch.cuda.empty_cache()
            gc.collect()
        logging.info(f"TTSEngine: loading MeloTTS language='{language}' device='{self.device}'")
        self._melo = MeloTTS_API(language=language, device=self.device)
        self._current_melo_lang = language
        logging.info(
            f"TTSEngine: MeloTTS ready. Speakers: {list(self._melo.hps.data.spk2id.keys())}"
        )

    def load_qwen3(self) -> None:
        """Load Qwen3-TTS from HuggingFace Hub."""
        if not _QWEN3_AVAILABLE:
            logging.warning("TTSEngine: Qwen3-TTS unavailable — skipping load.")
            return
        model_id = f"Qwen/Qwen3-TTS-12Hz-{self.model_size}-CustomVoice"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        logging.info(f"TTSEngine: loading Qwen3-TTS '{model_id}'")
        self._qwen = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=self.device,
            dtype=dtype,
        )
        logging.info("TTSEngine: Qwen3-TTS ready.")

    # ------------------------------------------------------------------
    # Voice cloning (Issue 4c helper)
    # ------------------------------------------------------------------

    def extract_voice_embedding(
        self, reference_audio_path: str, reference_text: str = ""
    ) -> None:
        """Pre-compute a speaker embedding from a 3–10 s reference clip.

        The embedding is cached in ``self._reference_embedding`` and reused
        by :meth:`synthesize` for every segment.

        Args:
            reference_audio_path: Path to a clean WAV clip of the speaker.
            reference_text: Transcript of the reference clip (improves accuracy).

        Raises:
            RuntimeError: If Qwen3-TTS is not yet loaded.
        """
        if self._qwen is None:
            raise RuntimeError(
                "TTSEngine: Qwen3-TTS not loaded — call load_qwen3() first."
            )
        logging.info(
            f"TTSEngine: extracting voice embedding from '{reference_audio_path}'"
        )
        self._reference_embedding = self._qwen.create_voice_clone_prompt(
            ref_audio=reference_audio_path,
            ref_text=reference_text,
            x_vector_only_mode=False,   # full prompt embedding for best quality
        )
        logging.info("TTSEngine: voice-clone embedding cached.")

    # ------------------------------------------------------------------
    # Synthesis (unified interface)
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        language_code: str,
        output_path: str,
        speed: float = 1.0,
        speaker_id: Optional[str] = None,
        instruct: str = "",
    ) -> None:
        """Synthesise *text* and write audio to *output_path*.

        Routing logic:
        - ``mode='qwen3'`` with loaded model → voice-clone (if embedding cached)
          or preset speaker.
        - ``mode='melo'`` with loaded model → MeloTTS using *speaker_id*
          (auto-selects from config if *speaker_id* is ``None``).
        - ``mode='gtts'`` or no model loaded → gTTS network fallback.
        - Any backend failure → gTTS fallback.

        Args:
            text: Text to synthesise.
            language_code: Short language code (``"en"``, ``"fr"`` …).
            output_path: Destination WAV file path.
            speed: TTS speed multiplier (MeloTTS only; ignored by Qwen3/gTTS).
            speaker_id: Explicit MeloTTS speaker ID (e.g. ``"EN-BR"``).
            instruct: Natural-language style instruction for Qwen3-TTS
                      (e.g. ``"speak excitedly"``).
        """
        if self.mode == "qwen3" and self._qwen is not None:
            self._synthesize_qwen3(text, language_code, output_path, speaker_id, instruct)
        elif self.mode == "melo" and self._melo is not None:
            self._synthesize_melo(text, language_code, output_path, speed, speaker_id)
        else:
            self._synthesize_gtts(text, language_code, output_path)

    # ------------------------------------------------------------------
    # Backend helpers (private)
    # ------------------------------------------------------------------

    def _synthesize_qwen3(
        self,
        text: str,
        language_code: str,
        output_path: str,
        speaker_id: Optional[str],
        instruct: str,
    ) -> None:
        kwargs = dict(text=[text], language=[language_code.capitalize()])
        try:
            if self._reference_embedding is not None:
                wavs, sr = self._qwen.generate_voice_clone(
                    voice_clone_prompt=self._reference_embedding,
                    **kwargs,
                )
            else:
                spk = speaker_id or "Ryan"
                wavs, sr = self._qwen.generate_custom_voice(
                    speaker=[spk],
                    instruct=[instruct],
                    **kwargs,
                )
            if _SF_AVAILABLE:
                sf.write(output_path, wavs[0], sr)
            else:
                import scipy.io.wavfile as wavfile
                wavfile.write(output_path, sr, wavs[0])
        except Exception as e:
            logging.error(
                f"TTSEngine: Qwen3 synthesis failed: {e}. Falling back to gTTS.",
                exc_info=True,
            )
            self._synthesize_gtts(text, language_code, output_path)

    def _synthesize_melo(
        self,
        text: str,
        language_code: str,
        output_path: str,
        speed: float,
        speaker_id: Optional[str],
    ) -> None:
        spk2id = self._melo.hps.data.spk2id
        # Determine speaker integer ID
        if speaker_id and speaker_id in spk2id:
            sid = spk2id[speaker_id]
        else:
            lang_cfg = self._language_model_map.get(language_code, {})
            default_spk = lang_cfg.get("speaker_id")
            if default_spk and default_spk in spk2id:
                sid = spk2id[default_spk]
            else:
                sid = list(spk2id.values())[0]
        try:
            self._melo.tts_to_file(text, sid, output_path, speed=speed)
        except Exception as e:
            logging.error(
                f"TTSEngine: MeloTTS synthesis failed: {e}. Falling back to gTTS.",
                exc_info=True,
            )
            self._synthesize_gtts(text, language_code, output_path)

    def _synthesize_gtts(
        self, text: str, language_code: str, output_path: str
    ) -> None:
        lang_cfg = self._language_model_map.get(language_code, {})
        gtts_lang = lang_cfg.get("gtts_lang", "en")
        try:
            gTTS(text=text, lang=gtts_lang, slow=False).save(output_path)
        except Exception as e:
            logging.error(
                f"TTSEngine: gTTS synthesis failed: {e}. Writing silent placeholder.",
                exc_info=True,
            )
            AudioSegment.silent(duration=100).export(output_path, format="wav")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_speaker_ids(self) -> Dict[str, int]:
        """Return the speaker-ID map from the currently loaded MeloTTS instance."""
        return get_all_speaker_ids(self._melo)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Delete all loaded models and release VRAM / RAM."""
        if self._melo is not None:
            del self._melo
            self._melo = None
        if self._qwen is not None:
            del self._qwen
            self._qwen = None
        self._reference_embedding = None
        self._current_melo_lang = None
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("TTSEngine: all models unloaded.")

    def __enter__(self) -> "TTSEngine":
        return self

    def __exit__(self, *args) -> bool:
        self.unload()
        return False   # do not suppress exceptions
