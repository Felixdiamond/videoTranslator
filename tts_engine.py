"""Unified TTS engine for MeloTTS, Qwen3-TTS, and gTTS fallback."""

import gc
import logging
from typing import Dict, Optional

import torch
from gtts import gTTS
from pydub import AudioSegment

try:
    from melo.api import TTS as MeloTTS_API
    _MELO_AVAILABLE = True
except ImportError:
    MeloTTS_API = None
    _MELO_AVAILABLE = False
    logging.warning("tts_engine: MeloTTS not installed — MeloTTS mode disabled.")

try:
    from qwen_tts import Qwen3TTSModel  # pip install -U qwen-tts
    _QWEN3_AVAILABLE = True
except ImportError:
    try:
        from qwen3_tts import Qwen3TTSModel  # legacy source install path
        _QWEN3_AVAILABLE = True
    except ImportError:
        Qwen3TTSModel = None
        _QWEN3_AVAILABLE = False
        logging.warning(
            "tts_engine: Qwen3-TTS not installed — install with `pip install -U qwen-tts` "
            "(or source install from QwenLM/Qwen3-TTS)."
        )

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    sf = None
    _SF_AVAILABLE = False

_QWEN_GEN_DEFAULTS: dict = dict(
    max_new_tokens=2048,
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
)

MELO_ALL_SPEAKERS: Dict[str, list] = {
    "en": ["EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-Default"],
    "es": ["ES"],
    "fr": ["FR"],
    "zh": ["ZH"],
    "ja": ["JP"],
    "ko": ["KR"],
}


def get_all_speaker_ids(melo_instance) -> Dict[str, int]:
    """Return speaker IDs for the active MeloTTS model."""
    if melo_instance is None:
        return {}
    try:
        return dict(melo_instance.hps.data.spk2id)
    except AttributeError:
        return {}


class TTSEngine:
    """Unified TTS interface with backend routing and lifecycle helpers."""

    def __init__(
        self,
        mode: str = "qwen3",
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
        self._qwen_variant: Optional[str] = None  # "Base" | "CustomVoice"
        self._reference_embedding = None   # pre-computed voice-clone embedding

    def load_melo(self, language: str) -> None:
        """Load or hot-reload MeloTTS for the requested language."""
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

    def load_qwen3(self, for_voice_cloning: bool = False) -> None:
        """Load Qwen3-TTS from HuggingFace Hub."""
        if not _QWEN3_AVAILABLE:
            logging.warning("TTSEngine: Qwen3-TTS unavailable — skipping load.")
            return
        variant = "Base" if for_voice_cloning else "CustomVoice"
        model_id = f"Qwen/Qwen3-TTS-12Hz-{self.model_size}-{variant}"
        device_str = str(self.device)
        if device_str.startswith("cuda"):
            bf16_supported = bool(
                torch.cuda.is_available()
                and hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            )
            dtype = torch.bfloat16 if bf16_supported else torch.float16
        else:
            dtype = torch.float32
        logging.info(
            f"TTSEngine: loading Qwen3-TTS '{model_id}' on device='{device_str}' with dtype='{dtype}'"
        )
        extra_kwargs: dict = {}
        if device_str.startswith("cuda"):
            try:
                import flash_attn  # noqa: F401
                extra_kwargs["attn_implementation"] = "flash_attention_2"
                logging.info("TTSEngine: using Flash Attention 2 for Qwen3-TTS.")
            except ImportError:
                logging.info(
                    "TTSEngine: flash-attn not installed — using default attention. "
                    "Install with: pip install flash-attn"
                )
        self._qwen = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device_str,
            dtype=dtype,
            **extra_kwargs,
        )
        self._qwen_variant = variant
        logging.info("TTSEngine: Qwen3-TTS ready.")

    def extract_voice_embedding(
        self, reference_audio_path: str, reference_text: str = ""
    ) -> bool:
        """Create and cache a voice embedding for cloning."""
        if self._qwen is None:
            logging.warning(
                "TTSEngine: skipping voice-clone embedding — "
                "Qwen3-TTS not loaded (package missing or load failed)."
            )
            return False
        logging.info(
            f"TTSEngine: extracting voice embedding from '{reference_audio_path}'"
        )
        use_icl = bool(reference_text and reference_text.strip())
        self._reference_embedding = self._qwen.create_voice_clone_prompt(
            ref_audio=reference_audio_path,
            ref_text=reference_text if use_icl else None,
            x_vector_only_mode=not use_icl,
        )
        logging.info(
            f"TTSEngine: voice-clone embedding cached "
            f"(mode={'ICL' if use_icl else 'x-vector-only'})"
        )
        return True

    def synthesize(
        self,
        text: str,
        language_code: str,
        output_path: str,
        speed: float = 1.0,
        speaker_id: Optional[str] = None,
        instruct: str = "",
    ) -> None:
        """Synthesize speech and write it to output_path."""
        if self.mode == "qwen3":
            if self._qwen is None:
                raise RuntimeError(
                    "TTSEngine: Qwen3-TTS is not loaded. "
                    "Install it with: pip install -U qwen-tts"
                )
            self._synthesize_qwen3(text, language_code, output_path, speaker_id, instruct, speed)
        elif self.mode == "melo" and self._melo is not None:
            self._synthesize_melo(text, language_code, output_path, speed, speaker_id)
        else:
            self._synthesize_gtts(text, language_code, output_path)

    def _normalize_qwen_language(self, language_code: str) -> str:
        """Map short ISO-like language code to Qwen expected language names."""
        lang_map = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "de": "German",
            "fr": "French",
            "ru": "Russian",
            "pt": "Portuguese",
            "es": "Spanish",
            "it": "Italian",
        }
        if not language_code:
            return "Auto"
        return lang_map.get(language_code.lower(), "Auto")

    def _synthesize_qwen3(
        self,
        text: str,
        language_code: str,
        output_path: str,
        speaker_id: Optional[str],
        instruct: str,
        speed: float = 1.0,
    ) -> None:
        qwen_language = self._normalize_qwen_language(language_code)

        if speed >= 1.3:
            pace_instruct = "speak quickly and clearly"
        elif speed <= 0.8:
            pace_instruct = "speak slowly and clearly"
        else:
            pace_instruct = "speak at a natural pace"

        combined_instruct = f"{pace_instruct}. {instruct}".strip(". ") if instruct else pace_instruct

        kwargs = dict(text=text, language=qwen_language)
        try:
            if self._reference_embedding is not None:
                if pace_instruct != "speak at a natural pace":
                    logging.info(
                        "[QWEN3_TTS] Pace instruction ignored for voice-clone (style "
                        "is set by reference audio, not instruct text)."
                    )
                wavs, sr = self._qwen.generate_voice_clone(
                    voice_clone_prompt=self._reference_embedding,
                    **_QWEN_GEN_DEFAULTS,
                    **kwargs,
                )
            else:
                if self._qwen_variant == "Base":
                    raise RuntimeError(
                        "Qwen Base model loaded but no voice clone prompt is available. "
                        "Enable/repair reference extraction or disable voice cloning."
                    )
                spk = speaker_id or "Ryan"
                wavs, sr = self._qwen.generate_custom_voice(
                    speaker=spk,
                    instruct=combined_instruct,
                    **_QWEN_GEN_DEFAULTS,
                    **kwargs,
                )
            if _SF_AVAILABLE:
                sf.write(output_path, wavs[0], sr)
            else:
                import scipy.io.wavfile as wavfile
                wavfile.write(output_path, sr, wavs[0])
        except Exception as e:
            logging.error(f"TTSEngine: Qwen3 synthesis failed: {e}", exc_info=True)
            raise

    def _synthesize_melo(
        self,
        text: str,
        language_code: str,
        output_path: str,
        speed: float,
        speaker_id: Optional[str],
    ) -> None:
        spk2id = self._melo.hps.data.spk2id
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

    def get_speaker_ids(self) -> Dict[str, int]:
        """Return the active MeloTTS speaker-ID map."""
        return get_all_speaker_ids(self._melo)

    def unload(self) -> None:
        """Unload all models and clear caches."""
        if self._melo is not None:
            del self._melo
            self._melo = None
        if self._qwen is not None:
            del self._qwen
            self._qwen = None
        self._qwen_variant = None
        self._reference_embedding = None
        self._current_melo_lang = None
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("TTSEngine: all models unloaded.")

    def __enter__(self) -> "TTSEngine":
        return self

    def __exit__(self, *args) -> bool:
        self.unload()
        return False
