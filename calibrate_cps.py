#!/usr/bin/env python3
"""
calibrate_cps.py — CPS (chars-per-second) calibration for MeloTTS

Usage:  python calibrate_cps.py

Synthesises a reference sentence per language at speed=1.0,
measures actual output duration, prints corrected CPS_MAP.
Run once after any MeloTTS update and paste the output into translator.py.
"""

import tempfile
import os

from pydub import AudioSegment
from tts_engine import TTSEngine

REFERENCE_SENTENCES = {
    "en": "Hello, I am doing well and the weather is nice today.",        # 53 chars
    "fr": "Bonjour, je vais bien et le temps est agréable aujourd'hui.",   # 59 chars
    "es": "Hola, estoy bien y el tiempo es agradable hoy.",                # 47 chars
    # DE and PT removed — MeloTTS has no model for them
    "zh": "你好，我今天很好，天气也很好。",                                        # 15 chars
    "ja": "こんにちは、今日は元気で、天気もいいです。",                               # 20 chars
    "ko": "안녕하세요, 저는 잘 지내고 있고 오늘 날씨도 좋습니다.",                    # 26 chars
}

LANGUAGE_MODEL_MAP = {
    "en": {"melo_language": "EN", "speaker_id": "EN-US"},
    "fr": {"melo_language": "FR", "speaker_id": "FR"},
    "es": {"melo_language": "ES", "speaker_id": "ES"},
    # DE and PT removed — MeloTTS has no model for them
    "zh": {"melo_language": "ZH", "speaker_id": "ZH"},
    "ja": {"melo_language": "JP", "speaker_id": "JP"},
    "ko": {"melo_language": "KR", "speaker_id": "KR"},
}


def main():
    results = {}
    for lang, sentence in REFERENCE_SENTENCES.items():
        lang_cfg = LANGUAGE_MODEL_MAP[lang]
        tts = TTSEngine(mode="melo", device="cuda", language_model_map=LANGUAGE_MODEL_MAP)
        tts.load_melo(lang_cfg["melo_language"])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        tts.synthesize(sentence, lang, out, speed=1.0, speaker_id=lang_cfg["speaker_id"])
        duration_s = len(AudioSegment.from_file(out)) / 1000.0
        cps = len(sentence) / duration_s
        results[lang] = round(cps, 1)
        os.remove(out)
        tts.unload()
        print(f"  {lang}: {len(sentence)} chars / {duration_s:.2f}s = {cps:.1f} CPS")

    print("\nPaste this into translator.py:")
    print("CPS_MAP: Dict[str, float] = {")
    for lang, cps in results.items():
        print(f'    "{lang}": {cps},')
    print("}")


if __name__ == "__main__":
    main()
