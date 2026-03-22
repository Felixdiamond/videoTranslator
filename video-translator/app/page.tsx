"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Loader2, FolderOpen, Globe2, Sparkles, WandSparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type LanguageOption = {
  code: string;
  label: string;
  supportsMelo: boolean;
  defaultSpeakerId?: string;
};

type TtsModeOption = {
  value: "melo" | "qwen3" | "gtts";
  label: string;
};

type OptionsResponse = {
  languages: LanguageOption[];
  ttsModes: TtsModeOption[];
  qwen3ModelSizes: string[];
  meloSpeakerIdsByLanguage: Record<string, string[]>;
  defaults?: {
    ttsMode?: "melo" | "qwen3" | "gtts";
    qwen3ModelSize?: string;
    enableVoiceCloning?: boolean;
  };
};

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const DEFAULT_WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");
const WS_BASE_URL = (process.env.NEXT_PUBLIC_WS_BASE_URL || DEFAULT_WS_BASE_URL).replace(/\/$/, "");

const FALLBACK_OPTIONS: OptionsResponse = {
  languages: [
    { code: "en", label: "English (US Speaker)", supportsMelo: true, defaultSpeakerId: "EN-US" },
    { code: "es", label: "Spanish (Spain Speaker)", supportsMelo: true, defaultSpeakerId: "ES" },
    { code: "fr", label: "French (France Speaker)", supportsMelo: true, defaultSpeakerId: "FR" },
    { code: "zh", label: "Chinese (Mandarin Speaker)", supportsMelo: true, defaultSpeakerId: "ZH" },
    { code: "ja", label: "Japanese (Japan Speaker)", supportsMelo: true, defaultSpeakerId: "JP" },
    { code: "ko", label: "Korean (Korea Speaker)", supportsMelo: true, defaultSpeakerId: "KR" },
    { code: "de", label: "German", supportsMelo: false },
    { code: "pt", label: "Portuguese", supportsMelo: false },
  ],
  ttsModes: [
    { value: "melo", label: "MeloTTS" },
    { value: "qwen3", label: "Qwen3-TTS" },
    { value: "gtts", label: "gTTS" },
  ],
  qwen3ModelSizes: ["0.6B", "1.7B"],
  meloSpeakerIdsByLanguage: {
    en: ["EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-Default"],
    es: ["ES"],
    fr: ["FR"],
    zh: ["ZH"],
    ja: ["JP"],
    ko: ["KR"],
  },
  defaults: {
    ttsMode: "melo",
    qwen3ModelSize: "1.7B",
    enableVoiceCloning: true,
  },
};

function encodePathForRoute(path: string): string {
  return path.replace(/\\/g, "/").split("/").map(encodeURIComponent).join("/");
}

function buildWsUrl(params: {
  filePath: string;
  targetLanguage: string;
  ttsMode: "melo" | "qwen3" | "gtts";
  qwen3ModelSize: string;
  speakerId: string;
  enableVoiceCloning: boolean;
}): string {
  const encodedPath = encodePathForRoute(params.filePath);
  const url = new URL(`${WS_BASE_URL}/translate/${encodedPath}/${encodeURIComponent(params.targetLanguage)}`);
  url.searchParams.set("tts_mode", params.ttsMode);
  url.searchParams.set("qwen3_model_size", params.qwen3ModelSize);
  url.searchParams.set("enable_voice_cloning", String(params.enableVoiceCloning));
  if (params.speakerId.trim()) {
    url.searchParams.set("speaker_id", params.speakerId.trim());
  }
  return url.toString();
}

export default function Home() {
  const { toast } = useToast();

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoPath, setVideoPath] = useState("");

  const [options, setOptions] = useState<OptionsResponse>(FALLBACK_OPTIONS);
  const [targetLanguage, setTargetLanguage] = useState("");
  const [ttsMode, setTtsMode] = useState<"melo" | "qwen3" | "gtts">("melo");
  const [qwen3ModelSize, setQwen3ModelSize] = useState("1.7B");
  const [enableVoiceCloning, setEnableVoiceCloning] = useState(true);
  const [meloSpeakerId, setMeloSpeakerId] = useState("");

  const [isLoadingOptions, setIsLoadingOptions] = useState(true);
  const [isTranslating, setIsTranslating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Ready.");
  const [translatedVideoPath, setTranslatedVideoPath] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    const loadOptions = async () => {
      setIsLoadingOptions(true);
      try {
        const response = await fetch(`${API_BASE_URL}/options`);
        if (!response.ok) {
          throw new Error(`Options request failed (${response.status})`);
        }
        const data = (await response.json()) as OptionsResponse;
        if (!active) {
          return;
        }

        setOptions(data);
        setTargetLanguage((prev) => prev || data.languages[0]?.code || "");
        setTtsMode(data.defaults?.ttsMode || "melo");
        setQwen3ModelSize(data.defaults?.qwen3ModelSize || "1.7B");
        setEnableVoiceCloning(data.defaults?.enableVoiceCloning ?? true);
      } catch {
        if (!active) {
          return;
        }
        setOptions(FALLBACK_OPTIONS);
        setTargetLanguage((prev) => prev || FALLBACK_OPTIONS.languages[0]?.code || "");
        toast({
          title: "Backend options unavailable",
          description: "Using local fallback settings. Check that the backend is running.",
          variant: "destructive",
        });
      } finally {
        if (active) {
          setIsLoadingOptions(false);
        }
      }
    };

    loadOptions();

    return () => {
      active = false;
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [toast]);

  useEffect(() => {
    if (!targetLanguage) {
      return;
    }
    const selectedLang = options.languages.find((language) => language.code === targetLanguage);
    if (!selectedLang) {
      return;
    }

    setMeloSpeakerId((previous) => previous || selectedLang.defaultSpeakerId || "");
  }, [targetLanguage, options.languages]);

  const meloSpeakersForLanguage = useMemo(() => {
    if (!targetLanguage) {
      return [];
    }
    return options.meloSpeakerIdsByLanguage[targetLanguage] || [];
  }, [options.meloSpeakerIdsByLanguage, targetLanguage]);

  const outputDownloadUrl = useMemo(() => {
    if (!translatedVideoPath) {
      return null;
    }
    return `${API_BASE_URL}/files/${encodePathForRoute(translatedVideoPath)}`;
  }, [translatedVideoPath]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setSelectedFile(file);
    setVideoPath(file.name);
    setTranslatedVideoPath(null);
    setProgress(0);
    setStatus("Ready to translate.");
  };

  const handleTranslate = async () => {
    if (!selectedFile || !targetLanguage) {
      toast({
        title: "Missing input",
        description: "Choose a video file and target language before starting.",
        variant: "destructive",
      });
      return;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsTranslating(true);
    setProgress(4);
    setStatus("Uploading file...");
    setTranslatedVideoPath(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        const text = await uploadResponse.text();
        throw new Error(text || `Upload failed (${uploadResponse.status})`);
      }

      const uploadData = (await uploadResponse.json()) as { filePath?: string };
      if (!uploadData.filePath) {
        throw new Error("Upload completed but file path was missing in server response.");
      }

      setStatus("Connecting to translation server...");
      setProgress(12);

      const wsUrl = buildWsUrl({
        filePath: uploadData.filePath,
        targetLanguage,
        ttsMode,
        qwen3ModelSize,
        speakerId: ttsMode === "melo" ? meloSpeakerId : "",
        enableVoiceCloning: ttsMode === "qwen3" ? enableVoiceCloning : false,
      });

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      let hasError = false;
      let completed = false;

      ws.onopen = () => {
        setStatus("Connected. Translation started.");
        setProgress(18);
      };

      ws.onmessage = (event) => {
        const message = String(event.data || "");
        setStatus(message);

        if (message.startsWith("Error:")) {
          hasError = true;
          setProgress(0);
          setIsTranslating(false);
          toast({
            title: "Translation failed",
            description: message,
            variant: "destructive",
          });
          return;
        }

        if (message.includes("Translation complete. Output video:")) {
          completed = true;
          setProgress(100);
          setIsTranslating(false);
          const outputPath = message.split("Output video: ")[1]?.trim();
          if (outputPath) {
            setTranslatedVideoPath(outputPath);
          }
          toast({
            title: "Translation complete",
            description: "Your translated video is ready.",
          });
          return;
        }

        if (message.includes("Video processing in progress")) {
          setProgress((prev) => Math.max(prev, 35));
          return;
        }

        setProgress((prev) => Math.min(prev + 6, 92));
      };

      ws.onerror = () => {
        hasError = true;
        setIsTranslating(false);
        setStatus("Connection error while translating.");
        toast({
          title: "WebSocket error",
          description: "Could not keep the translation connection open.",
          variant: "destructive",
        });
      };

      ws.onclose = (closeEvent) => {
        wsRef.current = null;

        if (completed || hasError) {
          return;
        }

        setIsTranslating(false);
        if (!closeEvent.wasClean) {
          setStatus("Connection closed unexpectedly.");
          toast({
            title: "Connection interrupted",
            description: "The server connection dropped before completion.",
            variant: "destructive",
          });
        } else {
          setStatus("Translation session ended.");
        }
      };
    } catch (error) {
      setIsTranslating(false);
      setStatus("Unable to start translation.");
      toast({
        title: "Request failed",
        description: (error as Error).message,
        variant: "destructive",
      });
    }
  };

  return (
    <main className="studio-shell">
      <section className="studio-card" aria-label="Video translation controls">
        <header className="mb-8">
          <p className="text-sm tracking-[0.22em] text-[#0f766e] uppercase">Open Source Pipeline</p>
          <h1 className="mt-2 text-4xl font-bold leading-tight text-[#102a43]">Video Translator Studio</h1>
          <p className="mt-3 max-w-2xl text-sm text-[#334e68]">
            Upload a video, choose translation settings, and stream backend progress in real time.
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-5">
            <label htmlFor="video-path" className="field-label">
              Source video
            </label>
            <div className="flex gap-2">
              <Input
                id="video-path"
                type="text"
                value={videoPath}
                readOnly
                aria-describedby="video-help"
                placeholder="Select .mp4 / .mov / .mkv"
                className="h-11 bg-white/80"
              />
              <Button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                aria-label="Choose video file"
                className="h-11 bg-[#0f766e] px-4 text-white hover:bg-[#0e5f59]"
              >
                <FolderOpen className="h-5 w-5" />
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={handleFileSelect}
              />
            </div>
            <p id="video-help" className="text-xs text-[#486581]">
              Files are uploaded to the backend workspace before translation starts.
            </p>

            <label htmlFor="target-language" className="field-label">
              Target language
            </label>
            <Select value={targetLanguage} onValueChange={setTargetLanguage} disabled={isLoadingOptions || isTranslating}>
              <SelectTrigger id="target-language" className="h-11 bg-white/80">
                <SelectValue placeholder="Select target language" />
              </SelectTrigger>
              <SelectContent>
                {options.languages.map((language) => (
                  <SelectItem key={language.code} value={language.code}>
                    {language.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-5">
            <label htmlFor="tts-mode" className="field-label">
              TTS engine
            </label>
            <Select
              value={ttsMode}
              onValueChange={(value) => setTtsMode(value as "melo" | "qwen3" | "gtts")}
              disabled={isLoadingOptions || isTranslating}
            >
              <SelectTrigger id="tts-mode" className="h-11 bg-white/80">
                <SelectValue placeholder="Select TTS mode" />
              </SelectTrigger>
              <SelectContent>
                {options.ttsModes.map((mode) => (
                  <SelectItem key={mode.value} value={mode.value}>
                    {mode.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {ttsMode === "melo" && (
              <>
                <label htmlFor="melo-speaker" className="field-label">
                  Melo speaker
                </label>
                <Select
                  value={meloSpeakerId || "auto"}
                  onValueChange={(value) => setMeloSpeakerId(value === "auto" ? "" : value)}
                  disabled={isTranslating || meloSpeakersForLanguage.length === 0}
                >
                  <SelectTrigger id="melo-speaker" className="h-11 bg-white/80">
                    <SelectValue placeholder="Auto speaker" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (language default)</SelectItem>
                    {meloSpeakersForLanguage.map((speaker) => (
                      <SelectItem key={speaker} value={speaker}>
                        {speaker}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {meloSpeakersForLanguage.length === 0 && (
                  <p className="text-xs text-amber-700">
                    Melo is unavailable for this target language. Backend will fall back to gTTS.
                  </p>
                )}
              </>
            )}

            {ttsMode === "qwen3" && (
              <>
                <label htmlFor="qwen-size" className="field-label">
                  Qwen3 model size
                </label>
                <Select
                  value={qwen3ModelSize}
                  onValueChange={setQwen3ModelSize}
                  disabled={isTranslating}
                >
                  <SelectTrigger id="qwen-size" className="h-11 bg-white/80">
                    <SelectValue placeholder="Select model size" />
                  </SelectTrigger>
                  <SelectContent>
                    {options.qwen3ModelSizes.map((size) => (
                      <SelectItem key={size} value={size}>
                        {size}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <label className="mt-1 flex cursor-pointer items-center gap-3 rounded-md border border-[#bcccdc] bg-white/70 px-3 py-2 text-sm text-[#243b53]">
                  <input
                    type="checkbox"
                    checked={enableVoiceCloning}
                    onChange={(event) => setEnableVoiceCloning(event.target.checked)}
                    disabled={isTranslating}
                    className="h-4 w-4 rounded border-[#829ab1]"
                  />
                  Enable voice cloning reference extraction
                </label>
              </>
            )}
          </div>
        </div>

        <div className="mt-8 flex flex-wrap gap-3">
          <Button
            type="button"
            onClick={handleTranslate}
            disabled={isTranslating || isLoadingOptions || !selectedFile || !targetLanguage}
            className="h-11 bg-[#102a43] px-5 text-white hover:bg-[#0b1f33]"
          >
            {isTranslating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Translating
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Start translation
              </>
            )}
          </Button>

          <a
            href="https://github.com/Felixdiamond/videoTranslator"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-11 items-center gap-2 rounded-md border border-[#9fb3c8] bg-white/70 px-4 text-sm text-[#102a43] transition-colors hover:bg-white"
          >
            <Globe2 className="h-4 w-4" />
            Project repository
          </a>
        </div>

        <section className="mt-8 rounded-lg border border-[#d9e2ec] bg-white/70 p-4" aria-live="polite" aria-atomic="true">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium text-[#243b53]">
            <WandSparkles className="h-4 w-4" />
            Live status
          </div>
          <Progress value={progress} className="h-2 bg-[#d9e2ec]" />
          <p className="mt-3 text-sm text-[#334e68]" role="status">
            {status}
          </p>
        </section>

        {translatedVideoPath && (
          <section className="mt-6 rounded-lg border border-emerald-300 bg-emerald-50 p-4">
            <p className="text-sm font-semibold text-emerald-900">Translation complete</p>
            <p className="mt-2 break-all text-xs text-emerald-900/90">{translatedVideoPath}</p>
            {outputDownloadUrl && (
              <a
                href={outputDownloadUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-3 inline-flex items-center rounded-md bg-emerald-700 px-3 py-2 text-sm text-white hover:bg-emerald-600"
              >
                Download translated video
              </a>
            )}
          </section>
        )}
      </section>
    </main>
  );
}