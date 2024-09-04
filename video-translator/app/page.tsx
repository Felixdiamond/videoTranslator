"use client";

import React, { useState, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Loader2, Upload, FolderOpen } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const LANGUAGE_MODEL_MAP = {
  en: "English",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  pl: "Polish",
  tr: "Turkish",
  ru: "Russian",
  nl: "Dutch",
};

export default function Home() {
  const [videoPath, setVideoPath] = useState("");
  const [targetLanguage, setTargetLanguage] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [translatedVideoUrl, setTranslatedVideoUrl] = useState(null);
  const { toast } = useToast();
  const fileInputRef = useRef(null);

  const handlePathChange = (event) => {
    setVideoPath(event.target.value);
  };

  const handleFileBrowse = () => {
    fileInputRef.current.click();
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideoPath(file.path);
    }
  };

  const handleTranslate = async () => {
    if (!videoPath || !targetLanguage) {
      toast({
        title: "Error",
        description: "Please select a video file and target language.",
        variant: "destructive",
      });
      return;
    }

    setIsTranslating(true);
    setProgress(0);
    setStatus("Initiating translation...");

    try {
      const ws = new WebSocket(
        `ws://localhost:8000/translate/${encodeURIComponent(
          videoPath
        )}/${targetLanguage}`
      );

      ws.onmessage = (event) => {
        const message = event.data;
        setStatus(message);
        if (message.includes("Audio extracted")) setProgress(20);
        else if (message.includes("Text translated")) setProgress(40);
        else if (message.includes("Synced audio created")) setProgress(60);
        else if (message.includes("Sound effects preserved")) setProgress(80);
        else if (message.includes("Translation complete")) {
          setProgress(100);
          const outputPath = message.split("Output video: ")[1];
          setTranslatedVideoUrl(outputPath);
        }
      };

      ws.onclose = () => {
        setIsTranslating(false);
        toast({
          title: "Translation complete",
          description: "Your video has been successfully translated.",
        });
      };

      ws.onerror = (error) => {
        setIsTranslating(false);
        setStatus("Error occurred");
        toast({
          title: "Error",
          description: "An error occurred during translation.",
          variant: "destructive",
        });
      };
    } catch (error) {
      setIsTranslating(false);
      setStatus("Error occurred");
      toast({
        title: "Error",
        description: error.message || "An error occurred during translation.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-slate-50 to-slate-100">
      <div className="w-full max-w-md p-8 bg-white rounded-lg shadow-lg">
        <h1 className="text-3xl font-bold mb-6 text-slate-900">
          Video Translator
        </h1>
        <div className="space-y-6">
          <div>
            <label
              htmlFor="video-path"
              className="block text-sm font-medium text-slate-700 mb-2"
            >
              Video File
            </label>
            <div className="flex">
              <Input
                id="video-path"
                type="text"
                placeholder="/path/to/your/video.mp4"
                value={videoPath}
                onChange={handlePathChange}
                className="flex-grow mr-2"
                readOnly
              />
              <Button
                onClick={handleFileBrowse}
                className="bg-slate-200 text-slate-700 hover:bg-slate-300"
              >
                <FolderOpen className="w-5 h-5" />
              </Button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept="video/*"
                className="hidden"
              />
            </div>
          </div>
          <div>
            <label
              htmlFor="target-language"
              className="block text-sm font-medium text-slate-700 mb-2"
            >
              Target Language
            </label>
            <Select onValueChange={setTargetLanguage}>
              <SelectTrigger id="target-language" className="w-full">
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(LANGUAGE_MODEL_MAP).map(([code, language]) => (
                  <SelectItem key={code} value={code}>
                    {language}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button
            onClick={handleTranslate}
            disabled={isTranslating}
            className="w-full bg-slate-800 text-white hover:bg-slate-700 transition-colors duration-200"
          >
            {isTranslating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Translating...
              </>
            ) : (
              "Translate"
            )}
          </Button>
        </div>
        {isTranslating && (
          <div className="mt-8 space-y-4">
            <Progress value={progress} className="w-full h-2" />
            <p className="text-center text-sm text-slate-600">{status}</p>
          </div>
        )}
        {translatedVideoUrl && (
          <div className="mt-8">
            <p className="text-sm text-slate-600 mb-2">
              Translated video saved at:
            </p>
            <p className="text-sm font-medium text-slate-900 break-all">
              {translatedVideoUrl}
            </p>
          </div>
        )}
      </div>
      <span className="text-center absolute bottom-3">
        Made with ❤️ by{" "}
        <a
          href="https://github.com/Felixdiamond"
          target="_blank"
          rel="noopener noreferrer"
          className="text-slate-800 underline"
        >
          Felix
        </a>
      </span>
    </div>
  );
}
