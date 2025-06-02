"use client";

import React, { useState, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Loader2, FolderOpen } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Updated LANGUAGE_MODEL_MAP to match translator.py (keys are important)
// Values are display names for the dropdown.
const LANGUAGE_MODEL_MAP = {
  en: "English (US Speaker)",
  es: "Spanish (Spain Speaker)",
  fr: "French (France Speaker)",
  zh: "Chinese (Mandarin Speaker)",
  ja: "Japanese (Japan Speaker)", // Changed from jp
  ko: "Korean (Korea Speaker)",   // Changed from kr
  de: "German (Female Speaker)",
  pt: "Portuguese (Female Speaker)",
  // Add other languages from translator.py's map if they are intended for frontend selection
};

export default function Home() {
  const [videoPath, setVideoPath] = useState(""); // Stores the display name of the file
  const [selectedFile, setSelectedFile] = useState<File | null>(null); // Type annotation for File
  const [targetLanguage, setTargetLanguage] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);
  const [progress, setProgress] = useState(0); // Progress might be less granular now
  const [status, setStatus] = useState("");
  const [translatedVideoUrl, setTranslatedVideoUrl] = useState<string | null>(null); // Type annotation
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement | null>(null); // Type for ref

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => { // Type for event
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setVideoPath(file.name); // Display name
      setTranslatedVideoUrl(null); // Reset previous result
      setProgress(0);
      setStatus("");
    }
  };

  const handleTranslate = async () => {
    if (!selectedFile || !targetLanguage) {
      toast({
        title: "Error",
        description: "Please select a video file and target language.",
        variant: "destructive",
      });
      return;
    }

    setIsTranslating(true);
    setProgress(0);
    setStatus("Uploading and translating...");

    try {
      // Upload the file
      const formData = new FormData();
      formData.append('file', selectedFile);

      const uploadResponse = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      const uploadData = await uploadResponse.json();
      const filePath = uploadData.filePath; // This should be relative path e.g., "uploaded_files/video.mp4"

      // The WebSocket URL now expects the video_path to be part of the URL path.
      // FastAPI automatically decodes URL parameters, so no need for double encoding if filePath is simple.
      // However, if filePath can contain special characters like '/' (which it will),
      // it needs to be properly encoded for the URL path segment.
      // The server-side uses `:path` converter for `video_path`, which handles slashes.
      // Normalize path separators to forward slashes before splitting and encoding
      const pathSegments = filePath.replace(/\\/g, '/').split('/');
      const encodedFilePath = pathSegments.map(encodeURIComponent).join('/');

      const ws = new WebSocket(
        `ws://localhost:8000/translate/${encodedFilePath}/${targetLanguage}`
      );

      ws.onopen = () => {
        setStatus("Connection established. Starting translation...");
        setProgress(10); // Initial progress
      };

      ws.onmessage = (event) => {
        const message = event.data as string; // Type assertion
        setStatus(message);

        // Simplified progress based on backend messages
        if (message.startsWith("Error:")) {
          setProgress(0); // Reset progress on error
          toast({
            title: "Translation Error",
            description: message,
            variant: "destructive",
          });
          setIsTranslating(false); // Stop loading state on error
        } else if (message.includes("Video processing in progress...")) {
          setProgress(30); // General progress update
        } else if (message.includes("Translation complete. Output video:")) {
          setProgress(100);
          const outputPath = message.split("Output video: ")[1];
          setTranslatedVideoUrl(outputPath); // This path is relative to project root
          toast({ // Success toast moved here from onclose for clarity
            title: "Translation Successful!",
            description: `Video translated. Output: ${outputPath}`,
          });
        } else {
          // For other messages, you might increment progress or just display status
           if (progress < 90) setProgress((prev: number) => Math.min(prev + 5, 90)); // Gradual progress for other messages
        }
      };

      ws.onclose = (event) => {
        // Only set isTranslating to false if it wasn't an error case that already did it
        if (!event.wasClean && !status.startsWith("Error:")) {
            // If connection closed uncleanly and not due to a reported error
            setStatus("Connection closed unexpectedly.");
            toast({
                title: "Connection Issue",
                description: "The connection to the server was lost.",
                variant: "destructive",
            });
        } else if (event.wasClean && progress !== 100 && !status.startsWith("Error:")) {
            // If connection closed cleanly but process didn't complete fully (e.g. server closed it early)
             setStatus("Translation process ended.");
        }
        // If translation was successful, isTranslating is already false from onmessage.
        // If there was an error, isTranslating is also set to false.
        // This ensures the button re-enables correctly.
        if (progress !== 100) { // If not completed successfully
            setIsTranslating(false);
        }
      };

      ws.onerror = (errorEvent) => { // errorEvent is of type Event
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
        description: (error as Error).message || "An error occurred during translation.", // Type assertion for error
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
                readOnly
                className="flex-grow mr-2"
              />
              <Button
                onClick={() => fileInputRef.current?.click()} // Optional chaining for ref
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