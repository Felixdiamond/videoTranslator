const asyncHandler = require("express-async-handler");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { createClient } = require("@deepgram/sdk");
const ffmpeg = require("fluent-ffmpeg");
const { translate } = require("@vitalets/google-translate-api");
const { spawn, exec } = require("child_process");
const dotenv = require("dotenv");
dotenv.config();

// Configure multer to preserve file extensions
const storage = multer.diskStorage({
  destination: "temp/uploads/",
  filename: function (req, file, cb) {
    cb(
      null,
      file.fieldname + "-" + Date.now() + path.extname(file.originalname)
    );
  },
});

const upload = multer({ storage: storage });

const deepgram = createClient(process.env.DEEPGRAM_API_KEY);

// @desc translate a file
// @route POST /api/translate
// @access public
const entryFunction = [
  upload.single("file"),
  asyncHandler(async (req, res, next) => {
    const filename = req.file.path;
    // Check if video or audio
    let fileExtension = path.extname(filename);
    const data = fs.readFileSync(filename);
    if (fileExtension === ".mp4" || fileExtension === ".webm") {
      return handleVideo(req, res, next, data);
    } else {
      return handleAudio(req, res, next, data);
    }
  }),
];

const getVideoDuration = async (videoFilePath) => {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(videoFilePath, (err, metadata) => {
      if (err) {
        reject(err);
      } else {
        const duration = metadata.format.duration;
        resolve(duration);
      }
    });
  });
};

const getAudioBuffer = async (response) => {
  const reader = response.getReader();
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
  }

  const dataArray = chunks.reduce(
    (acc, chunk) => Uint8Array.from([...acc, ...chunk]),
    new Uint8Array(0)
  );

  return Buffer.from(dataArray.buffer);
};

const audioToText = asyncHandler(async (data) => {
  const { result, error } = await deepgram.listen.prerecorded.transcribeFile(
    data,
    {
      model: "nova-2",
      smart_format: true,
    }
  );
  if (error) throw error;
  // return data.results.channels[0].alternatives[0].transcript;
  return result.results.channels[0].alternatives[0].transcript;
});

const splitAudioVideo = async (data) => {
  return new Promise((resolve, reject) => {
    try {
      // Create a temporary file to write the input data
      const tempDir = path.join(__dirname, "../temp");
      if (!fs.existsSync(tempDir)) {
        fs.promises.mkdir(tempDir, { recursive: true });
      }
      const inputFilePath = path.join(tempDir, "input.mp4");
      fs.promises.writeFile(inputFilePath, data);

      // Run the Python script to split audio and video
      const pythonScriptPath = path.join(__dirname, "../scripts/videotowav.py");
      const pythonProcess = spawn("python", [pythonScriptPath, inputFilePath], {
        stdio: ["inherit", "pipe", "pipe"],
      });

      pythonProcess.stdout.on("data", (data) => {
        console.log(`Python stdout: ${data}`);
      });

      pythonProcess.stderr.on("data", (data) => {
        console.error(`Python stderr: ${data}`);
      });

      const handleError = (err) => {
        console.error("Python script error:", err);
        pythonProcess.kill();
        reject(new Error("Error running Python script"));
      };

      pythonProcess.on("error", handleError);

      pythonProcess.on("exit", async (code) => {
        if (code === 0) {
          console.log("Audio and video streams split successfully");
          const audioFilePath = path.join(tempDir, "audio.wav");
          const videoFilePath = path.join(tempDir, "video.mp4");

          let audioBuffer = await fs.promises.readFile(audioFilePath);
          console.log(`Audio file size: ${audioBuffer.length} bytes`);
          let videoBuffer = await fs.promises.readFile(videoFilePath);
          console.log(`Video file size: ${videoBuffer.length} bytes`);

          if (!audioBuffer || !videoBuffer) {
            console.error("Error: audioBuffer or videoBuffer is undefined");
            reject(new Error("Error: audioBuffer or videoBuffer is undefined"));
          } else {
            const result = [audioBuffer, videoBuffer]; // Create an array instead of an object
            console.log("Result:", result);
            resolve(result);
          }

          // Clean up the temporary files
          // await Promise.all([
          //   fs.promises.unlink(inputFilePath),
          //   fs.promises.unlink(audioFilePath),
          //   fs.promises.unlink(videoFilePath),
          // ]);
        } else {
          console.error(`Python script exited with code ${code}`);
          reject(new Error(`Python script exited with code ${code}`));
        }
      });
    } catch (error) {
      console.error("Error splitting audio and video:", error);
      reject(error);
    }
  });
};

const mergeAudioVideo = async (videoBuffer, audioFilePath) => {
  return new Promise((resolve, reject) => {
    try {
      // Create temporary files for the video and audio
      const tempDir = path.join(__dirname, "../temp");
      if (!fs.existsSync(tempDir)) {
        fs.promises.mkdir(tempDir, { recursive: true });
      }
      const videoFilePath = path.join(tempDir, "temp-video.mp4");
      const outputFilePath = path.join(tempDir, "merged-output.mp4");
      fs.promises.writeFile(videoFilePath, videoBuffer);

      console.log("Video file path:", videoFilePath);
      console.log("Audio file path:", audioFilePath);
      console.log("Output file path:", outputFilePath);

      // Run the Python script to merge audio and video
      const pythonScriptPath = path.join(
        __dirname,
        "../scripts/merge_audio_video.py"
      );
      const pythonProcess = spawn(
        "python",
        [pythonScriptPath, videoFilePath, audioFilePath, outputFilePath],
        {
          stdio: ["inherit", "pipe", "pipe"],
        }
      );

      pythonProcess.stdout.on("data", (data) => {
        console.log(`Python stdout: ${data}`);
      });

      pythonProcess.stderr.on("data", (data) => {
        console.error(`Python stderr: ${data}`);
      });

      const handleError = (err) => {
        console.error("Python script error:", err);
        pythonProcess.kill();
        reject(new Error("Error running Python script"));
      };

      pythonProcess.on("error", handleError);

      pythonProcess.on("exit", async (code) => {
        if (code === 0) {
          console.log("Audio and video merged successfully");
          resolve(outputFilePath);

          // Clean up the temporary files
          // await Promise.all([
          //   fs.promises.unlink(videoFilePath),
          // ]);
        } else {
          console.error(`Python script exited with code ${code}`);
          reject(new Error(`Python script exited with code ${code}`));
        }
      });
    } catch (error) {
      console.error("Error merging audio and video:", error);
      reject(error);
    }
  });
};

const PYTHON_EXECUTABLE = path.join(
  "..",
  "backend",
  "sandbox",
  "Scripts",
  "python.exe"
);
console.log("Python executable path:", path.resolve(PYTHON_EXECUTABLE));

const textToSpeech = asyncHandler(async (text, language, videoDuration, isAudio) => {
  try {
    const pythonScriptPath = path.join(__dirname, "..", "scripts", "tts.py");
    const tempDir = path.join(__dirname, "..", "temp");
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir);
    }
    const outputFilePath = path.join(tempDir, "speech.wav");
    const argsFilePath = path.join(tempDir, "args.json");

    fs.writeFileSync(
      argsFilePath,
      JSON.stringify({ text, language, output_path: outputFilePath, video_duration: videoDuration, isAudio: isAudio })
    );

    const pythonProcess = spawn(PYTHON_EXECUTABLE, [pythonScriptPath], {
      stdio: ["inherit", "pipe", "pipe"],
    });

    pythonProcess.stdout.on("data", (data) => {
      console.log(`Python stdout: ${data}`);
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error(`Python stderr: ${data}`);
    });

    const handleError = (err) => {
      console.error("Python script error:", err);
      pythonProcess.kill();
      throw new Error("Error running Python script");
    };

    pythonProcess.on("error", handleError);

    return new Promise((resolve, reject) => {
      pythonProcess.on("exit", (code) => {
        if (code === 0) {
          resolve(outputFilePath);
        } else {
          reject(new Error(`Python script exited with code ${code}`));
        }
      });
    });
  } catch (error) {
    console.error("Error in textToSpeech:", error);
    throw error;
  }
});

const translateText = async (text, toLang) => {
  try {
    const res = await translate(text, { to: toLang });
    return res.text;
  } catch (error) {
    console.error("Error translating text:", error);
    throw error;
  }
};

const handleVideo = asyncHandler(async (req, res, next, data) => {
  console.log("Video file detected");

  try {
    // Split audio and video
    console.log("Splitting audio and video...");
    // const [audioBuffer, videoBuffer] = await splitAudioVideo(data);
    // console.log("Audio and video split");

    // // Get the duration of the original video
    // const videoFilePath = path.join(__dirname, "../temp/temp-video.mp4");
    // await fs.promises.writeFile(videoFilePath, videoBuffer);
    // const videoDuration = await getVideoDuration(videoFilePath);

    // Process audio
    console.log("Transcribing audio to text...");
    let audioFilePath = path.join(__dirname, "../temp/audio.wav");
    let audioBuffer = fs.readFileSync(audioFilePath);
    let transcript = await audioToText(audioBuffer);
    console.log("Transcription complete");
    console.log("transcript: ", transcript);
    res.send({ message: "Success", transcript: transcript });

    // console.log("Translating text...");
    // let translatedText = await translateText(transcript, "en");
    // console.log("Translation complete");

    // console.log("Converting text to speech...");
    // let audioFileName = await textToSpeech(translatedText, "en", videoDuration);
    // console.log("Text to speech conversion complete");

    // // Merge muted video with new audio
    // console.log("Merging audio and video...");
    // const mergedFilePath = await mergeAudioVideo(videoBuffer, audioFileName);
    // console.log("Audio and video merged");

    // // Clean up temporary files
    // await fs.promises.unlink(audioFileName);
    // await fs.promises.unlink(videoFilePath);

    // // Automatically download the file
    // res.download(mergedFilePath);
    // return res.send({ message: "Success" });
  } catch (error) {
    console.error("Error processing video:", error);
    return res.status(500).send({ message: error.message });
  }
});

const handleAudio = asyncHandler(async (req, res, next, data) => {
  console.log("Audio file detected");
  console.log("Transcribing audio to text...");
  let transcript = await audioToText(data);
  console.log("Transcription complete");
  console.log("Translating text...");
  let translatedText = await translateText(transcript, "en");
  console.log("Translation complete");
  console.log("Converting text to speech...");
  let videoFilePath = req.file.path;
  const videoDuration = await getVideoDuration(videoFilePath);
  let isAudio = true;
  let audioFilePath = await textToSpeech(translatedText, "ja", videoDuration, isAudio);
  console.log("Text to speech conversion complete");
  // Automatically download the file
  res.download(audioFilePath);
  return res.send({ message: "Success" });
});

module.exports = { entryFunction };
