# VideoTranslator ヾ(⌐■_■)ノ♪

## Welcome to my open-source video translation adventure! (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧

I got tired of waiting for someone else to make a cool open-source video translator, so I decided to create one myself! This project aims to make video translation accessible to everyone by leveraging powerful open-source AI models. (￣ω￣)

### Current Status & ✨ Major Update! ✨
The script has undergone a significant upgrade! ٩(◕‿◕｡)۶ We've integrated **MeloTTS** for higher-quality, more natural-sounding voice synthesis, and implemented **GPU acceleration** (where available) for faster processing. The entire pipeline, from transcription to final video generation, is now more robust and optimized. Performance monitoring has also been added to help track resource usage.

### How You Can Help
Got brilliant ideas to further improve synchronization, add more voice options, or enhance performance? Don't be shy! Contribute or drop me a line. Let's make this translator awesome together! ᕦ(ò_óˇ)ᕤ

## Features (ﾉ´ヮ`)ﾉ*: ･ﾟ
- **High-Quality Translation & TTS**: Translates videos using advanced models and synthesizes speech with MeloTTS (falling back to gTTS if needed).
- **Multiple Languages Supported**: Current support includes English, Spanish, French, Chinese, Japanese, Korean, German, and Portuguese (see `translator.py` for the latest `LANGUAGE_MODEL_MAP`).
- **GPU Accelerated**: Leverages your GPU (if CUDA is available and PyTorch is set up correctly) for faster transcription, translation, and other ML tasks.
- **Preserves Original Audio**: Keeps background music and sound effects intact.
- **Improved Synchronization**: Advanced audio processing techniques for better lip-sync and timing.
- **Performance Monitoring**: Logs processing times and memory usage for different stages.
- **Web UI & CLI**: Use the simple web interface or run directly from the command line.

## 🎬 Demo: English to French Translation

Here's a sample of the VideoTranslator in action!

I translated the first 5 minutes of this video by Fern:
- **Original Video (English):** [The Hunt for America's Smartest Killer](https://youtu.be/wkVygetgeRY?si=hKF2XqJD3jZU3KIL)

The full 28-minute video wasn't translated as it would take a significant amount of time (likely well over an hour) on Kaggle T4 GPU at 360p resolution. This 5-minute clip demonstrates the translation quality and process.

- **Translated Output (First 5 mins, French):** [View Translated Sample (translated_fern_eng.mp4)](./translated_fern_eng.mp4)

## Installation (⌐■_■)

### Prerequisites
- **Python 3.11 or later**: This is the recommended and tested version.
- **CUDA-enabled GPU (Recommended for speed)**: Ensure you have NVIDIA drivers and a CUDA toolkit version compatible with PyTorch.

### Note on MeloTTS
This project now uses MeloTTS. You'll need to clone its repository and install it. The `setup.py` script attempts to handle this.

### Option 1: The Classic (CLI Lovers)
1.  **Clone this repository:**
    ```bash
    git clone https://github.com/Felixdiamond/videoTranslator.git
    cd videoTranslator
    ```
2.  **Install MeloTTS (if not handled by `requirements.txt` or `setup.py`):**
    It's recommended to follow the official MeloTTS installation if you encounter issues. The project structure expects MeloTTS to be in a directory named `MeloTTS` at the same level as `videoTranslator` or for the package to be installed in the environment.
    ```bash
    # Example:
    # git clone https://github.com/myshell-ai/MeloTTS.git ../MeloTTS
    # cd ../MeloTTS
    # pip install -e .
    # python -m unidic download
    # cd ../videoTranslator
    ```
    *(The `setup.py` script aims to automate part of this, but manual setup might be needed depending on your environment.)*

3.  **Install dependencies:**
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    pip install -r requirements.txt
    ```
    *The `requirements.txt` will be updated to include new dependencies like `accelerate` and ensure PyTorch with CUDA support is specified if possible (though direct CUDA version in `requirements.txt` can be tricky).*

### Option 2: The Fancy Setup (Automated)
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Felixdiamond/videoTranslator.git
    cd videoTranslator
    ```
2.  **Run the setup script:**
    This script will attempt to create a virtual environment, install backend dependencies (including trying to set up MeloTTS), and set up the frontend.
    ```bash
    python setup.py
    ```
3.  **Follow script prompts.** It will guide you through installing Python if not present and other dependencies.

## Usage (╯°□°）╯︵ ┻━┻

### If you chose Option 1 (CLI)
1.  **Activate your virtual environment** (if you created one).
2.  **Run the script:**
    ```bash
    python translator.py <path_to_your_video> <target_language_code>
    ```
    Example: `python translator.py my_video.mp4 fr`
    Supported language codes: `en`, `es`, `fr`, `zh`, `ja`, `ko`, `de`, `pt`.
3.  Grab a (potentially larger) snack! Processing can take time, especially on CPU. ✨
4.  Find your translated video in a newly created folder (e.g., `my_video_translated_output/translated_my_video_fr.mp4`). Logs are saved in the `logs` directory.

### If you chose Option 2 (Web UI)
1.  **Activate your virtual environment:**
    -   On Windows: `venv\Scripts\activate`
    -   On macOS/Linux: `source venv/bin/activate`
2.  **Run the application:**
    ```bash
    python run.py
    ```
3.  Open `http://localhost:3000` in your browser.
4.  Upload your video and select the target language.
5.  Wait for the magic! The UI will indicate when processing is complete and show the output path.

## Known Issues & Considerations (;´༎ຶД༎ຶ`)
- **Resource Intensive**: AI models, especially for video, require significant CPU/GPU and RAM. GPU is highly recommended. **But you don't need something too high end, all tests i've done is with kaggle + T4 GPU**
- **Time Consuming**: Translation is a multi-step process. Be patient!
- **MeloTTS Speaker IDs**: The current implementation uses default speaker IDs for MeloTTS. Voice variety might be limited per language.
- **Error Handling**: While improved, complex pipelines can have various failure points. Check `logs/video_translator.log` for details if issues arise.
- **Speech sometimes unnaturally fast/slow**

## Future Plans (づ｡◕‿‿◕｡)づ
- **Fix speech speed**
- **Voice Cloning/Selection**: Integrate more advanced voice options, potentially voice cloning for the target language.
- **Subtitle Generation**: Option to generate and embed translated subtitles.
- **Further Performance Optimization**: Continuously improve speed and resource efficiency.
- **Expanded Language Support**: Add more languages as high-quality open-source models become available.
- **UI Enhancements**: More detailed progress, error reporting, and configuration options in the UI.

## Contributing ᕕ( ᐛ )ᕗ
Ideas? Bugs? Want to add Klingon (MeloTTS might need some training data!)? Contributions are welcome! Open an issue or submit a PR.

ありがとう for checking out this VideoTranslator! Let's keep pushing the boundaries of open-source AI! (oﾟvﾟ)ノ