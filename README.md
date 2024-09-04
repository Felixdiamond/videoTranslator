# VideoTranslator ヾ(⌐■_■)ノ♪

## Welcome to my open-source video translation adventure! (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧

I got tired of waiting for someone else to make a cool open-source video translator, so I decided to create one myself! This project aims to make video translation accessible to everyone by using free services. It's still a work in progress, but hey, Rome wasn't built in a day, right? (￣ω￣)

### Current Status
The basic script is up and running! ٩(◕‿◕｡)۶ But like any budding masterpiece, it needs some TLC, especially in the playback speed department. I'm working on it, so stay tuned! (✿◠‿◠)

### How You Can Help
Got any brilliant ideas to improve the sync between video and split audio? Don't be shy! Contribute or drop me a line. Let's make this translator awesome together! ᕦ(ò_óˇ)ᕤ

## Features (ﾉ´ヮ`)ﾉ*: ･ﾟ
- Translates videos to your desired language
- Uses freely available services (because who doesn't love free stuff?)
- Works with various video formats
- Preserves original sound effects and music

## Installation (⌐■_■)

### Note
You need **python 3.11.9** to run this, that's the only python version I've tested that works with this script. But don't worry, we've got options! Choose your fighter:

### Option 1: The Classic (POV: You hate GUI)

1. Clone this bad boy:
```
git clone https://github.com/Felixdiamond/videoTranslator.git && cd videoTranslator
```

2. Install the cool kids (I mean, dependencies):
```
pip install -r requirements.txt
```

### Option 2: The Fancy Setup (Script kiddies)

1. Clone the repo (same as above, we're not reinventing the wheel here):
```
git clone https://github.com/Felixdiamond/videoTranslator.git && cd videoTranslator
```

2. Run our setup script:
```
python setup.py
```

3. Wait for the script to create a virtual environment, install backend dependencies, and sets up the frontend.

## Usage (╯°□°）╯︵ ┻━┻

### If you chose Option 1

1. Run the script:
```
python translator.py <path_to_your_video> <target_language>
```

2. Grab a snack and wait for the translation magic to happen! ✨

3. Find your shiny new translated video in the root directory of the created folder.

### If you chose Option 2

1. Activate your virtual environment:
- On Windows: `venv\Scripts\activate`
- On macOS/Linux: `source venv/bin/activate`


2. Run the runner script (run the runner script lol):
```
python run.py
```

3. Open `http://localhost:3000` on your browser and use that to upload your video

4. Choose your target language

5. Wait for it to complete then navigate to the directory specified on the frontend (i'm not teaching you how to use file explorer)

## Known Issues (;´༎ຶД༎ຶ`)
- Playback speed needs some work (it's not perfect... yet!)
- Resource hungry (or maybe that's just because i'm using an 8gb ram pc with 512 HDD lol)
- Time consuming (again, maybe just my pc)
- May occasionally translate "hello" to "goodbye" (just kidding, but you never know with AI!)

## Future Plans (づ｡◕‿‿◕｡)づ
- Improve synchronization (because timing is everything in comedy and video translation)
- Add support for more languages (maybe even Klingon?)
- Make the UI so pretty, you'll want to frame it

## Contributing ᕕ( ᐛ )ᕗ
Got ideas? Found a bug? Want to add Klingon support? Contributions are always welcome! Feel free to open an issue or submit a pull request.

ありがとう for checking out my project! Let's make video translation great again! (oﾟvﾟ)ノ