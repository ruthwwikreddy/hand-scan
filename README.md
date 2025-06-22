# Hand Scan Application

A web-based application that detects hand gestures and overlays a video when the hand is placed correctly.

## Features

- Real-time hand detection using MediaPipe
- Transparent hand template overlay
- Video playback when hand is detected in correct position
- Fullscreen support
- Modern web interface
- Automatic video overlay on camera feed

## Prerequisites

- Python 3.7+
- Webcam
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ruthwwikreddy/hand-scan.git
```

2. Create a virtual environment:
```bash
python -m venv cv-env
source cv-env/bin/activate  # On Windows: cv-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:5050
```

3. Place your hand in the template area:
   - The template will appear as a semi-transparent overlay
   - When your hand is detected correctly, the video will automatically play
   - The video will keep playing even if you remove your hand
   - Click the camera feed to go fullscreen

## Project Structure

```
hand_scan-main/
├── main.py              # Main Flask application
├── requirements.txt     # Python dependencies
├── static/
│   ├── hand_template.png  # Hand template image
│   └── vid.mp4          # Video file to play
├── templates/
│   └── index.html       # Web interface
└── docs/
    ├── hand_template.png  # Documentation images
    └── index.html        # Documentation page
```

## Customization

1. To change the video:
   - Place your video file in `static/vid.mp4`
   - Ensure it's in MP4 format with H.264 codec

2. To change the hand template:
   - Replace `static/hand_template.png` with your own template
   - Make sure it's a PNG with transparency

## Troubleshooting

1. If the video doesn't play:
   - Check if `static/vid.mp4` exists
   - Ensure the video format is supported
   - Try a different video file

2. If hand detection doesn't work:
   - Make sure your webcam is working
   - Check if proper lighting is available
   - Try moving your hand closer to the camera

## License

MIT License
