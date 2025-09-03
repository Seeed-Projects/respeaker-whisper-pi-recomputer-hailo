# Automatic Speech Recognition with Whisper model

This application performs a speech-to-text transcription using OpenAI's *Whisper-tiny* and *Whisper-base* model on the Hailo-8/8L/10H AI accelerators.

## Prerequisites

Ensure your system matches the following requirements before proceeding:

- Platforms tested: x86, Raspberry Pi 5
- OS: Ubuntu 22 (x86) or Raspberry OS.
- **HailoRT 4.20 or 4.21** and the corresponding **PCIe driver** must be installed. You can download them from the [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- **ffmpeg** and **libportaudio2** installed for audio processing.
  ```
  sudo apt update
  sudo apt install ffmpeg
  sudo apt install libportaudio2
  ```
- **Python 3.10 or 3.11** installed.

## Installation - Inference only

Follow these steps to set up the environment and install dependencies for inference:

1. Clone this repository:

   ```sh
   git clone https://github.com/hailo-ai/Hailo-Application-Code-Examples.git
   cd Hailo-Application-Code-Examples/runtime/hailo-8/python/speech_recognition
   ```
   If you have any authentication issues, add your SSH key or download the zip.

2. Run the setup script to install dependencies:  

   ```sh
   python3 setup.py
   ```

3. Activate the virtual environment from the repository root folder:

   ```sh
   source whisper_env/bin/activate
   ```

4. Install PyHailoRT inside the virtual environment (must be downloaded from the Hailo Developer Zone), for example:
   ```sh
   pip install hailort-4.20.0-cp310-cp310-linux_x86_64.whl
   ```
   The PyHailoRT version must match the installed HailoRT version.
   **_NOTE:_** This step is not necessary for Raspberry Pi 5 users who installed the *hailo-all* package, since the *venv* will inherit the system package.

## Before running the app

- Make sure you have a microphone connected to your system. If you have multiple microphones connected, please make sure the proper one is selected in the system configuration, and that the input volume is set to a medium/high level.  
  A good quality microphone (or a USB camera) is suggested to acquire the audio.
- The application allows the user to acquire and process an audio sample up to 5 seconds long. The duration can be modified in the application code.
- The current pipeline supports **English language only**.

## Usage from CLI
1. Activate the virtual environment from the repository root folder:

   ```sh
   source whisper_env/bin/activate
   ```
2. Run the command line app (from the root folder)
   ```sh
   python3 -m app.app_hailo_whisper
   ```
   The app uses Hailo-8 models as default. If you have an Hailo-8L device, run the following command instead:
   ```sh
   python3 -m app.app_hailo_whisper --hw-arch hailo8l
   ```
   If you want to select a specific Whisper model, use the *--variant* argument:
   ```sh
   python3 -m app.app_hailo_whisper --variant base
   python3 -m app.app_hailo_whisper --variant tiny
   ```
   

### Command line arguments
Use the `python3 -m app.app_hailo_whisper --help` command to print the helper.

The following command line options are available:

- **--reuse-audio**: Reloads the audio from the previous run.
- **--hw-arch**: Selects the Whisper models compiled for the target architecture (*hailo8* / *hailo8l / hailo10h*). If not specified, the *hailo8* architecture is selected.
- **--variant**: Variant of the Whisper model to use (*tiny* / *base*). If not specified, the *base* model is used.
- **--multi-process-service**: Enables the multi-process service, to run other models on the same chip in addition to Whisper


## Performance Optimizations

This version includes several performance optimizations for real-time speech-to-text processing:

1. **Reduced Chunk Length**: Optimized audio chunk processing for faster response times
2. **Zero-Copy Memory Management**: Minimized memory allocations and copies for better performance
3. **Multi-Process Support**: Enabled running multiple models concurrently on Hailo devices
4. **Fast Mode Option**: Reduced accuracy for speed with the `--fast-mode` parameter
5. **Streaming Output**: Character-by-character output streaming with the `--stream-output` parameter
6. **Timing Analysis**: Performance profiling with the `--timing` parameter


