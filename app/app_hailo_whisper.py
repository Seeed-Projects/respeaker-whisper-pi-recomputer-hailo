"""Main app for Hailo Whisper"""

import time
import argparse
import os
import sys
import queue
import threading
import numpy as np
import sounddevice as sd
from app.hailo_whisper_pipeline import HailoWhisperPipeline
from common.audio_utils import load_audio, SAMPLE_RATE, pad_or_trim
from common.preprocessing import preprocess, improve_input_audio
from common.postprocessing import clean_transcription
from common.timing_utils import set_timing_display
from app.whisper_hef_registry import HEF_REGISTRY


# Audio parameters for real-time processing
DURATION = 5  # chunk duration in seconds
BLOCK_SIZE = 2048  # Increased block size for better performance


def get_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Whisper Hailo Pipeline")
    parser.add_argument(
        "--reuse-audio", 
        action="store_true", 
        help="Reuse the previous audio file (sampled_audio.wav)"
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        choices=["hailo8", "hailo8l"],
        help="Hardware architecture to use (default: hailo8)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "tiny", "tiny.en"],
        help="Whisper variant to use (default: base)"
    )
    parser.add_argument(
        "--multi-process-service", 
        action="store_true", 
        help="Enable multi-process service to run other models in addition to Whisper"
    )
    parser.add_argument(
        "--real-time", 
        action="store_true", 
        help="Enable real-time speech-to-text processing"
    )
    parser.add_argument(
        "--fast-mode", 
        action="store_true", 
        help="Enable fast mode for real-time processing (reduced accuracy for speed)"
    )
    parser.add_argument(
        "--stream-output", 
        action="store_true", 
        help="Enable streaming output (character by character) in real-time mode"
    )
    parser.add_argument(
        "--timing", 
        action="store_true", 
        help="Enable timing analysis for performance profiling"
    )
    parser.add_argument(
        "--chunk-length",
        type=float,
        default=None,
        help="Audio chunk length in seconds (default: 10 for tiny, 5 for base)"
    )
    return parser.parse_args()


def get_hef_path(model_variant: str, hw_arch: str, component: str) -> str:
    """
    Method to retrieve HEF path.

    Args:
        model_variant (str): e.g. "tiny", "base"
        hw_arch (str): e.g. "hailo8", "hailo8l"
        component (str): "encoder" or "decoder"

    Returns:
        str: Absolute path to the requested HEF file.
    """
    try:
        hef_path = HEF_REGISTRY[model_variant][hw_arch][component]
    except KeyError as e:
        raise FileNotFoundError(
            f"HEF not available for model '{model_variant}' on hardware '{hw_arch}'."
        ) from e

    if not os.path.exists(hef_path):
        raise FileNotFoundError(f"HEF file not found at: {hef_path}\nIf not done yet, please run python3 ./download_resources.py --hw-arch {hw_arch} from the app/ folder to download the required HEF files.")
    return hef_path


def real_time_stt(whisper_hailo, is_nhwc=True, chunk_length=5, fast_mode=False, stream_output=False, timing=False):
    """
    Real-time speech-to-text processing function.
    
    Args:
        whisper_hailo: HailoWhisperPipeline instance
        is_nhwc: Whether to use NHWC format
        chunk_length: Length of audio chunks in seconds
        fast_mode: Whether to use fast mode (reduced accuracy for speed)
        stream_output: Whether to stream output character by character
        timing: Whether to show timing analysis
    """
    print("🎙️  Real-time speech-to-text started")
    print("🗣️  Please speak into the microphone")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 40)
    
    # Buffer to accumulate audio data
    audio_buffer = np.array([], dtype=np.float32)
    buffer_lock = threading.Lock()
    
    def audio_callback(indata, frames, time, status):
        """Callback function for audio input stream."""
        if status:
            print(f"Audio status: {status}")
        with buffer_lock:
            nonlocal audio_buffer
            # Append new audio data to buffer
            audio_buffer = np.concatenate([audio_buffer, indata[:, 0]], dtype=np.float32)  # Assuming mono audio
    
    def transcription_worker():
        """Worker thread to process audio chunks and get transcriptions."""
        chunks_processed = 0
        
        while True:
            audio_chunk = None
            with buffer_lock:
                nonlocal audio_buffer
                # Check if we have enough audio data for processing
                required_samples = int(chunk_length * SAMPLE_RATE)
                # Process as soon as we have enough data
                if len(audio_buffer) >= required_samples and required_samples > 0:
                    # Extract a chunk of audio
                    audio_chunk = audio_buffer[:required_samples].copy()  # Copy only when necessary
                    # Remove processed audio from buffer
                    audio_buffer = audio_buffer[required_samples:]
            
            if audio_chunk is not None and len(audio_chunk) > 0:
                chunks_processed += 1
                
                # Process the audio chunk
                try:
                    # Disable VAD for faster processing in fast mode
                    use_vad = not fast_mode  # Only use VAD in normal mode
                    improved_audio, start_time = improve_input_audio(audio_chunk, vad=use_vad, low_audio_gain=True)
                    
                    # Handle case where start_time might be None
                    if start_time is not None:
                        chunk_offset = start_time - 0.2
                        if chunk_offset < 0:
                            chunk_offset = 0
                    else:
                        chunk_offset = 0

                    # Preprocess to get mel spectrograms with corrected data types
                    mel_spectrograms = preprocess(
                        improved_audio.astype(np.float32),  # Ensure float32
                        is_nhwc=is_nhwc,
                        chunk_length=chunk_length,
                        chunk_offset=chunk_offset
                    )

                    # Send to Whisper pipeline for each mel spectrogram
                    for mel in mel_spectrograms:
                        whisper_hailo.send_data(mel.astype(np.float32))  # Ensure float32
                        time.sleep(0.05)  # Allow processing time
                        raw_transcription = whisper_hailo.get_transcription()
                        transcription = clean_transcription(raw_transcription)
                        
                        # Output transcriptions cleanly
                        if transcription.strip() and transcription.strip() != ".":  # Only print meaningful transcriptions
                            if stream_output:
                                # Stream the transcription character by character
                                print("[Transcription] ", end="", flush=True)
                                for char in transcription:
                                    print(char, end="", flush=True)
                                    time.sleep(0.01)  # Very small delay for smooth streaming
                                print()  # New line after transcription
                            else:
                                # Normal output
                                print(f"[Transcription] {transcription}")
                        elif transcription.strip() == ".":
                            # Don't show empty periods
                            pass
                        else:
                            # Only show empty transcriptions if audio level is significant
                            if np.max(np.abs(improved_audio)) > 0.05:  # Increased threshold to reduce noise
                                if stream_output:
                                    print("[Transcription] (unclear)", end="", flush=True)
                                else:
                                    print("[Transcription] (unclear audio)")
                except Exception as e:
                    print(f"Error: {e}")
            
            time.sleep(0.01)  # Small delay to prevent busy waiting
    
    # Start transcription worker thread
    worker_thread = threading.Thread(target=transcription_worker, daemon=True)
    worker_thread.start()
    
    # Start audio input stream
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype='float32',
            callback=audio_callback
        ):
            print("🎧 Listening...")
            # Keep the stream running
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n" + "-" * 40)
        print("🛑 Stopping...")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """
    Main function to run the Hailo Whisper pipeline.
    """
    # Get command line arguments
    args = get_args()

    variant = args.variant
    print(f"Selected variant: Whisper {variant}")
    encoder_path = get_hef_path(variant, args.hw_arch, "encoder")
    decoder_path = get_hef_path(variant, args.hw_arch, "decoder")

    # Set timing display
    set_timing_display(args.timing)
    
    whisper_hailo = HailoWhisperPipeline(encoder_path, decoder_path, variant, multi_process_service=args.multi_process_service)
    print("Hailo Whisper pipeline initialized.")
    
    if args.timing:
        print("Timing analysis enabled. Performance metrics will be displayed.")
    
    if args.real_time:
        # Run real-time STT
        is_nhwc = True
        # Use custom chunk length if specified, otherwise use model defaults
        if args.chunk_length is not None:
            chunk_length = args.chunk_length
        else:
            # Use model-specific default chunk lengths
            chunk_length = 10 if "tiny" in variant else 5
        # Use fast mode if specified
        real_time_stt(whisper_hailo, is_nhwc, chunk_length, fast_mode=args.fast_mode, stream_output=args.stream_output, timing=args.timing)
    else:
        # Original batch processing mode
        audio_path = "sampled_audio.wav"
        is_nhwc = True
        chunk_length = 10 if "tiny" in variant else 5

        while True:
            if args.reuse_audio:
                # Reuse the previous audio file
                if not os.path.exists(audio_path):
                    print(f"Audio file {audio_path} not found. Please record audio first.")
                    break
            else:
                user_input = input("\nPress Enter to start recording, or 'q' to quit: ")
                if user_input.lower() == "q":
                    break
                # Record audio
                from common.record_utils import record_audio
                sampled_audio = record_audio(DURATION, audio_path=audio_path)

            # Process audio
            sampled_audio = load_audio(audio_path)

            sampled_audio, start_time = improve_input_audio(sampled_audio, vad=True)
            chunk_offset = start_time - 0.2
            if chunk_offset < 0:
                chunk_offset = 0

            mel_spectrograms = preprocess(
                sampled_audio,
                is_nhwc=is_nhwc,
                chunk_length=chunk_length,
                chunk_offset=chunk_offset
            )

            for mel in mel_spectrograms:
                whisper_hailo.send_data(mel)
                time.sleep(0.1)
                transcription = clean_transcription(whisper_hailo.get_transcription())
                print(f"\n{transcription}")

            if args.reuse_audio:
                break  # Exit the loop if reusing audio

    whisper_hailo.stop()


if __name__ == "__main__":
    main()