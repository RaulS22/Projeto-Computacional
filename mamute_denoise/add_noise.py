from pydub import AudioSegment


def overlay_wav_files(audio1_file, audio2_file, overlayed_file):
    # Load the audio files
    sound1 = AudioSegment.from_file(audio1_file)
    sound2 = AudioSegment.from_file(audio2_file)

    # Adjust properties to make them compatible
    sound2 = sound2.set_frame_rate(sound1.frame_rate)
    sound2 = sound2.set_channels(sound1.channels)
    sound2 = sound2.set_sample_width(sound1.sample_width)

    # If the durations are different, trim or pad the shorter one
    if len(sound1) < len(sound2):
        sound1 = sound1 + AudioSegment.silent(duration=len(sound2) - len(sound1))
    elif len(sound2) < len(sound1):
        sound2 = sound2 + AudioSegment.silent(duration=len(sound1) - len(sound2))

    # Overlay the audio files
    overlay_sound = sound1.overlay(sound2)

    # Export the result to a new file
    overlay_sound.export(overlayed_file, format="wav")

if __name__ == "__main__":
    audio1_file = "New Recording 5.wav"
    audio2_file = "sample-1.wav"
    overlayed_file = "overlayed_test.wav"

    overlay_wav_files(audio1_file, audio2_file, overlayed_file)