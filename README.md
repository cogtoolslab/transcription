# Whisper transcription script
This python script transcribes audio in any format to a json file. The output is timestamped by word and looks like:

```
[
    {"text": "Hello", "timestamp": (0.0, 1.3)},
    {"text": "world", "timestamp": (1.4, 2.2)}
]
```
It uses the whisper-large-v3-turbo model by openai, provided by Huggingface (at: [https://huggingface.co/openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo))

## Requirements
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (in your own data/\<username\> directory)
and set up a conda environment using the requirements file by running:

```
conda env create -f /data/models/transciption/basic-huggingface-environment.yml
```

Then, to use the already downloaded model weights, set your hugging face home environment variable by running:

```
export HF_HOME="/data/models/.cache/huggingface
```

## Usage
Make sure you are inside your huggingface conda environment (activate with ``conda activate huggingface``).

To transcribe a list of audio files, run ``python3 transcribe.py [audio_file ...] [--output_dir]`` 

The first argument(s) are one or more paths to audio files,
and the second optional argument is preceded by the flag ``-o`` or ``--output_dir`` and is a path to a directory where
the jsons will be written.

For example,

```python3 transcribe.py /data/example_user/audio1.wav /data/example_user/audio2.wav -o /data/example_user/transcripts/```

Will transcribe two audio files and output them to ``/data/example_user/transcripts/audio1.json`` and ``/data/example_user/transcripts/audio2.json``.

