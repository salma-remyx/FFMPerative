# FFMPerative - Chat to Compose Video
<p align="center">
  <img src="https://github.com/remyxai/FFMPerative/blob/main/assets/ffmperative.gif" height=400px>
  <br>
  <img src="https://img.shields.io/pypi/v/ffmperative.svg">
  <img src="https://img.shields.io/pypi/dm/ffmperative">
  <img src="https://img.shields.io/github/license/remyxai/ffmperative.svg">

</p>

FFMPerative is your copilot for video editing workflows. Powered by Large Language Models (LLMs) through an intuitive chat interface, now you can compose video edits in natural language to do things like:

* Change Speed, Resize, Crop, Flip, Reverse Video/GIF
* Speech-to-Text Transcription and Closed-Captions

Just describe your changes like [these examples](https://remyxai.github.io/FFMPerative/).

## Setup 

### Requirements
* Python 3 
* [ffmpeg](https://ffmpeg.org)

PyPI:
```
pip install ffmperative
```

Or pip install from source:
```
git clone https://github.com/remyxai/FFMPerative.git
cd FFMPerative && pip install .
```

## Quickstart
Add closed-captions with:

```bash
ffmperative do --prompt "merge subtitles 'captions.srt' with video 'video.mp4' calling it 'video_caps.mp4'"
```

## Features

### Python Usage
Simply import the library and pass your command as a string to `ffmp`.

```python
from ffmperative import ffmp

ffmp("sample the 5th frame from '/path/to/video.mp4'")
```

### Compose 🎞️ 
Use the `compose` call to compose clips into an edited video. Use the optional `--prompt` flag to guide the composition by text prompt.
```bash
ffmperative compose --clips /path/to/video/dir --output /path/to/my_video.mp4 --prompt "Edit the video for social media"
```

### Resources
* [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/)
* [Sample FFMPerative Dataset](https://huggingface.co/datasets/remyxai/ffmperative-sample)
* [FFMPerative LLaMA2 checkpoint](https://huggingface.co/remyxai/ffmperative-7b)
* [Automatically Edit Videos from Google Drive in Colab](https://colab.research.google.com/drive/149byzCNd17dAehVuWXkiFQ2mVe_icLCa?usp=sharing)

### Community
* [Join us on Discord](https://discord.com/invite/b2yGuCNpuC)

### Trajectory Search — adapted from *Self-Improving Language Models with Bidirectional Evolutionary Search*

`ffmperative.ffmp(..., num_candidates=N)` can sample several agent rollouts and
pick the strongest tool-call sequence before executing it, instead of trusting a
single autoregressive draft (best-of-N). The selection in
`ffmperative/trajectory_search.py` brings two ideas from BES
([arXiv:2605.28814](https://arxiv.org/abs/2605.28814)) to inference time:

- **Dense per-step feedback** — each tool call is treated as a checkable
  subgoal (known tool, dependencies bound by earlier steps) and scored, rather
  than judging only the final result.
- **Forward candidate evolution** — a recombination operator splices groundable
  partial trajectories from different rollouts into a candidate no single
  rollout produced.

It is purely inference-time: no training, checkpoints, or gradients are
introduced.

Contributed via [Remyx Recommendation](https://engine.remyx.ai).
