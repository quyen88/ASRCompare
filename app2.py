import gradio as gr
from yt_dlp import YoutubeDL
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import jax
# from whisper_jax import FlaxWhisperPipline
# import jax.numpy as jnp
import whisper

# pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.float16)
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.initialize_cache("/home/tdh/whisper-jax-lambda/jax_cache")
    
# def download_video(url):
#   # ydl_opts = {'overwrites':True, 'format':'bestaudio[ext=m4a]', 'outtmpl':'/home/tdh/whisper-jax-lambda/audio.mp3'}
#     ydl_opts = {
#     'format': 'm4a/bestaudio/best',
#     'postprocessors': [{  # Extract audio using ffmpeg
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#     }]
# }
#     with YoutubeDL(ydl_opts) as ydl:
#       ydl.download(url)
#       return f"/home/tdh/whisper-jax-lambda/audio.mp3"
  # output_path = "/home/tdh/whisper-jax-lambda"
  # filename = "audio.mp3"
  # yt = YouTube(url)
  # audio_stream = yt.streams.filter(only_audio=True).first()
  # audio_stream.download(output_path=output_path, filename=filename)

# Copied from https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/utils.py#L50
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
    if seconds is not None:
        milliseconds = round(seconds * 1000.0)
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000
        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        return seconds



# def transcribe(audio_in):
#     # outputs = pipeline("/home/tdh/whisper-jax-lambda/audio.m4a", return_timestamps=True)
#     model = whisper.load_model("base")
#     outputs = model.transcribe(audio_in)
 
#     text = outputs["text"]
#     # chunks = outputs["chunks"]
#     output = ""
#     # https://huggingface.co/spaces/jeffistyping/Youtube-Whisperer/blob/main/app.py modifyed
#     for i, chunk in enumerate(chunks):
#       output += f"{i+1}\n"
#       output += f"{format_timestamp(chunk['timestamp'][0])} --> {format_timestamp(chunk['timestamp'][1])}\n"
#       output += f"{chunk['text']}\n\n"
#     return text, output

def crawScript(url):
  try:
    video_id = url.split('?')[1].split('=')[1]
  except:
    return "No response, pls check the url :)" 
    
  transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['vi', 'en'])
  formatter = TextFormatter()
  text_formatted = formatter.format_transcript(transcript)
  return text_formatted

# def download_video(url):
#   ydl_opts = {'overwrites':True, 'format':'bestaudio[ext=m4a]','postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3'}], 'outtmpl':'/home/tdh/whisper-jax-lambda/audio.mp3'}
#   with YoutubeDL(ydl_opts) as ydl:
#     ydl.download(url)
#     return f"/home/tdh/whisper-jax-lambda/audio.mp3.mp3"

def download_video(url):
  ydl_opts = {'overwrites':True, 'format':'bestaudio[ext=m4a]', 'outtmpl':'/home/tdh/whisper-jax-lambda/audio.m4a'}
  with YoutubeDL(ydl_opts) as ydl:
    ydl.download(url)
    return f"/home/tdh/whisper-jax-lambda/audio.m4a"

def transcribeWisper(audio_in):
    model = whisper.load_model("base")
    audio_in = '/home/tdh/whisper-jax-lambda/audio.m4a'
    result = model.transcribe(audio_in)
    transcribed_text = result["text"]
    return transcribed_text

app = gr.Blocks()
with app:
  gr.Markdown('## openai/whisper-large-v2')
  with gr.Row():
    with gr.Column():
      input_text = gr.Textbox(show_label=False, value="https://www.youtube.com/watch?v=dd1kN_myNDs")
      input_download_button = gr.Button(value="Download from YouTube")
      input_transcribe_button = gr.Button(value="Transcribe ASR")
      input_transcribe_youtube = gr.Button(value="Transcribe Youtube")
      text_out_fromGoogle = gr.Textbox(label="Crawl Youtube")
      input_transcribe_youtube.click(crawScript, inputs=[input_text], outputs=[text_out_fromGoogle])
    with gr.Column():
        audio_out = gr.Audio(label="Output Audio",type="filepath")
        text_out = gr.Textbox(label="Output ASR")
        # chunks_out = gr.Textbox(label="Output SRT")
    
    input_download_button.click(download_video, inputs=[input_text], outputs=[audio_out])
    input_transcribe_button.click(transcribeWisper, inputs=[audio_out], outputs=[text_out])
    # input_transcribe_button.click(transcribe, inputs=[audio_out], outputs=[text_out, chunks_out])
app.launch()
