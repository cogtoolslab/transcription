import torch, sys, argparse, json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

parser = argparse.ArgumentParser(prog='python3 transcribe.py')
parser.add_argument('audio_file', nargs='+')
parser.add_argument('-o', '--output_dir', nargs='?')
args = parser.parse_args()

in_files = args.audio_file
out_path = args.output_dir if args.output_dir is not None else ""

print("Reading audio from "+str(in_files) + 
        (" and outputting "+"here" if out_path == "" else "to "+out_path))


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

for in_file in in_files:
    result = pipe(in_file, return_timestamps="word")
    with open(out_path+in_file.split('/')[-1].split('.')[0]+".json", "w") as f:
        json.dump(result["chunks"], f)

