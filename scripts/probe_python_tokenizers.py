import json
import sys
from transformers import AutoTokenizer, T5TokenizerFast

qwen = AutoTokenizer.from_pretrained(
    r"C:\local\gnu\ComfyUI\kohya_ss\sd-scripts\configs\qwen3_06b",
    local_files_only=True,
)
t5 = T5TokenizerFast.from_pretrained(
    r"C:\local\gnu\ComfyUI\kohya_ss\sd-scripts\configs\t5_old",
    local_files_only=True,
)
paths = sys.argv[1:] or [None]
for path in paths:
    text = open(path, encoding="utf-8").read() if path else "Elvira, Atelier Sophie 2"
    print("caption=" + json.dumps(text[:40]))
    for name, tokenizer in (("qwen", qwen), ("t5", t5)):
        encoded = tokenizer(
            [text], return_tensors="pt", truncation=True, padding="max_length", max_length=512
        )
        ids = encoded["input_ids"][0].tolist()
        mask = encoded["attention_mask"][0].tolist()
        print(name + "_valid=" + str(sum(mask)))
        print(name + "_ids=" + json.dumps(ids[:20]))
        print(name + "_mask=" + json.dumps(mask[:20]))
