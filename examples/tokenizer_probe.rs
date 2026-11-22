use std::env;

use afterburner::{QwenTokenizer, T5Tokenizer};

fn main() {
    let mut args = env::args().skip(1);
    let qwen_path = args.next().expect("qwen tokenizer.json path");
    let t5_path = args.next().expect("t5 tokenizer.json path");
    let caption = args.collect::<Vec<_>>().join(" ");
    assert!(!caption.is_empty(), "caption is required");

    let qwen = QwenTokenizer::from_file(qwen_path, 512, 151_643).unwrap();
    let t5 = T5Tokenizer::from_file(t5_path, 512, 0).unwrap();
    let q = qwen.encode_caption(&caption).unwrap();
    let t = t5.encode_caption(&caption).unwrap();

    println!("caption={caption:?}");
    println!("qwen_ids={:?}", q.token_ids);
    println!("qwen_mask={:?}", q.attention_mask.unwrap());
    println!("t5_ids={:?}", t.token_ids);
    println!("t5_mask={:?}", t.attention_mask.unwrap());
}
