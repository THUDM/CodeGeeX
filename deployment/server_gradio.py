import json
import torch
import argparse
import gradio as gr

import codegeex
from codegeex.torch import CodeGeeXModel
from codegeex.tokenizer import CodeGeeXTokenizer
from codegeex.quantization import quantize
from codegeex.data.data_utils import LANGUAGE_TAG
from codegeex.megatron.inference import set_random_seed


def model_provider(args):
    """Build the model."""

    model = CodeGeeXModel(
        args.hidden_size,
        args.num_layers,
        args.num_attention_heads,
        args.padded_vocab_size,
        args.max_position_embeddings
    )
    return model


def add_code_generation_args(parser):
    group = parser.add_argument_group(title="code generation")
    group.add_argument(
        "--num-layers",
        type=int,
        default=39,
    )
    group.add_argument(
        "--hidden-size",
        type=int,
        default=5120,
    )
    group.add_argument(
        "--num-attention-heads",
        type=int,
        default=40,
    )
    group.add_argument(
        "--padded-vocab-size",
        type=int,
        default=52224,
    )
    group.add_argument(
        "--max-position-embeddings",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--tokenizer-path",
        type=str,
        default="./tokenizer",
    )
    group.add_argument(
        "--example-path",
        type=str,
        default="./",
    )
    group.add_argument(
        "--load",
        type=str,
    )
    group.add_argument(
        "--state-dict-path",
        type=str,
    )
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )
    group.add_argument(
        "--quantize",
        action="store_true",
    )
    
    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()
    
    print("Loading tokenizer ...")
    tokenizer = CodeGeeXTokenizer(
        tokenizer_path=args.tokenizer_path, 
        mode="codegeex-13b")

    print("Loading state dict ...")
    state_dict = torch.load(args.load, map_location="cpu")
    state_dict = state_dict["module"]

    print("Building CodeGeeX model ...")
    model = model_provider(args)
    model.load_state_dict(state_dict)
    model.eval()
    model.half()
    if args.quantize:
        model = quantize(model, weight_bit_width=8, backend="torch")
    model.cuda()

    def predict(
        prompt, 
        lang,
        seed, 
        out_seq_length, 
        temperature, 
        top_k, 
        top_p,
    ):
        set_random_seed(seed)
        if lang.lower() in LANGUAGE_TAG:
            prompt = LANGUAGE_TAG[lang.lower()] + "\n" + prompt
        
        generated_code = codegeex.generate(
            model,
            tokenizer,
            prompt,
            out_seq_length=out_seq_length,
            seq_length=args.max_position_embeddings,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            micro_batch_size=args.micro_batch_size,
            backend="megatron",
            verbose=True,
        )
        return prompt + generated_code

    examples = []
    with open(args.example_path, "r") as f:
        for line in f:
            examples.append(list(json.loads(line).values()))

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <img src="https://raw.githubusercontent.com/THUDM/CodeGeeX/main/resources/logo/codegeex_logo.png">
            """)
        gr.Markdown(
            """
            <p align="center">
                🏠 <a href="https://codegeex.cn" target="_blank">Homepage</a> | 📖 <a href="http://keg.cs.tsinghua.edu.cn/codegeex/" target="_blank">Blog</a> | 🪧 <a href="https://codegeex.cn/playground" target="_blank">DEMO</a> | 🛠 <a href="https://marketplace.visualstudio.com/items?itemName=aminer.codegeex" target="_blank">VS Code</a> or <a href="https://plugins.jetbrains.com/plugin/20587-codegeex" target="_blank">Jetbrains</a> Extensions | 💻 <a href="https://github.com/THUDM/CodeGeeX" target="_blank">Source code</a> | 🤖 <a href="https://models.aminer.cn/codegeex/download/request" target="_blank">Download Model</a>
            </p>
            """)
        gr.Markdown(
            """
            We introduce CodeGeeX, a large-scale multilingual code generation model with 13 billion parameters, pre-trained on a large code corpus of more than 20 programming languages. CodeGeeX supports 15+ programming languages for both code generation and translation. CodeGeeX is open source, please refer to our [GitHub](https://github.com/THUDM/CodeGeeX) for more details. This is a minimal-functional DEMO, for other DEMOs like code translation, please visit our [Homepage](https://codegeex.cn). We also offer free [VS Code](https://marketplace.visualstudio.com/items?itemName=aminer.codegeex) or [Jetbrains](https://plugins.jetbrains.com/plugin/20587-codegeex) extensions for full functionality. 
            """)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=13, placeholder='Please enter the description or select an example input below.',label='Input')
                with gr.Row():
                    gen = gr.Button("Generate")
                    clr = gr.Button("Clear")

            outputs = gr.Textbox(lines=15, label='Output')

        gr.Markdown(
            """
            Generation Parameter
            """)
        with gr.Row():
            with gr.Column():
                lang = gr.Radio(
                    choices=["C++", "C", "C#", "Python", "Java", "HTML", "PHP", "JavaScript", "TypeScript", "Go",
                             "Rust",
                             "SQL", "Kotlin", "R", "Fortran"], value='lang', label='Programming Language',
                    default="Python")
            with gr.Column():
                seed = gr.Slider(maximum=10000, value=8888, step=1, label='Seed')
                with gr.Row():
                    out_seq_length = gr.Slider(maximum=1024, value=128, minimum=1, step=1, label='Output Sequence Length')
                    temperature = gr.Slider(maximum=1, value=0.2, minimum=0, label='Temperature')
                with gr.Row():
                    top_k = gr.Slider(maximum=40, value=0, minimum=0, step=1, label='Top K')
                    top_p = gr.Slider(maximum=1, value=1.0, minimum=0, label='Top P')

        inputs = [prompt, lang, seed, out_seq_length, temperature, top_k, top_p]
        gen.click(fn=predict, inputs=inputs, outputs=outputs)
        clr.click(fn=lambda value: gr.update(value=""), inputs=clr, outputs=prompt)

        gr_examples = gr.Examples(examples=examples, inputs=[prompt, lang],
                                  label="Example Inputs (Click to insert an examplet it into the input box)",
                                  examples_per_page=20)
        
    demo.launch(server_port=6007)

if __name__ == '__main__':
    with torch.no_grad():
        main()