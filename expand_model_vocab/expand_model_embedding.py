import torch
from pathlib import Path
from transformers import AutoTokenizer
import lightning as L
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.pretrain import save_checkpoint
from litgpt.tokenizer import Tokenizer
from litgpt.utils import parse_devices


def initialize_fabric(devices: list, precision: str = "bf16-true", strategy: str = "auto") -> L.Fabric:
    """Initialize the Lightning Fabric with the specified devices and strategy."""
    return L.Fabric(
        devices=devices,
        num_nodes=1,
        strategy=strategy,
        precision=precision,
    )


def load_model(fabric: L.Fabric, model_config: Config, checkpoint_path: Path, tie_embeddings: bool = False) -> GPT:
    """Initialize and load the GPT model with or without tied embeddings."""
    with fabric.init_module(empty_init=True):
        model = GPT(model_config)
    
    if tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight

    if checkpoint_path.exists():
        fabric.load_raw(checkpoint_path / "lit_model.pth", model)

    return model


def expand_embeddings(fabric, model: GPT, tokenizer: Tokenizer, target_tokenizer: AutoTokenizer, tie_embeddings: bool = False) -> None:
    """Expand model embeddings to accommodate new tokens from the target tokenizer."""
    source_vocab = tokenizer.vocab_size
    target_vocab = target_tokenizer.vocab_size

    print(f"source vocab {source_vocab} target_vocab {target_vocab}")

    num_new_tokens = target_vocab - source_vocab

    if num_new_tokens <= 0:
        fabric.print("No new tokens to initialize.")
        return

    fabric.print(f"Initializing {num_new_tokens} new tokens.")
    input_embeddings = model.transformer.wte.weight.data
    output_embeddings = model.lm_head.weight.data if not tie_embeddings else input_embeddings

    avg_embedding = input_embeddings[:source_vocab].mean(dim=0)
    new_embeddings = avg_embedding.repeat(num_new_tokens, 1)

    model.transformer.wte.weight.data = torch.cat([input_embeddings, new_embeddings], dim=0)

    if not tie_embeddings:
        model.lm_head.weight.data = torch.cat([output_embeddings, new_embeddings], dim=0)


def save_model(fabric: L.Fabric, model: GPT, save_path: Path) -> None:
    """Save the model checkpoint in bfloat16 precision."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert model weights to bfloat16 before saving
    # for param in model.parameters():
    #     param.data = param.data.to(torch.bfloat16)

    fabric.save(save_path, model.state_dict())
    fabric.print(f"Model saved at {save_path}")


def print_embedding_statistics(fabric, model: GPT, avg_embedding: torch.Tensor, tie_embeddings: bool) -> None:
    """Print the embedding statistics to verify the expansion."""
    fabric.print(f"Shape of input embeddings: {model.transformer.wte.weight.shape}")
    fabric.print(f"Last token embedding: {model.transformer.wte.weight[-1][:10]}")

    if not tie_embeddings:
        fabric.print(f"Shape of output embeddings: {model.lm_head.weight.shape}")
        fabric.print(f"Last token output embedding: {model.lm_head.weight[-1][:10]}")

    fabric.print(f"Average embedding: {avg_embedding[:10]}")


def main():
    model_name = "google/gemma-2-2b"
    initial_checkpoint_dir = Path(f"checkpoints/{model_name}")
    save_path = Path("model/expanded_model/lit_model.pth")

    # Initialize components
    devices = parse_devices("auto")
    fabric = initialize_fabric(devices)
    fabric.launch()

    model_config = Config.from_name(model_name)
    tokenizer = Tokenizer(f"checkpoints/{model_name}")
    target_tokenizer = AutoTokenizer.from_pretrained("tokenization/expanded_tokenizer/")

    # Load and expand model
    model = load_model(fabric, model_config, initial_checkpoint_dir, tie_embeddings=False)
    expand_embeddings(fabric, model, tokenizer, target_tokenizer, tie_embeddings=False)

    # Save the updated model
    save_model(fabric, model, save_path)

    # Print embedding statistics
    input_embeddings = model.transformer.wte.weight.data
    avg_embedding = input_embeddings[:tokenizer.vocab_size].mean(dim=0)
    print_embedding_statistics(fabric, model, avg_embedding, tie_embeddings=False)


if __name__ == "__main__":
    main()
