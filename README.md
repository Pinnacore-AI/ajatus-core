# AjatusCore

**AjatusCore** is the heart of the Ajatuskumppani project. It contains the core language model (LLM), fine-tuning scripts, and the inference engine.

## Features

-   **Base Model**: Built upon [Mistral 7B Instruct](https://mistral.ai/news/announcing-mistral-7b/)
-   **Fine-tuning**: Includes scripts for fine-tuning the model with custom data.
-   **Inference**: Optimized for performance with [vLLM](https://github.com/vllm-project/vllm) and [Ollama](https://github.com/ollama/ollama).
-   **GGUF Support**: Ready for on-device AI with GGUF.

## Getting Started

### Prerequisites

-   Python 3.11+
-   PyTorch 2.0+
-   CUDA 11.8+

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/pinnacore-ai/ajatuskumppani.git
    cd ajatuskumppani/ajatus-core
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### Inference

To run inference with the model, use the following command:

```bash
python inference/run.py --prompt "Hello, world!"
```

#### Fine-tuning

To fine-tune the model with your own data, see the instructions in the `training` directory.

## License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.

