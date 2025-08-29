How to get the model before to run the code:

- download microsoft/Phi-4-mini-instruct-onnx (https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx) `huggingface-cli download microsoft/Phi-4-mini-instruct-onnx --local-dir ./`

- open the tokenizer.son file and change special from false to true for the tokens `<|tool_call|>` and `<|/tool_call|>`

- open the genai_config.json file and be sure that `provider_options` has en empty list as a value

- update the modelPath in the code
  