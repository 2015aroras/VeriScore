# VeriScore

[![arxiv](https://img.shields.io/badge/arXiv-2406.19276-b31b1b.svg)](https://arxiv.org/abs/2406.19276)

This repository contains code for our VeriScore factuality metric, which is a pipelined approach with three steps: (1) `claim extraction`, (2) `evidence retrieval`, and (3) `claim verification`. This package can be run with either closed-source LLMs (requires OpenAI/Anthropic API key), or with fine-tuned models for [claim extraction](https://huggingface.co/SYX/mistral_based_claim_extractor) and [verification](https://huggingface.co/SYX/llama3_based_claim_verifier) that you can download and use. The evidence retrieval step is performed with Google Search via the Serper API, so you will need to get a Serper API key [here](https://serper.dev/) to use VeriScore.

**Please see our [Colab notebook](https://colab.research.google.com/drive/14cJsd5xu-paXb1ld72kF3WA97qzcyEn1?authuser=1#scrollTo=uhfwyPWBUojR) for a demo!** 

**You can choose between prompting OpenAI/Anthropic models or using our fine-tuned models via the `model_name` option. If you specify the path to the checkpoint of a fine-tuned model, it will automatically access the local model for the inference. If the model name is not in this format, it will use an API call instead.**

## Installation
1. Make a new Python 3.9+ environment using `virtualenv` or `conda`.
2. Install `veriscore` pacakge using `pip`
3. Download `en_core_web_sm` using `spacy` library
4. Our code supports inference using fine-tuned models based on the Unsloth library. To use this feature, you need to install the [Unsloth](https://github.com/unslothai/unsloth) library.
```
pip install --upgrade veriscore
python -m spacy download en_core_web_sm
```

## Set up environment before running code
1. Download `prompt` folder that contains txt files of the prompt templates (see the `prompt` folder in this repository)
2. Add an OpenAI or Claude API key to an environment variable in your `bash` for the prompting approach
```
export OPENAI_API_KEY_PERSONAL={your_openai_api_key}
export CLAUDE_API_KEY={your_claude_api_key}
```
3. Set SERPER API key to an environment variable in your `bash` for evidence retrieval
```
export SERPER_KEY_PRIVATE={your_serper_api_key}
```
4. For the prompt-based approach, you need to set `data_dir/demos/` with [few-shot examples](https://github.com/Yixiao-Song/VeriScore/blob/main/data/demos/few_shot_examples.jsonl).

## Running VeriScore in a command line
This is an end-to-end pipeline for running VeriScore.
```
 python3 -m veriscore.veriscore --data_dir {data_dir} --input_file {input_file} --model_name_extraction {model_name_extraction} --model_name_verification {model_name_verification}
```
* `data_dir`: Directory containing input data. `./data` by default.
* `input_file`: Name of the input data file. It should be in the `jsonl` format where each line contains
    * `question`: A query to prompt a language model for an output
    * `response`: An output generated by the language model given the `question`
    * `model`: Name of the model that generated the response
    * `prompt_source`: Name of the dataset from where the `question` is from (e.g., FreshQA)
* `model_name_extraction`: Name of the model used for claim extraction; `gpt-4-0125-preview` by default.
* `model_name_verification`: Name of the model used for claim verification; `gpt-4o` by default.
* `use_external_extraction_model`: If specified, it uses your custom model instead of the one from the API call. We use Unsloth for the fine-tuned model. False by default.
* `use_external_verification_model`: If specified, it uses your custom model instead of the one from the API call. We use Unsloth for the fine-tuned model. False by default.

Other optional flags:

* `output_dir`: Directory for saving ouptut data. `./data` by default.
* `cache_dir`: Directory for saving cache data. `./data/cache` by default.
* `label_n`: This is the type of label for claim verification. It could be `2` (binary) or `3` (ternary)
    * `2`: `supported` and `unsupported`
    * `3`: `supported`, `contradicted`, and `inconclusive`
* `search_res_num`: A Hyperparameter for number of search result. `10` by default.

Saving output: 
`input_file_name` is file name removed `jsonl` from `—-input_file`
* `extracted claims` will be saved to `output_dir/claims_{input_file_name}.jsonl`.
* `searched evidence` will be saved to `output_dir/evidence_{input_file_name}.jsonl`.
* `verified claims` will be saved to `output_dir/model_output/verification_{input_file_name}.jsonl`.

## Running individual components of the pipeline using the command line
`Claim extraction`:
```
 python3 -m veriscore.extract_claims --data_dir {data_dir} --input_file {input_file} --model_name {model_name} 
```
* `input_file`: Name of input data file. It should be `jsonl` format where each line contains
    * `question`: A query to prompt a language model for an output
    * `response`: An output generated by the language model given the `question`
    * `model`: Name of the model that generated the response
    * `prompt_source`: Name of the dataset from where the `question` is from (e.g., FreshQA)
* `model_name`: Name of model used for claim extraction. `gpt-4-0125-preview` by default.
* `use_external_model`: If specified, it uses your custom model instead of using API calls. We use Unsloth for the fine-tuned models. False by default.

output:
```dictionary
 {
  "question": question.strip(),
  "prompt_source": prompt_source,
  "response": response.strip(),
  "prompt_tok_cnt": prompt_tok_cnt,
  "response_tok_cnt": response_tok_cnt,
  "model": model,
  "abstained": False,  
  "claim_list": list of claims for each snippet,
  "all_claims": list of all claims
 }
```
`Evidence searching`:
```
 python3 -m veriscore.retrieve_evidence --data_dir {data_dir} --input_file {input_file}
```
* `input_file`: Name of input data file. It should be in the `jsonl` format where each line contains the keys of the output dictionary from the `Claim extraction`.
output:
```dictionary
 {
  ...
  "claim_snippets_dict": dictionary for claim and list of searched evidence. each evidence have dictionary of {"title": title, "snippet": snippet, "link": link}
 }
```

`Claim verification`:
```
 python3 -m veriscore.verify_claims --data_dir {data_dir} --input_file {input_file} --model_name {model_name}
```
* `input_file`: Name of input data file. It should be `jsonl` format where each line contains the keys of the output dictionary from the `Evidence searching`.
* `use_external_model`: If specified, it uses your custom model instead model used by API call. We use Unsloth to inference from specified model. False by default.

output:
```dictionary
 {
  ...
  "claim_verification_result":[
    {
     "claim": claim text,
     "search_results": concatenated search result
     "verification_result": verification label
 }
```
