# IntPiece: efficient compress algorithm for integer sequence

## What is IntPiece and What i do

Int piece is an efficient integer-sequence compression algorithm. 
It accepts an integer-sequence as input and outputs another sequence of integers. 
It reduces the length of dequence but increasing the vocabulary size.
As its name suggests, it is inherited from Su's BytePiece(
[more details can be found here](https://github.com/bojone/bytepiece/blob/main/README_en.md)). 
The core algorithm of IntPiece is almost the same as BytePiece. 
I just changed a few insignificant lines so that the core code can accept integer-sequences 
instead of string or chars as input and output.
Because of the powerful performance of BytePiece itself, IntPiece also has some good features, 
such as Lossless reconstruction/High compression rate/Training-friendly.


## Why we need IntPiece
As for most autoregressive language models, 
the cost of algorithm complexity increase linearly with vocabulary size, 
but quadratic with the sequence length. 
Shortening the sequence length is critical for the large language models.
Therefore, we prefer to a larger vocabulary but shorter sequence. 
At the same time, our sequence does not consist of explicit words, but some implicit tokens.
For example, 
[Amazon's base-tts](https://assets.amazon.science/6e/82/1d037a4243c9a6cf4169895482d5/base-tts-lessons-from-building-a-billion-parameter-text-to-speech-model-on-100k-hours-of-data.pdf) 
system maps audio into integer tokens and then 
uses llm to predict these tokens one by one. 
By using a similar compression algorithm(Byte-pair), 40% length compression is obtained(can be seen in their papers).

## Installation

you can also see document of BytePieces, but actually you just need to run 

```bash
pip uninstall pyahocorasick

AHOCORASICK_BYTES=1 pip install git+https://github.com/WojciechMula/pyahocorasick.git
```
to get pyahocorasick for BYTE version. And run (Optional)

```bash
python setup.py build_ext --inplace
```
to build c++ core function. 
The c++ functions will be faster, but not much. The project also contains a python version. 

## Usage

It is the same with BytePiece: 
All source code of BytePiece is actually in a single file, 
including `Trainer` and `Tokenizer` two classes, corresponding to training and tokenization(inferece) respectively.

you can run 
```bash
python intpiece --data_path xxx.json --model_path xx.model --train
python intpiece --data_path xxx.json --model_path xx.model 
```
to train or inference(evaluation). 

And also, i put some important parameters into arguments, such as order or min_count. And you can use -h to see more details.
About dataset format, you can see `Corpus` class. It is very easy to use.


# Thanks

Finally, I would like to express my gratitude to Su and its project 
again, especially for his perseverance in open source.



