# Persuasive Argument Analysis

This repository contains code that I used to perform a replication of [Tan et al.]([url](https://chenhaot.com/pubs/winning-arguments.pdf)https://chenhaot.com/pubs/winning-arguments.pdf) with GPT 3.5 

Please find analysis in **analysis.md**

### Getting Started

1. create a `.env`, copy the `env.example`, and add your OpenAI key
2.  run `pip install -r requirements.txt`
3.  download the data from [https://chenhaot.com/data/cmv/cmv.tar.bz2]([url](https://chenhaot.com/data/cmv/cmv.tar.bz2))
4.  drop the cmv file into the directory root 

All code was run in the command line, within IPython


## OP Malleability
code used to perform the first task: predict and explain OP malleability can be found in **op_analysis.py**

## Pair Analysis
code used to perform the second task: compare a pair of arguments, one that changed an OP's mind versus one that did not can be found in **pair_analysis.py**
