#!/usr/bin/env bash
cd ..

CUDA_VISIBLE_DEVICES=0 python benchmark\launch_algo.py --domain sp --algo bc
