#!/usr/bin/env bash
cd ..

CUDA_VISIBLE_DEVICES=3 python policy_train.py --algo bc --domain thickener --level random --amount 100
