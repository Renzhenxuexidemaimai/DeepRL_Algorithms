#!/usr/bin/env bash

envs=(HalfCheetah-v3 Hopper-v3 Walker2d-v3 Swimmer-v3 Ant-v3)

for (( i = 0; i < ${#envs[@]}; ++i )); do
    echo run ${envs[$i]} -----
    python -m PolicyGradient.PPO.main --env_id ${envs[$i]} --mini_batch True --model_path PolicyGradient/PPO/trained_models & python -m PolicyGradient.PPO.main --env_id ${envs[$i]} --mini_batch False --model_path PolicyGradient/PPO/trained_models
done

