#!/usr/bin/env bash

#envs=(HalfCheetah-v3 Hopper-v3 Walker2d-v3 Swimmer-v3 Ant-v3)
envs=(BipedalWalker-v2)
max_iter=500
for (( i = 0; i < ${#envs[@]}; ++i )); do
      echo ============================================
      echo starting Env: ${envs[$i]} -----

      python -m PolicyGradient.PPO.main --env_id ${envs[$i]} --max_iter ${max_iter} --model_path PolicyGradient/PPO/trained_models

      echo finishing Env: ${envs[$i]} -----
      echo ============================================
done

