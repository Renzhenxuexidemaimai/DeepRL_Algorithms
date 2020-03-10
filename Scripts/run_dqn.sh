#!/usr/bin/env bash

envs=(CartPole-v1 MountainCar-v0 Acrobot-v1 LunarLander-v2)
seeds=10
algs=(DQN DoubleDQN DuelingDQN)
for (( k = 0; k < ${#algs[@]}; ++k )); do
    alg=${algs[$k]}
    for (( j = 1; j <= seeds; ++j )); do
        for (( i = 0; i < ${#envs[@]}; ++i )); do
            echo ============================================
            echo Algo: ${alg}, starting Env: ${envs[$i]} ----- Exp_id $j

            python -m DQN.${alg}.main --env_id ${envs[$i]} --model_path DQN/${alg}/trained_models --seed $j

            echo Algo: ${alg}, finishing Env: ${envs[$i]} ----- Exp_id $j
            echo ============================================
        done
    done
done
