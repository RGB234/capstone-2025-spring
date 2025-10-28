#!/bin/bash

PYTHONPATH=../../custom_bgem3

# sh 실행
echo "Running bgem3_5ep.sh..."
bash bgem3_5ep.sh
if [ $? -ne 0 ]; then
    echo "bgem3_5ep.sh failed. Exiting."
    exit 1
fi

# sh 실행
echo "Running ctrl_5ep.sh..."
bash ctrl_5ep.sh
if [ $? -ne 0 ]; then
    echo "ctrl_5ep.sh failed. Exiting."
    exit 1
fi

# sh 실행
echo "Running saq_5ep.sh..."
bash saq_5ep.sh
if [ $? -ne 0 ]; then
    echo "saq_5ep.sh failed. Exiting."
    exit 1
fi

echo "All scripts executed successfully!"
