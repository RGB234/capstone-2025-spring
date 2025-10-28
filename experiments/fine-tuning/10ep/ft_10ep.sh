#!/bin/bash

PYTHONPATH=../../custom_bgem3

# sh 실행
echo "Running bgem3_10ep.sh..."
bash bgem3_10ep.sh
if [ $? -ne 0 ]; then
    echo "bgem3_10ep.sh failed. Exiting."
    exit 1
fi

# sh 실행
echo "Running ctrl_10ep.sh..."
bash ctrl_10ep.sh
if [ $? -ne 0 ]; then
    echo "ctrl_10ep.sh failed. Exiting."
    exit 1
fi

# sh 실행
echo "Running saq_10ep.sh..."
bash saq_10ep.sh
if [ $? -ne 0 ]; then
    echo "saq_10ep.sh failed. Exiting."
    exit 1
fi

echo "All scripts executed successfully!"
