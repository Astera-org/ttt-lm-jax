#!/bin/bash

#ttt_layer_nobias_fixed_v0

TTT_IMPLEMENTATIONS=(


 #   "custom.ttt_layer_nobias_bilevel_frobeniusv4"
    "custom.ttt_layer_nobias_bilevel_frobeniusv5"
    "custom.ttt_layer_nobias_bilevel_mixed"
   

    #"custom.ttt_layer_nobias_bilevel"

)

for TTT_IMPLEMENTATION in "${TTT_IMPLEMENTATIONS[@]}"; do
    echo "Running experiment with TTT implementation: ${TTT_IMPLEMENTATION}"
    ./scripts/ttt_linear/125m_ppl.sh "${TTT_IMPLEMENTATION}"
    if [ $? -ne 0 ]; then
        echo "Experiment with TTT implementation ${TTT_IMPLEMENTATION} failed."
     #   exit 1
    fi
    echo "Experiment with TTT implementation ${TTT_IMPLEMENTATION} completed successfully."
done    