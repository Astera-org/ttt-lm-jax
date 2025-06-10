TTT_IMPLEMENTATIONS=(
    "ttt_layer_nobias_l1l2reg"
    "ttt_layer_nobias_l1reg"
    "ttt_layer_nobias_l2reg"
    "ttt_layer_nobias_orthonorm"
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