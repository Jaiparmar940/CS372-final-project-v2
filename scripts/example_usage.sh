#!/bin/bash
# Example usage of diagnostic scripts

echo "=== Training Diagnostics Usage Examples ==="
echo ""

echo "1. Train an agent (this will create CSV logs):"
echo "   python training/train_tabular.py --episodes 100"
echo ""

echo "2. Analyze the training log:"
echo "   python scripts/analyze_training.py --log logs/tabular_q_training_log.csv --window 100"
echo ""

echo "3. Analyze and save plot:"
echo "   python scripts/analyze_training.py --log logs/tabular_q_training_log.csv --window 100 --save_plot plots/analysis.png"
echo ""

echo "4. Evaluate a trained agent:"
echo "   python scripts/evaluate_agent.py --algorithm tabular_q --checkpoint checkpoints/tabular_q/tabular_q_best.pkl --num_episodes 50"
echo ""

echo "=== Quick Test ==="
echo "To generate a log file, first run training:"
echo "   python training/train_tabular.py --episodes 10"
echo ""
echo "Then analyze it:"
echo "   python scripts/analyze_training.py --log logs/tabular_q_training_log.csv"

