python main.py ^
--ai minimax ^
--depth 2 ^
--init_n 40 ^
--n 225 ^
--eval_legal_drops 0 ^
--mode method1 ^
    ^
--n_play 2000 ^
--k 0.5 ^
--load_path ./RL/models/policy_Simple__best_epoch8350.pth ^
--internal_model Simple ^
--device cuda ^