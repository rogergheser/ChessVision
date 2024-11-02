sh ./eval.sh data/chess_games data/PGNS/chess_games models://start_pos_only
sh ./eval.sh data/chess_games data/PGNS/chess_games models://brown-white-img-only
sh ./eval.sh data/chess_games data/PGNS/chess_games models://transfer-learning
sh ./eval_postproc.sh data/chess_games data/PGNS/chess_games models://start_pos_only
sh ./eval_postproc.sh data/chess_games data/PGNS/chess_games models://brown-white-img-only
sh ./eval_postproc.sh data/chess_games data/PGNS/chess_games models://transfer-learning
