conda activate chessvision

# DATADIR='data/chess_games'
# PNG_DIR='data/PGNS/chess_games'
# MODEL_DIR='models://transfer_learning'
DATADIR=$1
PGN_DIR=$2
MODEL_DIR=$3 # pass base model directory
EVAL_ONLY=True
MAX_RETRIES=6
BOARD_SIZE=400
MAX_QUEUE_LEN=40
MAX_ERRORS=64
RESDIR="results"

CMD="python main.py --datadir $DATADIR --pgndir $PGN_DIR --model_dir $MODEL_DIR --max_retries $MAX_RETRIES --board_size $BOARD_SIZE"
CMD="$CMD --max_queue_len $MAX_QUEUE_LEN --max_errors $MAX_ERRORS --resdir $RESDIR"
CMD="$CMD --nopostprocess" #Baseline uses no postprocessing
if [ "$EVAL_ONLY" = "True" ]; then
    CMD="$CMD --eval_only"
fi

$CMD