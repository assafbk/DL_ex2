Next word prediction model based on multilayer LSTM\GRU.

train LSTM without dropout:
python3 main.py --phase train --epochs 13 --dropout_p 0 --lr 3.5 --show

train LSTM with dropout:
python3 main.py --phase train --epochs 35 --dropout_p 0.4 --lr 3.5 --gdp 25 --show

train GRU without dropout:
python3 main.py --phase train --epochs 13 --dropout_p 0 --lr 1.25 --use_gru --show

train GRU with dropout:
python3 main.py --phase train --epochs 45 --dropout_p 0.4 --lr 1.25 --weight_init 0.15 --gdp 35 --use_gru --show


test a model (e.g. lstm with dropout):
python3 main.py --phase test --model-path <path_to_model_weights>



Notes:
 --gdp is the gradient decay param - from the given iter, start decaying the grad by a factor of 0.5
 --show shows the convergence graph at the end of training
