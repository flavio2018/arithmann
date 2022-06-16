from model.archies.CustomLSTM import build_lstm
from model.archies.CustomMLP import build_mlp
from model.archies.CustomRNN import build_rnn
from model.dntm.DynamicNeuralTuringMachine import build_dntm


def build_model(model_conf, device):
    if model_conf.name == 'dntm':
        return build_dntm(model_conf, device)
    elif model_conf.name == 'mlp':
        return build_mlp(model_conf, device)
    elif model_conf.name == 'rnn':
        return build_rnn(model_conf, device)
    else:
        return build_lstm(model_conf, device)
