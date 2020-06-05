import all_constants as ac
import structs as struct


base = {
    'embed_dim': 512,
    'ff_dim': 2048,
    'num_enc_layers': 6,
    'num_dec_layers': 6,
    'num_heads': 8,

    # architecture,
    'use_bias': True,
    'fix_norm': True,
    'scnorm': True,
    'mask_logit': True,
    'pre_act': True,

    'clip_grad': 1.0,
    'lr_scheduler': ac.NO_WU,
    'warmup_steps': 8000,
    'lr': 3e-4,
    'lr_scale': 1.,
    'lr_decay': 0.8,
    'stop_lr': 5e-5,
    'patience': 3,
    'embed_scale_lr': 0.03,
    'alpha': 0.7,
    'label_smoothing': 0.1,
    'batch_size': 4096,
    'epoch_size': 1000,
    'max_epochs': 200,
    'dropout': 0.3,
    'att_dropout': 0.3,
    'ff_dropout': 0.3,
    'word_dropout': 0.1,

    # Penalize position embeddings that are too big
    'pos_norm_penalty': 5e-3, # set to 0 if you want no penalty
    'pos_norm_scale': (lambda args: (args.embed_dim / 2) ** 0.5),

    # Module
    'struct': struct.sequence,

    # Decoding
    'beam_size': 4,
    'beam_alpha': 0.6,
}

fun2com = {'struct': struct.tree, 'batch_size':2048, 'epoch_size':5000}
fun2comb = {'struct': struct.tree2, 'batch_size':2048, 'epoch_size':5000}
fun2com_seq = {'struct': struct.sequence, 'batch_size':2048, 'epoch_size':5000}
