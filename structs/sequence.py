import torch
from utils import get_positional_encoding
from .struct import Struct

class SequenceStruct(Struct):

  def __init__(self, data):
    self.data = data

  def __str__(self):
    return " ".join([str(x) for x in self.data])

  def flatten(self):
    return self.data

  def map(self, f):
    return SequenceStruct([f(x) for x in self.data])

  def get_pos_embedding(self, embed_dim, params):
    return SequenceStruct(get_positional_encoding(embed_dim, self.size()) * ((embed_dim / 2) ** -0.5))
#    if len(params) == 0:
#      return SequenceStruct(get_positional_encoding(embed_dim, self.size()) * ((embed_dim / 2) ** -0.5))
#    else:
#      pos_seq = params[0]
#      return SequenceStruct(pos_seq[:self.size(), :])

def parse(s):
  return SequenceStruct(s.strip().split())

def get_params(args):
  return {}
#  if args.learned_pos:
#    embed_dim = args.embed_dim
#    max_len = args.max_train_length # ?
#    pos_seq = torch.Tensor(max_len, embed_dim)
#    torch.nn.init.normal_(pos_seq, mean=0, std=embed_dim ** -0.5)
#    return {"pos_seq":pos_seq}
#  else:
#    return {}
