import torch
import torch.nn as nn


class RNNBase(nn.Module):
    def __init__(self,
                 body_net,
                 rnn_features=128,
                 in_features=128,
                 rnn_layers=1,
                 ):
        super().__init__()
        self.body = body_net
        self.gru = nn.GRU(input_size=in_features,
                          hidden_size=rnn_features,
                          num_layers=rnn_layers,
                          batch_first=True)
        self.fcs = nn.Sequential(
            nn.Linear(in_features=rnn_features, out_features=rnn_features),
            nn.ELU()
        )

    def forward(self, x=None, hidden_state=None, done=None):
        print(x['state'].shape, x['ob'].shape)
        #print(x)
        if len(x['state'].shape) > 2:
          b = x['state'].shape[0]
          t = x['state'].shape[1]#1#x['state'].shape[1]
          x['state'] = x['state'].view(b * t, *x['state'].shape[2:])
          print(b, t, *x['ob'].shape)
          x['ob'] = x['ob'].view(b * t, *x['ob'].shape[2:])
        else:
          b = x['state'].shape[0]
          t = 1

        obs_feature = self.body(x)
        #print(obs_feature.shape)
        #if len(x['state'].shape) > 2:
        obs_feature = obs_feature.view(b, t, *obs_feature.shape[1:])

        if self.training:
            done_ts = (done == 1).any(dim=0).nonzero(as_tuple=False).squeeze(dim=-1).cpu().numpy() + 1
            done_ts = done_ts.tolist()
            done_ts = [0] + done_ts
            if done_ts[-1] != t:
                done_ts = done_ts + [t]
            rnn_features = []

            for idx in range(len(done_ts) - 1):
                sid = done_ts[idx]
                eid = done_ts[idx + 1]
                if hidden_state is not None and sid > 0:
                    hidden_state = hidden_state * (1 - done[:, sid-1]).view(1, -1, 1)
                rfeatures, hidden_state = self.gru(obs_feature[:, sid:eid],
                                                   hidden_state)
                rnn_features.append(rfeatures)
            rnn_features = torch.cat(rnn_features, dim=1)
        else:
            rnn_features, hidden_state = self.gru(obs_feature,
                                                  hidden_state)
        out = self.fcs(rnn_features)
        print('out.shape', out.shape)
        return out, hidden_state
