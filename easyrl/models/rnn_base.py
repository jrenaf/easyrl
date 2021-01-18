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
                          batch_first=False)
        self.fcs = nn.Sequential(
            nn.Linear(in_features=rnn_features, out_features=rnn_features),
            nn.ELU()
        )

    def forward(self, x=None, hidden_state=None, done=None):
        #print(x['state'].shape, x['ob'].shape)
        if hidden_state is not None: print('hidden state', hidden_state.shape)
        #print(x)

        #x['state'] = x['state'].view(b * t, *x['state'].shape[1:])
        if len(x['ob'].shape) > 3:
          x['ob'] = x['ob'].permute(1, 0, 2, 3)
          x['state'] = x['state'].permute(1, 0, 2)
          #print(x['ob'].shape)

          b = x['ob'].shape[0]
          t = x['ob'].shape[1]
          print(x['ob'].shape)
          print(b*t, 1, x['ob'].shape[2:])
          x['ob'] = x['ob'].contiguous().view(b*t, 1, *x['ob'].shape[2:])
          print(x['ob'].shape)
          x['state'] = x['state'].contiguous().view(b*t, 1, *x['state'].shape[2:])

        else:
          t = x['ob'].shape[0]
          b = 1

        print('b', b, 't', t)
        obs_feature = self.body(x)
        obs_feature = obs_feature.view(b, t, *obs_feature.shape[1:])

        if done is not None: print(done.shape)
        if done is not None: done = done.permute(1, 0)
        if self.training:
            done_ts = (done == 1).any(dim=0).nonzero(as_tuple=False).squeeze(dim=-1).cpu().numpy() + 1
            done_ts = done_ts.tolist()
            done_ts = [0] + done_ts
            if done_ts[-1] != t:
                done_ts = done_ts + [t]
            rnn_features = []
            
            print('done_ts', done_ts)
            #print('done_ts', done_ts)
            for idx in range(len(done_ts) - 1):
                sid = done_ts[idx]
                eid = done_ts[idx + 1]
                if hidden_state is not None and sid > 0:
                    hidden_state = hidden_state * (1 - done[sid-1, :]).view(-1, 1, 1)
                print('rnnbase2', sid, eid, obs_feature[sid:eid, :].shape, hidden_state.shape)
                #print('hidden_state2', hidden_state.shape)
                rfeatures, hidden_state = self.gru(obs_feature[sid:eid, :],
                                                   hidden_state)
                rnn_features.append(rfeatures)
            rnn_features = torch.cat(rnn_features, dim=1)
        else:
            #if hidden_state is not None: print('rnnbase in_hidden_state', hidden_state.shape)
            #else: print('rnnbase in_hidden_state', None)
            print("in_obs_feature", obs_feature.shape)
            rnn_features, hidden_state = self.gru(obs_feature,
                                                  hidden_state)
            #print('rnnbase out_hidden_state', hidden_state.shape)
        out = self.fcs(rnn_features)
        return out, hidden_state
