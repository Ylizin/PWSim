from .generic import GeneralEval
import logging
from torch.utils.data import DataLoader
from models.WMD.WMD import cal_WMD
from collections import defaultdict
import numpy as np
from configs.globalConfigs import args as _args

class WMDEval(GeneralEval):
    def __init__(self,args=_args):
        super().__init__()
        self.init_data_loader()
    def init_data_loader(self):
        self.test_record = defaultdict(list)
        for k,v in self.data_set.test_pos:
            self.test_record[k.item()].append(v.item())

    def train(self):
        p,r = [],[]
        for test_id,pos_ids in self.test_record.items():
            if len(pos_ids)== 1:
                # pass case with only diagonal 
                continue
            test_t = self.data_set.words_df.loc[test_id,'rs']
            sims = []
            for _a_t in self.data_set.words_df['rs']:
                sims.append(cal_WMD(test_t,_a_t.strip().split()))
            sims.sort()
            _p,_r = self.get_metrics(sims,pos_ids,self.args.topk) #s
            p.append(_p)
            r.append(_r)
        print('ave_pre:{}\tave_rec:{}'.format(np.mean(p),np.mean(r)))