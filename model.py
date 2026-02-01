import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utlis import EncoderDrug, EncoderCell, CCFM



class DCCFM(nn.Module):
    def __init__(self, cell_exp_dim, cell_mut_dim, cell_meth_dim, cell_path_dim, **config):
        super(DCCFM, self).__init__()
        drug_out_dim = config['drug']['drug_out_dim']
        cell_out_dim = config['cell']['cell_out_dim']
        mlp_in_dim = config['mlp']['mlp_in_dim']
        mlp_hidden_dim = config['mlp']['mlp_hidden_dim']
        self.drug_embedding = DrugEmbedding(drug_out_dim, use_ecfp=True, use_espf=True, use_pubchem=True)
        self.cell_embedding = CellEmbedding(cell_exp_dim, cell_mut_dim, cell_meth_dim, cell_path_dim, cell_out_dim,
                                            use_exp=True, use_mut=True, use_meth=True, use_path=True)

        self.DrugEncoder = EncoderDrug(128,512,4,0.3,0.3)  
        self.CellEncoder = EncoderCell(128,512,4,0.3,0.3) 
        self.DCAtt = CCFM(128,512,4,0.3,0.3)      
        
        self.mlp = MLP(mlp_in_dim, mlp_hidden_dim, out_dim=1)
        self.init_weights()

    def init_weights(self):
        self.drug_embedding.init_weights()
        self.cell_embedding.init_weights()
        self.mlp.init_weights()

    def forward(self, drug_data, cell_data):

        v_d0 = self.drug_embedding(drug_data)
        v_c0 = self.cell_embedding(cell_data[0], cell_data[1], cell_data[2], cell_data[3])

        v_d2, _ = self.DrugEncoder(v_d0, None)
        v_c2, _ = self.CellEncoder(v_c0, None)

        v_d3,v_c3,_,_ = self.DCAtt(v_d2,v_c2,None,None)
        predict = self.mlp(v_d3,v_c3)

        predict = torch.squeeze(predict)

        return predict, None

class DrugEmbedding(nn.Module):
    def __init__(self, out_dim, use_ecfp=True, use_espf=True, use_pubchem=True):
        super(DrugEmbedding, self).__init__()
        self.use_ecfp = use_ecfp
        self.use_espf = use_espf
        self.use_pubchem = use_pubchem
        
        # ecpf
        ecfp_layers1 = [
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
            nn.ReLU()
        ]
        self.ecfp1 = nn.Sequential(*ecfp_layers1)
        ecfp_layers2 = [
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU()
        ]
        self.ecfp2 = nn.Sequential(*ecfp_layers2)

        # espf
        espf_layers1 = [
            nn.Linear(2586, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
            nn.ReLU()
        ]
        self.espf1 = nn.Sequential(*espf_layers1)
        espf_layers2 = [
            nn.Linear(2586, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU()
        ]
        self.espf2 = nn.Sequential(*espf_layers2)

        # pubchem_fp
        pubchem_layers = [
            nn.Linear(881, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU()
        ]
        self.pubchem_fp = nn.Sequential(*pubchem_layers)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight) 

    def forward(self, finger_print):
        x_drug = []

        if self.use_ecfp:
            ecfp_f1 = self.ecfp1(finger_print[0])
            x_drug.append(ecfp_f1)
            ecfp_f2 = self.ecfp2(finger_print[0])
            x_drug.append(ecfp_f2)

        if self.use_espf:
            espf_f1 = self.espf1(finger_print[1])
            x_drug.append(espf_f1)
            espf_f2 = self.espf2(finger_print[1])
            x_drug.append(espf_f2)

        if self.use_pubchem:
            puchem_f = self.pubchem_fp(finger_print[2])
            x_drug.append(puchem_f)

        out = torch.stack(x_drug, dim=1)
        return out


class CellEmbedding(nn.Module):
    def __init__(self, exp_in_dim, mut_in_dim, meth_in_dim, path_in_dim, out_dim, use_exp=True, use_mut=True,
                 use_meth=True, use_path=True):
        super(CellEmbedding, self).__init__()
        self.use_exp = use_exp # 714
        self.use_mut = use_mut # 715
        self.use_meth = use_meth # 603
        self.use_path = use_path # 1283

        # exp_layer
        self.gexp_fc1 = nn.Linear(exp_in_dim, 256)
        self.gexp_bn = nn.BatchNorm1d(256)
        self.gexp_fc2 = nn.Linear(256, out_dim)

        # mut_layer
        self.mut_fc1 = nn.Linear(mut_in_dim, 256)
        self.mut_bn = nn.BatchNorm1d(256)
        self.mut_fc2 = nn.Linear(256, out_dim)

        # methy_layer
        self.methylation_fc1 = nn.Linear(meth_in_dim, 256)
        self.methylation_bn = nn.BatchNorm1d(256)
        self.methylation_fc2 = nn.Linear(256, out_dim)

        # pathway_layer1
        self.pathway_fc1 = nn.Linear(path_in_dim, 256)
        self.pathway_bn = nn.BatchNorm1d(256)
        self.pathway_fc2 = nn.Linear(256, out_dim)

        # pathway_layer2
        pathway__layers = [
                nn.Linear(path_in_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, out_dim),
                nn.ReLU(),
            ]
        self.pathway_layers = nn.Sequential(*pathway__layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight) 


    def forward(self, expression_data, mutation_data, methylation_data, pathway_data):
        x_cell = []
        #  expression representation
        if self.use_exp:
            x_exp = self.gexp_fc1(expression_data)
            x_exp = F.relu(self.gexp_bn(x_exp))
            x_exp = F.relu(self.gexp_fc2(x_exp))
            x_cell.append(x_exp)

        # mutation representation
        if self.use_mut:
            x_mut = self.mut_fc1(mutation_data)
            x_mut = F.relu(self.mut_bn(x_mut))
            x_mut = F.relu(self.mut_fc2(x_mut))
            x_cell.append(x_mut)

        # methylation representation
        if self.use_meth:
            x_meth = self.methylation_fc1(methylation_data)
            x_meth = F.relu(self.methylation_bn(x_meth))
            x_meth = F.relu(self.methylation_fc2(x_meth))
            x_cell.append(x_meth)

        # pathway representation
        if self.use_path:
            x_path = self.pathway_fc1(pathway_data)
            x_path = F.relu(self.pathway_bn(x_path))
            x_path = F.relu(self.pathway_fc2(x_path))
            x_cell.append(x_path)

            x_ = self.pathway_layers(pathway_data)
            x_cell.append(x_)

        x_cell = torch.stack(x_cell, dim=1)
        return x_cell

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU())    

        self.dense = nn.Sequential(
            nn.Linear(in_dim,hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim[0],hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1],out_dim)
            )    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight) 

    def forward(self, x, y):
        x = self.fc1(x)
        y = self.fc2(y)
        x = x.flatten(1)
        y = y.flatten(1)
        xy = torch.cat((x,y),dim=1)
        f = self.dense(xy)
        return f
