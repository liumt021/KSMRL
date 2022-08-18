import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import numpy as np

def cal_dis_info(mol, max_size=-1):
    num_atoms = mol.GetNumAtoms()
    mol = Chem.RemoveHs(mol)
    mol_addHs = Chem.AddHs(mol)

    try:
        AllChem.EmbedMultipleConfs(mol_addHs, numConfs=3, randomSeed=2021, clearConfs=True, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(mol_addHs, numThreads=0)
        eng_list = [i[1] if i[0] == 0 else 99999999 for i in res]
        min_index = eng_list.index(min(eng_list))
        xyz_index = mol_addHs.GetConformers()[min_index].GetPositions()[:num_atoms]
        dts_adj = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        for i in range(num_atoms):
            for j in range(i, num_atoms):
                dts_adj[i][j] = np.sqrt(np.sum(np.square(xyz_index[i] - xyz_index[j])))
                dts_adj[j][i] = dts_adj[i][j]
        dis2zero = np.sqrt(np.sum(np.square(xyz_index), axis=1))

    except:
        return None, None

    tem_dis_feat = dis2zero

    if max_size < 0:
        return np.array(tem_dis_feat, dtype=np.float32), dts_adj
    elif max_size >= num_atoms:

        dts_array = np.zeros(max_size, dtype=np.float32)
        dts_adj_t = np.zeros((max_size, max_size), dtype=np.float32)
        dts_array[:num_atoms] = np.array(tem_dis_feat, dtype=np.float32)
        dts_adj_t[:num_atoms, :num_atoms] = dts_adj

        return dts_array, dts_adj_t
    else:
        raise ValueError('error')



