from rdkit import Chem

def ring_systems(mol):
    bonds = [b for b in mol.GetBonds() if b.IsInRing()]
    graph = {}
    for b in bonds:
        a, c = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        graph.setdefault(a, set()).add(c)
        graph.setdefault(c, set()).add(a)
    pending = set(graph)
    answer = []
    while pending:
        todo = [min(pending)]
        atoms = set()
        while todo:
            a = todo.pop()
            if a in atoms:
                continue
            atoms.add(a)
            todo.extend(graph[a] - atoms)
        pending -= atoms
        bs = [b.GetIdx() for b in bonds if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
        sub = Chem.PathToSubmol(mol, bs)
        code = Chem.MolToSmiles(sub, isomericSmiles=False)
        answer.append((code, sub, len(atoms), len(bs)))
    return answer
