import numpy as np
import matplotlib.pyplot as plt
import RDF_Package


def read_xyz_file(path):
    aType = []
    coords = np.zeros((1, 3))
    with open(path, 'r') as f:
        line_data = f.readline()
        while line_data is not '':
            try:
                int(line_data)
                line_data = f.readline()
                line_data = f.readline()
                continue
            except:
                pass
            line_data = line_data.split()
            aType.append(atom2number(line_data[0]))
            for i, item in enumerate(line_data[1:]):
                line_data[i+1] = float(item)
            coords = np.vstack((coords, np.array(line_data[1:])))
            line_data = f.readline()
        coords = coords[1:, :]

    mn = np.array([[np.amax(coords[:, 0]) - np.amin(coords[:, 0]), 0, 0],
                   [0, np.amax(coords[:, 1]) - np.amin(coords[:, 1]), 0],
                   [0, 0, np.amax(coords[:, 2]) - np.amin(coords[:, 2])]])
    return aType, coords, mn


def compute_pdf(coords, aType, atomKinds, dr, Nz, Nr, Npdf, Mm, s, f):
    Constrain = dr * Nr
    rMax = Constrain
    PDF = np.zeros((Nr, Npdf))
    phai = np.zeros((np.size(s), Npdf))
    s[0] = 0.0000001
    pdfCount = -1
    for i in np.arange(Nz):
        for j in np.arange(i, Nz):
            pdfCount = pdfCount + 1
            index1 = np.where(aType == atomKinds[i])
            p1 = coords[index1[0]]
            index2 = np.where(aType == atomKinds[j])
            p2 = coords[index2[0]]
            atom1_num = np.shape(p1)[0]
            atom2_num = np.shape(p2)[0]
            for m in np.arange(atom1_num):
                if atomKinds[i] == atomKinds[j]:
                    l = m
                else:
                    l = 0

                for n in np.arange(l, atom2_num):
                    d = np.sqrt(np.sum(np.power(p1[m, :]-p2[n, :], 2)))
                    if 0 < d < rMax:
                        rol = abs(int(d / dr+0.5))
                        PDF[rol-1, pdfCount] += 1
            fsqr = np.multiply(f[:, i], f[:, j])
            for k in np.arange(1, Nr):
                phai[:, pdfCount] += PDF[k, pdfCount] * np.multiply(fsqr, np.sinc(2*s*dr*k))
    return PDF, 2 * phai


def count_pdf_function(coords, aType, Mm, bin=1, dr=0.02, max_angle=10, califactor=0.00111625):
    rMax = np.linalg.norm(np.diag(Mm))
    r = np.arange(0, rMax+dr, dr)
    Nr = np.size(r)
    atom_kinds = np.unique(aType)
    atom_number = np.size(aType)
    Nz = np.size(atom_kinds)
    Npdf = int(Nz + Nz * (Nz - 1) / 2)
    califactor = califactor*bin
    n = int(max_angle/califactor)
    s = np.arange(n)*califactor
    s_sqr = np.power(s, 2)
    f = np.zeros((len(s), Nz))
    fmean = np.zeros(len(s))
    fsqrmean = np.zeros(len(s))

    for i, item in enumerate(np.arange(Nz)):
        fit_para = RDF_Package.ref_atom_para(atom_kinds[item])
        f[:, i] = (fit_para[0, 0] * np.power((s_sqr + fit_para[0, 1]), -1) +
            fit_para[0, 2] * np.power((s_sqr + fit_para[0, 3]), -1) +
            fit_para[1, 0] * np.power((s_sqr + fit_para[1, 1]), -1) +
            fit_para[1, 2] * np.exp(-1 * fit_para[1, 3] * s_sqr) +
            fit_para[2, 0] * np.exp(-1 * fit_para[2, 1] * s_sqr) +
            fit_para[2, 2] * np.exp(-1 * fit_para[2, 3] * s_sqr))
        Composition = np.size(np.where(aType == atom_kinds[item]))/atom_number
        fmean = fmean + Composition * f[:, i]
        fsqrmean = fsqrmean + Composition * np.power(f[:, i], 2)

    fmeansqr = np.power(fmean, 2)  # calculation of <f>^2
    PDF, phai = compute_pdf(coords, aType, atom_kinds, dr, Nz, Nr, Npdf, Mm, s, f)
    difstr = np.sum(phai, 1)/atom_number  # diffraction intensity caused by structure only
    diftot = np.sum(phai, 1) + atom_number*fsqrmean  # total diffraction intensity
    return phai, diftot, fmeansqr, atom_number, s, PDF, atom_kinds


def Gc_rdf(atom_number, s, diff, fmeansqr, maxrange=10, step=0.01,
           be=0.3, en=3, E=0.25, H=0.015, norm_factor=1):
    be = int(be/(s[1] - s[0]))
    en = int(en/(s[1] - s[0]))
    r = np.arange(0, maxrange, step)
    diffs = diff * 2
    damp = np.exp(-1 * E * np.power(s, 2))
    Npdf = np.shape(diff)[1]
    G = np.zeros((np.size(r), Npdf))
    for i in np.arange(Npdf):
        diffs[:, i] = np.divide(np.multiply(diff[:, i], s), fmeansqr*atom_number)*norm_factor
        diffs[:, i] = np.multiply(diffs[:, i], damp)
        G[:, i] = np.multiply(ftg(r, s, diffs[:, i], be-1, en),
                              np.exp(-1 * H * np.power(r, 2)))
        Gtot = np.sum(G, 1)
    return G, Gtot, r, diffs

def ftg(r, s, dif, be, en):
    ds = s[1] - s[0]
    rn = np.size(r)
    Gp = np.zeros(rn)
    for i in np.arange(rn):
        dif_temp = dif[be:en+1]
        s_temp = s[be:en+1]
        sin_temp = np.sin(s_temp * r[i] * 2 * np.pi)
        Gp[i] = np.sum(np.multiply(dif_temp, sin_temp)) * ds * 8 * np.pi
    return Gp

def simulation_with_xyz(path, parameter):
    try:
        califactor = parameter['cali']
        maxangle = parameter['maxangle']
        pdf_damp = parameter['pdf_damp']
        window_start = parameter['window_start']
        window_end = parameter['window_end']
        max_range = parameter['max_range']
        step_length = parameter['step_length']
        rdf_damp = parameter['rdf_damp']

    except KeyError:
        print('Error: input data Error')
        return

    aType, coords, Mm = read_xyz_file(path)
    phai, diftot, fmeansqr, atom_number, s, PDF, atom_kinds = count_pdf_function(coords, aType, Mm,
                                                                                 max_angle=maxangle,
                                                                                 califactor=califactor)
    recal_para = {'atom_number': atom_number,
                  's': s,
                  'phai': phai,
                  'fmeansqr': fmeansqr,
                  'PDF': PDF,
                  'atom_kinds': atom_kinds,
                  'diftot': diftot}

    G, Gtot, r, diffs = Gc_rdf(atom_number, s, phai, fmeansqr, maxrange=max_range, step=step_length,
                               be=window_start, en=window_end,
                               E=pdf_damp, H=rdf_damp)
    atom = []
    for item in atom_kinds:
        atom.append(atom2number(item, mode=2))
    return PDF, G, Gtot, r, atom, s, diffs, diftot, recal_para


def recal_damp(recal_para, parameter):
    try:
        pdf_damp = parameter['pdf_damp']
        window_start = parameter['window_start']
        window_end = parameter['window_end']
        max_range = parameter['max_range']
        step_length = parameter['step_length']
        rdf_damp = parameter['rdf_damp']

    except KeyError:
        print('Error: input data Error')
        return
    atom_number = recal_para['atom_number']
    s = recal_para['s']
    phai = recal_para['phai']
    fmeansqr = recal_para['fmeansqr']
    G, Gtot, r, diffs = Gc_rdf(atom_number, s, phai, fmeansqr, maxrange=max_range, step=step_length,
                               be=window_start, en=window_end,
                               E=pdf_damp, H=rdf_damp)
    PDF=recal_para['PDF']
    atom_kinds = recal_para['atom_kinds']
    diftot = recal_para['diftot']
    atom = []
    for item in atom_kinds:
        atom.append(atom2number(item, mode=2))
    return PDF, G, Gtot, r, atom, s, diffs, diftot, recal_para

def atom2number(atom_string, mode=1):
    atom_dict = {
        'H': 1,
        'He': 2,
        'Li': 3,
        'Be': 4,
        'B': 5,
        'C': 6,
        'N': 7,
        'O': 8,
        'F': 9,
        'Ne': 10,
        'Na': 11,
        'Mg': 12,
        'Al': 13,
        'Si': 14,
        'P': 15,
        'S': 16,
        'Cl': 17,
        'Ar': 18,
        'K': 19,
        'Ca': 20,
        'Sc': 21,
        'Ti': 22,
        'V': 23,
        'Cr': 24,
        'Mn': 25,
        'Fe': 26,
        'Co': 27,
        'Ni': 28,
        'Cu': 29,
        'Zn': 30,
        'Ga': 31,
        'Ge': 32,
        'As': 33,
        'Se': 34,
        'Br': 35,
        'Kr': 36,
        'Rb': 37,
        'Sr': 38,
        'Y': 39,
        'Zr': 40,
        'Nb': 41,
        'Mo': 42,
        'Tc': 43,
        'Ru': 44,
        'Rh': 45,
        'Pd': 46,
        'Ag': 47,
        'Cd': 48,
        'In': 49,
        'Sn': 50,
        'Sb': 51,
        'Te': 52,
        'I': 53,
        'Xe': 54,
        'Cs': 55,
        'Ba': 56,
        'La': 57,
        'Ce': 58,
        'Pr': 59,
        'Nd': 60,
        'Pm': 61,
        'Sm': 62,
        'Eu': 63,
        'Gd': 64,
        'Tb': 65,
        'Dy': 66,
        'Ho': 67,
        'Er': 68,
        'Tm': 69,
        'Yb': 70,
        'Lu': 71,
        'Hf': 72,
        'Ta': 73,
        'W': 74,
        'Re': 75,
        'Os': 76,
        'Ir': 77,
        'Pt': 78,
        'Au': 79,
        'Hg': 80,
        'Tl': 81,
        'Pb': 82,
        'Bi': 83,
        'Po': 84,
        'At': 85,
        'Rn': 86,
        'Fr': 87,
        'Ra': 88,
        'Ac': 89,
        'Th': 90,
        'Pa': 91,
        'U': 92,
        'Np': 93,
        'Pu': 94,
        'Am': 95,
        'Cm': 96,
        'Bk': 97,
        'Cf': 98,
        'Es': 99,
        'Fm': 100,
        'Md': 101,
        'No': 102,
        'Lr': 103,
        'Rf': 104,
        'Db': 105,
        'Sg': 106,
        'Bh': 107,
        'Hs': 108,
        'Mt': 109,
        'Ds': 110,
        'Rg': 111,
        'Uub': 112
    }
    if mode == 1:
        try:
            return atom_dict[atom_string]
        except KeyError:
            return 0
    elif mode == 2:
        for key in atom_dict.keys():
            if atom_dict[key] == atom_string:
                return key
    return 0

if __name__ == '__main__':
    step = 500
    strength = 1/30
    step += 4 / strength
    parameter_s = np.arange(0, 1000) * 0.0021745
    a = np.arange(0, 1000)

    c = np.exp((step-a)*strength) / (np.exp((step-a)*strength) + 1)
    plt.plot(c)
    plt.show()
    print(c)
    pass