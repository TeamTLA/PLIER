import numpy as np
import pandas as pd
from pathlib import Path
import pysam, re
import h5py

def overlap(que_item, ref_lst, include_ref_left=False, include_ref_right=False, offset=0):
    if isinstance(que_item, list):
        que_item = np.array(que_item)
    if isinstance(ref_lst, list):
        ref_lst = np.array(ref_lst)
    que_dim = que_item.shape[0]
    [n_ref, ref_dim] = np.shape(ref_lst)
    if (que_dim != ref_dim) or (que_item.ndim != 1):
        raise ValueError('Query or reference are inconsistent')

    crd_ind = 0
    has_ovl = np.ones(n_ref, dtype=bool)
    if que_dim == 4:  # Orientation
        has_ovl = que_item[3] == ref_lst[:, 3]
    if que_dim >= 3:  # Chromosome
        has_ovl = np.logical_and(has_ovl, que_item[0] == ref_lst[:, 0])
        crd_ind = 1

    if include_ref_left:
        lft_ovl = ref_lst[:, crd_ind] <= (que_item[crd_ind + 1] + offset)
    else:
        lft_ovl = ref_lst[:, crd_ind] <  (que_item[crd_ind + 1] + offset)
    if include_ref_right:
        rgh_ovl = ref_lst[:, crd_ind + 1] >= (que_item[crd_ind] - offset)
    else:
        rgh_ovl = ref_lst[:, crd_ind + 1] >  (que_item[crd_ind] - offset)
    has_ovl = np.logical_and(has_ovl, np.logical_and(lft_ovl, rgh_ovl))
    return has_ovl


def get_chr_info(reference_fasta, property='chr_name'):

    fasta_path = Path(reference_fasta)
    fasta_file = pysam.FastaFile(fasta_path) # it will create .fai if not exist as opposed to what the documentation says, raises valueError...

    if fasta_file.nreferences is None: # if empty .fai file...
        fasta_file.close()
        fasta_file.unlink()
        fasta_file = pysam.FastaFile(fasta_path) # to create new index

    if property == "chr_name":
        return_list = fasta_file.references
    elif property == "chr_size":
        return_list = fasta_file.lengths

    fasta_file.close()

    return return_list


def get_re_info(re_name='DpnII', property='seq', reference_fasta=None):
    re_details = dict({
        'DpnII': dict({'seq': 'GATC'}),
        'MboI': dict({'seq': 'GATC'}),
        'Csp6I': dict({'seq': 'GTAC'}),
        'NlaIII': dict({'seq': 'CATG'}),
        'XbaI': dict({'seq': 'TCTAGA'}),
        'BamHI': dict({'seq': 'GGATCC'}),
        'SacI': dict({'seq': 'GAGCTC'}),
        'PstI': dict({'seq': 'CTGCAG'}),
        'HindIII': dict({'seq': 'AAGCTT'})
    })

    if property == 'pos':
        re_fname = Path(reference_fasta).with_suffix(".renzs.npz")
        if re_fname.is_file():
            print('[w] Restriction enzyme file exists: ' + str(re_fname))

        if not re_fname.is_file():
            extract_re_positions(re_name_lst=[re_name], output_fname_renzs=re_fname, reference_fasta=reference_fasta)
        re_data = np.load(re_fname, allow_pickle=True)
        assert np.array_equal(re_data['chr_lst'], get_chr_info(reference_fasta, property='chr_name'))
        return re_data['pos_lst']
    else:
        return re_details[re_name][property]

def flatten(nested_lst):
    out_lst = []
    for item in nested_lst:
        if isinstance(item, list):
            out_lst.extend(flatten(item))
        else:
            out_lst.append(item)
    return out_lst


def extract_re_positions(re_name_lst, reference_fasta, output_fname_renzs):
    # Initialization
    chr_lst = get_chr_info(reference_fasta, property='chr_name')
    chr2idx = dict(zip(chr_lst, np.arange(len(chr_lst))))

    output_fname_renzs = Path(reference_fasta).with_suffix(".renzs.npz")
    if output_fname_renzs.is_file():
        print('[w] Restriction enzyme file exists: ' + output_fname)

    print('Searching in the reference genome defined in: ' + reference_fasta)

    # get re sequences
    seq_lst = []
    for re_name in re_name_lst:
        seq_lst.append(get_re_info(re_name=re_name, property='seq'))
    re_regex = '|'.join(seq_lst)

    # Loop over chromosomes
    re_pos_lst = [None] * len(chr_lst)
    re_type_lst = [None] * len(chr_lst)
    chr_observed = [None] * len(chr_lst)
    with pysam.FastxFile(reference_fasta) as ref_fid:
        print('Scanning chromosomes for restriction recognition sequences: {:s}'.format(', '.join(seq_lst)))
        for chr_ind, chr in enumerate(ref_fid):
            if not chr.name in chr_lst:
                print('\t{:s} is ignored,'.format(chr.name))
                continue
            print('\t{:s},'.format(chr.name))

            re_pos = []
            re_type = []
            for frg in re.finditer(re_regex, chr.sequence, re.IGNORECASE):
                re_pos.append(frg.start() + 1)
                re_type.append(seq_lst.index(str.upper(frg.group())) + 1)
            re_pos_lst[chr2idx[chr.name]] = np.array(re_pos, dtype=np.uint32)
            re_type_lst[chr2idx[chr.name]] = np.array(re_type, dtype=np.uint32)
            chr_observed[chr2idx[chr.name]] = chr.name
        assert np.array_equal(chr_lst, chr_observed), '[e] Inconsistent reference genome!'
        print()

    # save the result
    np.savez(output_fname_renzs, pos_lst=re_pos_lst, type_lst=re_type_lst, chr_lst=chr_observed, genome=reference_fasta, scan_regex=re_regex)


def get_vp_info(run_id, vp_info_path='./vp_info.tsv'):
    vpi_lst = pd.read_csv(vp_info_path, delimiter='\t', comment='#')  # , skip_blank_lines=False
    if isinstance(run_id, str):
        run_id = np.where(vpi_lst['run_id'] == run_id)[0]
        assert len(run_id) == 1, 'Error: Non-unique experiment is identified!'
        run_id = int(run_id[0])
    vp_info = vpi_lst.iloc[run_id].to_dict()
    vp_info['row_index'] = run_id
    if not vp_info:
        raise Exception('VP information could not be found.')
    return vp_info


def load_dataset(vp_info_lst, target_field='frg_np', data_path='./datasets', verbose=True, vp_width=None,
                 load_cols=('chr_idx', 'pos', '#read')):

    if not isinstance(vp_info_lst, list):
        vp_info_lst = [vp_info_lst]

    # loop over runs
    if verbose:
        print('Loading data from:')
    for vp_idx, vp_info in enumerate(vp_info_lst):
        h5_fpath = Path(data_path).joinpath(vp_info['expr_id']).with_suffix('.hdf5')
        if verbose:
            print('\t#{:d}: [{:d}: {:s}]: {:s}'.format(vp_idx + 1, vp_info['row_index'], vp_info['expr_id'], h5_fpath))
            if (vp_info_lst[0]['vp_chr_idx'] != vp_info['vp_chr_idx']) or (np.abs(vp_info_lst[0]['vp_pos'] - vp_info['vp_pos']) > 1e6):
                print('[w] Viewpoint is far away compared to others runs being loaded.')

        # load from hdf5
        with h5py.File(h5_fpath, 'r') as h5_fid:
            # python2: header_lst = list(h5_fid[target_field + '_header_lst'][()])
            header_lst = [hdr for hdr in h5_fid[target_field + '_header_lst']]
            frg_prt = pd.DataFrame(h5_fid[target_field][()], columns=header_lst)

        # only preserve requested columns
        if load_cols:
            frg_prt = frg_prt[list(load_cols)]

        # remove vp frag-ends?
        if vp_width is not None:
            print('\tRemoving frag-ends closer than {:0,.0f}bp to viewpoint.'.format(vp_width / 2.0))
            is_nei = (frg_prt['chr_idx'] == vp_info['vp_chr_idx']) & \
                     (np.abs(frg_prt['pos'] - vp_info['vp_pos']) < vp_width / 2.0)
            frg_prt = frg_prt.loc[~is_nei].reset_index(drop=True)
            print('\t{:,d} frag-ends are removed.'.format(np.sum(is_nei)))
            del is_nei

        # report stats
        if verbose:
            is_cis = frg_prt['chr_idx'] == vp_info['vp_chr_idx']
            is_far = is_cis & ~ overlap([vp_info['vp_chr_idx'], vp_info['vp_be'], vp_info['vp_en']], frg_prt[['chr_idx', 'pos', 'pos']].values, offset=50e3)
            print('\tData stats are:')
            print('\t\t   #fragment (far): {:0,.0f}'.format(np.sum(frg_prt.loc[ is_far, '#read'])))
            print('\t\t   #fragment (cis): {:0,.0f}'.format(np.sum(frg_prt.loc[ is_cis, '#read'])))
            print('\t\t #fragment (trans): {:0,.0f}'.format(np.sum(frg_prt.loc[~is_cis, '#read'])))

        if vp_idx == 0:
            frg_cmb = frg_prt.copy()
            del frg_prt
        else:
            agr_hlst = ['chr_idx', 'pos']
            if np.array_equal(frg_cmb[agr_hlst], frg_prt[agr_hlst]):
                print('\t[i] Identical restriction sites detected. Direct addition of coverages...')
                frg_cmb['#read'] += frg_prt['#read']
                del frg_prt
            else:
                print('\t[i] Diverse restriction sites are detected. Aggregation ...')
                frg_cmb = frg_cmb.append(frg_prt.copy(), ignore_index=True, sort=False)
                del frg_prt

                # aggregation
                rf_inv, rf_idx = np.unique(frg_cmb[agr_hlst], axis=0, return_inverse=True, return_index=True)[1:]
                rf_nrd = np.bincount(rf_idx, weights=frg_cmb['#read'], minlength=len(rf_inv))
                frg_cmb = frg_cmb.loc[rf_inv, :].reset_index(drop=True)
                frg_cmb['#read'] = rf_nrd
                del rf_inv, rf_idx, rf_nrd
        if verbose:
            print('\tCurrent memory usage: {:0,.2f}GB'.format(frg_cmb.memory_usage().sum() / 1e9))

    if verbose and (len(vp_info_lst) != 1):
        print('Final memory usage: {:0,.2f}GB'.format(frg_cmb.memory_usage().sum() / 1e9))
    return frg_cmb.copy()

def str2crd(str_crd, genome='hg19'):
    chr_lst = get_chr_info(genome=genome, property='chr_name')
    n_chr = len(chr_lst)
    chr2nid = dict(zip(chr_lst, range(1, n_chr + 1)))

    items = str_crd.split(':')
    if len(items) == 1:
        return [chr2nid[items[0]]]
    else:
        return [chr2nid[items[0]]] + [float(x) for x in items[1].split('-')]


def perform_sigtest(observed, smoothing_kernel, background=None, n_epoch=1000, nz_only=False, replacement=False):
    if (observed.ndim != 1) or (background.ndim != 1):
        raise Exception('Inconsistent data are provided.')
    n_obs = len(observed)
    if background is None:
        background = observed.copy()
    if len(background) < len(observed):
        replacement = True

    # Calculate observed
    observed_smoothed = np.convolve(observed, smoothing_kernel, mode='same')

    # Calculate background
    bkgnd_mat = np.zeros([n_epoch, n_obs])
    if nz_only:
        inz_obs = observed > 0
        n_obs = np.sum(inz_obs)
        drawn_array = observed.copy()
        background = background[background > 0].copy()
        for ei in range(n_epoch):
            drawn_array[inz_obs] = np.random.choice(background, n_obs, replace=replacement)
            bkgnd_mat[ei, :] = np.convolve(drawn_array, smoothing_kernel, mode='same')
    else:
        for ei in range(n_epoch):
            drawn_array = np.random.choice(background, n_obs, replace=replacement)
            bkgnd_mat[ei, :] = np.convolve(drawn_array, smoothing_kernel, mode='same')

    # observed_smoothed[observed_smoothed < 0] = 0
    # bkgnd_mat[bkgnd_mat < 0] = 0
    return observed_smoothed, bkgnd_mat


class OnlineStats(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    citation: Welford, B. P. (1962). doi:10.2307/1266577
    from: https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
    example:
        from _utilities import OnlineStats
        ostat = OnlineStats(ddof=0)
        for item in item_lst:
            bkg_ostat.include(item)
        assert np.allclose(ostat.mean, np.mean(item_lst))
        assert np.allclose(ostat.std, np.std(item_lst))
    """

    def __init__(self, items=None, ddof=0):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if items is not None:
            for item in items:
                self.include(item)

    def include(self, item):
        delta = item - self.mean
        self.n += 1
        self.mean += delta / self.n
        self.M2 += delta * (item - self.mean)

    def combine(self, arr):
        if isinstance(arr, OnlineStats):
            arr_n, arr_mean, arr_m2 = arr.n, arr.mean, arr.M2
        else:
            arr_n, arr_mean, arr_m2 = len(arr), np.mean(arr), np.var(arr) * (len(arr) - self.ddof)
        ostat_cmb = OnlineStats(ddof=self.ddof)
        ostat_cmb.n = self.n + arr_n
        delta_mean = arr_mean - self.mean
        ostat_cmb.mean = (self.n * self.mean + arr_n * arr_mean) / ostat_cmb.n
        ostat_cmb.M2 = self.M2 + arr_m2 + (delta_mean * delta_mean) * self.n * arr_n / ostat_cmb.n
        return ostat_cmb

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)


def get_read(sam):
    return_l = []
    for read in sam:
        read_name = read.query_name
        if read.is_unmapped:
            continue

        if return_l == list():
            return_l.append(read)
        else:
            if read_name == return_l[-1].query_name:
                return_l.append(read)
            else:
                yield return_l
                return_l = [read]

    yield return_l



