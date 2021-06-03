#!/usr/bin/env python3
import os
import sys
import argparse

import numpy as np
import pandas as pd
import pysam
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from utilities import get_chr_info, overlap, get_read, str2crd, get_re_info


# initialization
np.set_printoptions(linewidth=500, threshold=5000, formatter={'int_kind': '{:3,d}'.format})
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--vpi_fpath', default='./vp_info.tsv', type=str, required=True)
arg_parser.add_argument('--bam_fpath', default=None, required=True)
arg_parser.add_argument('--vp_crd', default=None, type=str)
arg_parser.add_argument('--drv_crd', default=None, type=str)
arg_parser.add_argument('--expr_id', default=None)
arg_parser.add_argument('--genome', default=None)
arg_parser.add_argument('--bin_w', default=10e3, type=float)
arg_parser.add_argument('--max_prc', default=95, type=float)
arg_parser.add_argument('--output_dir', default=None, type=str, required=True)
inp_args = arg_parser.parse_args()

# load vp infos
vp_info = None
if inp_args.expr_id:
    vp_infos = pd.read_csv(inp_args.vpi_fpath, sep='\t')
    sel_expr = vp_infos['expr_id'] == inp_args.expr_id
    assert np.sum(sel_expr == 1)
    vp_info = vp_infos.loc[sel_expr].iloc[0].to_dict()

# get chr info
if inp_args.genome is None:
    assert vp_info is not None
    chr_lst = get_chr_info(vp_info['genome'])
    inp_args.genome = vp_info['genome']
else:
    chr_lst = get_chr_info(inp_args.genome)
n_chr = len(chr_lst)
chr2nid = dict(zip(chr_lst, range(1, n_chr + 1)))


# define regions
if inp_args.vp_crd:
    vp_crd = [int(float(c)) for c in inp_args.vp_crd.split(',')]
else:
    vp_crd = [vp_info['vp_chr_idx'], vp_info['vp_be'], vp_info['vp_end']]
drv_crd = [int(float(c)) for c in inp_args.drv_crd.split(',')]
print('    VP: {:s}:{:0.3f}-{:0.3f}\n'.format(chr_lst[vp_crd[0] - 1], vp_crd[1] / 1e6, vp_crd[2] / 1e6) +
      'Target: {:s}:{:0.3f}-{:0.3f}'.format(chr_lst[drv_crd[0] - 1], drv_crd[1] / 1e6, drv_crd[2] / 1e6))
valid_chrs = [chr_lst[vp_crd[0] - 1],
              chr_lst[drv_crd[0] - 1]]
              
              
vp_size = vp_crd[2]-vp_crd[1]
vp_pos = vp_info['vp_pos']
peak_size = vp_size
drv_pos = vp_info['call_pos'] if vp_info.get('call_pos') else vp_info['vp_pos']

vp_crd = [int(item) for item in (vp_crd[0], (vp_pos-(peak_size/2))- peak_size * 0.2, (vp_pos+(peak_size/2)) + peak_size * 0.2 )]
drv_crd = [int(item) for item in (drv_crd[0], (drv_pos-(peak_size/2))- peak_size * 0.2, (drv_pos+(peak_size/2)) + peak_size * 0.2 )]

inp_args.bin_w = vp_size/100

# defining bins
x_edges = np.arange(drv_crd[1], drv_crd[2] + inp_args.bin_w, inp_args.bin_w, dtype=int)
y_edges = np.arange(vp_crd[1], vp_crd[2] + inp_args.bin_w, inp_args.bin_w, dtype=int)
n_xbin = len(x_edges) - 1
n_ybin = len(y_edges) - 1
x_lim = [x_edges[0], x_edges[-1]]
y_lim = [y_edges[0], y_edges[-1]]

# get restriction enzyme positions
print('Loading restriction sites: {:s}'.format(vp_info['res_enzyme']))
re_pos_lst = get_re_info(property='pos', re_name=vp_info['res_enzyme'], reference_fasta=inp_args.genome)
xbin_rfs = re_pos_lst[drv_crd[0] - 1].copy()
xbin_rfs = xbin_rfs[(xbin_rfs >= x_lim[0]) &
                    (xbin_rfs < x_lim[1])]
del re_pos_lst

# loading ligation-products
# samtools sort --threads=4 F209_MYC_TLC2004.bam > F209_MYC_TLC2004.sorted.bam
# samtools index F209_MYC_TLC2004.sorted.bam
n_used = 0
bin_frq = np.zeros([n_ybin, n_xbin], dtype=np.int)
is_rf_cpt = np.zeros([n_ybin, len(xbin_rfs)], dtype=np.bool)
print('Loading ligation-products from: {:s}'.format(inp_args.bam_fpath))
with pysam.AlignmentFile(os.path.expanduser(inp_args.bam_fpath), 'r') as inp_fid:
    # get a batch of reads overlapping with ROI
    # assert inp_fid.header['HD']['SO'] == 'queryname'  # make sure alignment is name sorted: 'coordinate' or 'queryname'
    read_idx = 0
    BAM_EOF = False
    while not BAM_EOF:
        BAM_EOF = True

        # load a batch of reads
        reads = []
        for pysam_read in get_read(inp_fid):
            read_idx += 1
            for pysam_frag in pysam_read:
                if pysam_frag.reference_name not in valid_chrs:
                    continue
                frag = [
                    read_idx,
                    chr2nid[pysam_frag.reference_name],
                    pysam_frag.reference_start,
                    pysam_frag.reference_end
                ]
                reads.append(frag)
            if len(reads) >= 1e6:
                print('{:12,d} reads are scanned, {:8,d} are used ...'.format(read_idx, n_used))
                BAM_EOF = False
                break
        reads = np.array(reads)

        # check overlap
        is_vp = overlap(vp_crd, reads[:, 1:4])
        is_val = np.isin(reads[:, 0], reads[is_vp, 0])
        if not np.all(is_val):
            n_rem = len(np.unique(reads[~is_val, 0]))
            reads = reads[is_val, :]
        del is_vp, is_val
        is_roi = overlap(drv_crd, reads[:, 1:4])
        reads = reads[np.isin(reads[:, 0], reads[is_roi, 0]), :]
        if len(reads) == 0:
            continue
        n_used += len(np.unique(reads[:, 0]))

        # loop over bins
        for bi in range(n_ybin):
            ybin_crd = [vp_info['vp_chr_idx'], y_edges[bi], y_edges[bi + 1] - 1]
            is_ybin = overlap(ybin_crd, reads[:, 1:4])
            is_sel = np.isin(reads[:, 0], reads[is_ybin, 0]) & (reads[:, 1] == drv_crd[0])
            reads_sel = reads[is_sel, :]
            if len(reads_sel) == 0:
                continue

            # store capture status
            assert np.all(reads_sel[:, 1] == drv_crd[0])
            rf_idx = np.searchsorted(xbin_rfs, reads_sel[:, 2], side='right') - 1
            is_rf_cpt[bi, rf_idx] = True

            # calculate coverage
            for bj in range(n_xbin):
                xbin_crd = [drv_crd[0], x_edges[bj], x_edges[bj + 1] - 1]
                has_ovl = overlap(xbin_crd, reads_sel[:, 1:4])
                n_ovl = len(np.unique(reads_sel[has_ovl, 0]))
                bin_frq[bi, bj] += n_ovl

# calculate #captures
bin_cpt = np.zeros([n_ybin, n_xbin], dtype=np.int)
for bi in range(n_ybin):
    xbin_idx = np.searchsorted(x_edges, xbin_rfs[is_rf_cpt[bi]], side='right') - 1
    bin_cpt[bi, :] = np.bincount(xbin_idx, minlength=n_xbin)

# plotting
plt.close('all')
fig_h, axes_h = plt.subplots(nrows=1, ncols=2, figsize=[12, 4])
plt.suptitle('{:s}'.format(os.path.basename(inp_args.bam_fpath)), fontweight='bold')
data_collection = [
    ('Coverage', bin_frq),
    ('Capture', bin_cpt)
]
for img_idx, (img_name, img_np) in enumerate(data_collection):
    ax_h = axes_h[img_idx]

    # plot the coverage (often a butterfly)
    c_lim = [1, np.maximum(5, np.percentile(img_np, q=inp_args.max_prc))]
    clr_map = LinearSegmentedColormap.from_list('test', ['#ccffcc', '#00eb00', '#08b508'])  # , '#00d100'
    # clr_map = cm.get_cmap('hot_r', 12)
    clr_map.set_under('#ffffff')
    clr_map.set_bad('#fcbd5f')
    img_h = ax_h.imshow(img_np, cmap=clr_map, vmin=c_lim[0], vmax=c_lim[1], origin='lower', interpolation='nearest',
                        extent=x_lim + y_lim)
    ax_h.set_aspect('auto')
    cb_h = plt.colorbar(mappable=img_h, ax=ax_h, extend='both')
    cb_h.set_label(img_name)

    # add max point
    img_maxrow, img_maxcol = np.unravel_index(img_np.argmax(), img_np.shape)
    ax_h.plot(x_edges[img_maxcol], y_edges[img_maxrow], 'x', color='#0acaff', markersize=10)

    # add probed boundary
    ax_h.hlines(y=[vp_info['vp_be'], vp_info['vp_en']], xmin=x_lim[0], xmax=x_lim[-1],
                linewidth=0.5, color='#d18100', alpha=1.0, linestyle=':', label='Probed area')

    # adjustments
    ax_h.set_xlim(x_lim[0], x_lim[1])
    ax_h.set_ylim(y_lim[0], y_lim[1])
    xticks = np.linspace(x_lim[0], x_lim[1], 9)
    xticks_lbl = ['{:0,.1f}'.format(x / 1e6) for x in xticks]
    ax_h.set_xticks(xticks)
    ax_h.set_xticklabels(xticks_lbl)
    ax_h.set_xlabel('Partner={:s}:{:0.03f}m\n'.format(chr_lst[drv_crd[0] - 1], np.mean(drv_crd[1:]) / 1e6) +
                    'max(val)={:0.0f}, '.format(np.nanmax(img_np)) +
                    'color={:0.0f}-{:0.0f}'.format(*c_lim))
    yticks = np.linspace(y_lim[0], y_lim[1], 9)
    yticks_lbl = ['{:0,.2f}'.format(y / 1e6) for y in yticks]
    ax_h.set_yticks(yticks)
    ax_h.set_yticklabels(yticks_lbl)
    # ax_h.yaxis.set_label_coords(-0.15, 0.5)
    ax_h.set_ylabel('Target={:s}:{:0.03f}m\n'.format(chr_lst[vp_crd[0] - 1], np.mean(vp_crd[1:]) / 1e6))
    ax_h.set_title('#read={:0.2f}k, #capture={:0,.0f}, vp-width={:0.1f}'.format(n_used / 1e3, np.sum(bin_cpt), (vp_info['vp_en'] - vp_info['vp_be']) / 1e3))
    ax_h.invert_yaxis()
    # plt.show()

# final adjustments
# plt.tight_layout(h_pad=1.0)
plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.1)

# store the plot
png_fpath = os.path.join(inp_args.output_dir,'butterfly-plot_{:s}_'.format(os.path.basename(inp_args.bam_fpath)) + \
            'vp{:0.0f},{:0.1f}-{:0.1f}m_'.format(vp_crd[0], vp_crd[1] / 1e6, vp_crd[2] / 1e6) + \
            'roi{:0.0f},{:0.1f}-{:0.1f}m_'.format(drv_crd[0], drv_crd[1] / 1e6, drv_crd[2] / 1e6) + \
            'bw{:0.1f}k.png'.format(inp_args.bin_w / 1e3))
                  
os.makedirs(os.path.dirname(png_fpath), exist_ok=True)
plt.savefig(png_fpath, bbox_inches='tight')
print('Plot is generated in: {:s}'.format(png_fpath))


