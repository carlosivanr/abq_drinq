#!/usr/bin/env python3
# Look at alpha in abqdrinq resting data

import numpy as np
import os
import fnmatch
import mne
from mne import read_proj, setup_source_space
from mne.minimum_norm import read_inverse_operator, make_inverse_operator, write_inverse_operator, compute_source_psd_epochs

study_root = '/export/research/analysis/human/eclaus/abqdrinq_20159/MEG/'
subjects_dir = '/export/research/analysis/human/eclaus/abqdrinq_20159/MEG_fs/'
task = 'rest'
visits = ['1']

# List of subjects with available MRI, fs, and MEG data. Excludes a pilot subject as well
active_subjects = ['M87101886', 'M87102015', 'M87102344', 'M87102606', 'M87102916', 'M87102958', 
                    'M87104203', 'M87104347', 'M87106152', 'M87107347', 'M87107491', 'M87107493', 
                    'M87108658', 'M87109656', 'M87109886', 'M87110232', 'M87110330', 'M87112120', 
                    'M87112171', 'M87112688', 'M87114254', 'M87115189', 'M87115406', 'M87115644', 
                    'M87118852', 'M87118897', 'M87121250', 'M87121886', 'M87122643', 'M87123852', 
                    'M87124115', 'M87124467', 'M87124607', 'M87124875', 'M87125284', 'M87125508', 
                    'M87125714', 'M87126179', 'M87126313', 'M87126589', 'M87126702', 'M87127006', 
                    'M87128896', 'M87128913', 'M87129739', 'M87131758', 'M87131982', 'M87132303', 
                    'M87132581', 'M87133223', 'M87133817', 'M87133894', 'M87135491', 'M87135971', 
                    'M87136667', 'M87137864', 'M87137882', 'M87137932', 'M87137934', 'M87138354', 
                    'M87139435', 'M87139626', 'M87140738', 'M87141001', 'M87142622', 'M87144263', 
                    'M87145099', 'M87145616', 'M87146159', 'M87146244', 'M87147495', 'M87149649', 
                    'M87149957', 'M87151334', 'M87152540', 'M87153994', 'M87154384', 'M87154540', 
                    'M87154694', 'M87155541', 'M87155721', 'M87156785', 'M87157321', 'M87157378', 
                    'M87157630', 'M87157788', 'M87158086', 'M87158737', 'M87159475', 'M87160631', 
                    'M87161349', 'M87162737', 'M87163450', 'M87165913', 'M87166078', 'M87166861', 
                    'M87167790', 'M87168844', 'M87169561', 'M87169933', 'M87171072', 'M87173293', 
                    'M87174322', 'M87174640', 'M87178337', 'M87179500', 'M87180693', 'M87181016', 
                    'M87181551', 'M87181675', 'M87182755', 'M87183825', 'M87185065', 'M87186609', 
                    'M87187003', 'M87187330', 'M87187721', 'M87188231', 'M87189209', 'M87189336', 
                    'M87189762', 'M87190774', 'M87191165', 'M87191254', 'M87192109', 'M87192361', 
                    'M87192493', 'M87192626', 'M87192942', 'M87192945', 'M87192997', 'M87194258', 
                    'M87195009', 'M87195831', 'M87196102', 'M87197470', 'M87198709', 'M87198879', 
                    'M87199012', 'M87199888', 'M87104875', 'M87107981', 'M87111902', 'M87115993', 
                    'M87133734', 'M87137809', 'M87142928', 'M87159741', 'M87162920', 'M87164283', 
                    'M87166562', 'M87169517', 'M87179368', 'M87196443', 'M87109419', 'M87133734', 
                    'M87171389', 'M87196443']

for visit in visits:
    freqdata = []  # To hold our results
    for active_subject in active_subjects:
        #active_subject = active_subjects[4]
        subject = active_subject + '_' + visit
        data_path = study_root + active_subject + '/visit' + visit
        fname_src = subjects_dir + subject + '/bem/' + subject + '-oct-6p-src.fif'
        fname_bem = subjects_dir + subject + '/bem/' + subject + '-5120-bem-sol.fif'
        fname_t1 = subjects_dir + '/' + subject + '/mri/T1.mgz'
        fname_fwd = data_path + '/' + active_subject + '_' + task + '_raw_tsss_mc_trans-' + subject + '-oct-6p-src-fwd.fif'
        fname_inv = data_path + '/' + active_subject + '_' + task + '_raw_tsss_mc_trans-' + subject + '-oct-6p-src-fwd-inv.fif'
        fname_proj_eog = data_path + '/' + active_subject + task + 'all_EOG_auto-proj.fif'
        fname_proj_ecg = data_path + '/' + active_subject + task + 'all_ECG_auto-proj.fif'
        fname_cov = data_path + active_subject + '/visit' + visit + '/' + active_subject + '_empty_raw_sss-cov.fif'
        event = 999
        mindist = 5
        snr = 1.0  # use smaller SNR for raw data
        lambda2 = 1.0 / snr ** 2
        method = "dSPM"
        # define frequencies of interest
        fmin, fmax = 7., 12.  # This gives out data on 8-12 hz
        bandwidth = 4.  # bandwidth of the windows in Hz
        n_jobs = 6


        try:
            # Get rest raw filename (fault tolerant-ish)
            pattern1 = '*rest*raw*_trans.fif'
            fnames = []
            for root, dirs, files in os.walk(study_root + active_subject + '/visit' + visit):
              for name in files:
                if fnmatch.fnmatch(name, pattern1):
                  fnames.append(os.path.join(root, name))
            fnames.sort()
            fname_raw = fnames[-1]

            # get empty room raw filename
            pattern2 = '*empty*raw*_sss.fif'
            fnames = []
            for root, dirs, files in os.walk(study_root + active_subject + '/visit' + visit):
              for name in files:
                if fnmatch.fnmatch(name, pattern2):
                  fnames.append(os.path.join(root, name))
            fnames.sort()
            fname_raw_empty = fnames[-1]

            # get head trans filename
            pattern3 = '*-trans.fif'
            fnames = []
            for root, dirs, files in os.walk(study_root + active_subject + '/visit' + visit):
              for name in files:
                if fnmatch.fnmatch(name, pattern3):
                  fnames.append(os.path.join(root, name))
            fnames.sort()
            fname_trans = fnames[-1]
        except:
            print('Problem finding raw files for %s, skipping...' % active_subject)
            continue

        # Setup for reading the raw data
        try:
            raw = mne.io.read_raw_fif(fname_raw)
            raw_empty = mne.io.read_raw_fif(fname_raw_empty)
            events = mne.find_events(raw, stim_channel='STI101')
            events = mne.make_fixed_length_events(raw, id=999, start=0, stop=None, duration=0.5)
        except:
            print('Problem reading the raw data for %s, quitting...' % active_subject)
            continue

        # Load and apply projectors
        try:
          if os.path.isfile(fname_proj_eog):
            projs_eog = read_proj(fname_proj_eog)
            raw.add_proj(projs_eog, remove_existing=True)
          if os.path.isfile(fname_proj_ecg):
            projs_ecg = read_proj(fname_proj_ecg)
            raw.add_proj(projs_ecg, remove_existing=False)
        except:
          print('Problem with projectors for %s') % active_subject

        # Drop cHPI portion

        hpi_events = mne.find_events(raw, stim_channel="STI201", output='step', consecutive=False, shortest_event=1)
        num_events,_ = hpi_events.shape
        try:
          chpi_on_event = np.where(hpi_events==256)[0][-1] # row of last occurence
          chpi_on_idx = hpi_events[chpi_on_event,0]
        except:
          print('Shit, I got nothing')
        try:
          chpi_off_event = np.where(hpi_events==3840)[0][-1] # row of last occurence
          chpi_off_idx = hpi_events[chpi_off_event,0]
        except:
          print('Shit, I got nothing')
          continue
        #hpi_off_event = np.array([hpi_off_event[0][0],hpi_off_event[1][0]])
        #hpi_off_idx = events[hpi_off_event,0]
        sfreq = raw.info['sfreq']
        tstep = 1/sfreq
        start_chpi = (chpi_on_idx - raw.first_samp) * tstep
        stop_chpi = (chpi_off_idx - raw.first_samp) * tstep
        print('###############################################################################')
        print(stop_chpi)
        print(raw.times[-1])
        print('###############################################################################')
        if (stop_chpi > raw.times[-1]):
          stop_chpi = raw.times[-1]
        raw.crop(tmin=start_chpi, tmax=stop_chpi)

        # picks MEG gradiometers
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, ecg=True, stim=False)

        # Construct Epochs
        event_id = 999
        baseline = None
        epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=1.0, picks=picks,
                  baseline=baseline, proj=True, reject=None,
                  preload=True)

        ###############################################################################
        # Do fwd model
        if os.path.isfile(fname_fwd):
            print('Forward file exists, loading %s' % fname_fwd)
            forward = mne.read_forward_solution(fname_fwd)
        else:
            if os.path.isfile(fname_src):
                print('Source space exists, loading %s' % fname_src)
                head = mne.read_source_spaces(fname_src, patch_stats=True)
            else:
                head = setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir,
                                          surface='white', add_dist=True, n_jobs=n_jobs)
                try:
                    mne.write_source_spaces(fname_src, head, overwrite=True)
                except:
                    print('Problem writing source space for %s' % active_subject)
            # get head trans filename
            try:
                pattern3 = '*-trans.fif'
                fnames = []
                for root, dirs, files in os.walk(study_root + active_subject + '/visit' + visit):
                  for name in files:
                    if fnmatch.fnmatch(name, pattern3):
                      fnames.append(os.path.join(root, name))
                fnames.sort()
                fname_trans = fnames[-1]
            except:
                print('Problem with coregistration for %s') % active_subject
                break
            forward = mne.make_forward_solution(raw.info, fname_trans, head, fname_bem,
                                meg=True, eeg=False, mindist=mindist, n_jobs=n_jobs)
            try:
                mne.write_forward_solution(fname_fwd, forward, overwrite=True)
            except:
                print('Problem writing forward solution for %s' % active_subject)


        ###############################################################################
        # Do inverse
        if os.path.isfile(fname_inv):
            print('Inverse operator exists, loading %s' % fname_inv)
            inverse_operator = read_inverse_operator(fname_inv)
            # need to do prepare_inverse_operator?
        else:
            # Compute noise covar from processed empty room data
            picks_empty = mne.pick_types(raw_empty.info, meg=True, eeg=False, eog=False, ecg=False, stim=False)
            noise_cov = mne.compute_raw_covariance(raw_empty, tmin=0, tmax=None, picks=picks_empty)
            # noise_cov.save(fname_cov)
            inverse_operator = make_inverse_operator(raw.info, forward, noise_cov, loose=1, depth=None, fixed=False)  # drop loose to avoid surface? loose=0.2, depth=0.8, fixed=False                                                 )
            # note that as of most recent MNE loose has to be 'auto' or 1 in volume source spaces.
            # if depth is not None or loose ~=!, will convert to surface orientation
            try:
                write_inverse_operator(fname_inv, inverse_operator, verbose=True)
            except:
                print('Problem writing inverse operator for %s' % active_subject)

        ###############################################################################
        # Estimate source PSD

        if os.path.isfile(subjects_dir + subject + '/label/lh.aparc.a2009s.annot'):
            try:
                all_labels = mne.read_labels_from_annot(subject, parc='aparc.a2009s', subjects_dir=subjects_dir)
                label_names = [l.name for l in all_labels]
                label_names_occip = ['G_oc-temp_lat-fusifor','G_oc-temp_med-Lingual',
                                'G_occipital_middle','G_occipital_sup','Pole_occipital','S_calcarine',
                                'S_oc-temp_lat','S_oc_middle&Lunatus',
                                'S_oc_sup&transversal','S_occipital_ant','S_parieto_occipital',
                                'G&S_occipital_inf'] # dropped 'G_oc-temp_med-Parahip','S_oc-temp_med&Lingual'
                labels_occip = []
                # could just do something like labelz = [label for label in labels_occip]
                for hemi in ['-lh','-rh']:
                    for l in label_names_occip:
                        l = l + hemi
                        labels_occip.append(all_labels[label_names.index(l)])
                cmd = 'biglabel = '
                for i in range(len(labels_occip)):
                    if i < len(labels_occip) - 1:
                        cmd = cmd + 'labels_occip[%s] + ' % i
                    else:
                        cmd = cmd + 'labels_occip[%s]' % i
                exec(cmd)  # This is a bihemi label which may only work within mne-python...
            except:
                all_labels = mne.read_labels_from_annot(subject, parc='aparc.a2009s', subjects_dir=subjects_dir)
                label_names = [l.name for l in all_labels]
                label_names_occip = ['G_oc-temp_lat-fusifor','G_oc-temp_med-Lingual',
                                'G_occipital_middle','G_occipital_sup','Pole_occipital','S_calcarine',
                                'S_oc-temp_lat','S_oc_middle_and_Lunatus',
                                'S_oc_sup_and_transversal','S_occipital_ant','S_parieto_occipital',
                                'G_and_S_occipital_inf'] # dropped 'G_oc-temp_med-Parahip','S_oc-temp_med_and_Lingual'
                labels_occip = []
                # could just do something like labelz = [label for label in labels_occip]
                for hemi in ['-lh','-rh']:
                    for l in label_names_occip:
                        l = l + hemi
                        labels_occip.append(all_labels[label_names.index(l)])
                cmd = 'biglabel = '
                for i in range(len(labels_occip)):
                    if i < len(labels_occip) - 1:
                        cmd = cmd + 'labels_occip[%s] + ' % i
                    else:
                        cmd = cmd + 'labels_occip[%s]' % i
                exec(cmd)  # This is a bihemi label which may only work within mne-python...
        else:
            print('Aparc file not found for %s' % active_subject)
            break


        stcs = compute_source_psd_epochs(epochs, inverse_operator,
                                         lambda2=lambda2,
                                         method=method, fmin=fmin, fmax=fmax,
                                         bandwidth=bandwidth, label=biglabel,
                                         return_generator=True, verbose=True)

        psd_avg = 0.
        for i, stc in enumerate(stcs):
            psd_avg += stc.data
        psd_avg /= epochs.events.shape[0]  # epochs.n_events ?
        freqs = stc.times  # the frequencies are stored here
        stc.data = psd_avg

        freq_peak = stc.times[np.where(stc.data == np.max(stc.data))[1][0]]
        freq_power = stc.data[np.where(stc.data == np.max(stc.data))][0]  # in dB already
        freqdata.append([active_subject, visit, freq_peak, freq_power])

        #del epochs, stc, raw, raw_empty, biglabel, head, forward, inverse_operator # original line, but errored out on head
        del epochs, stc, raw, raw_empty, biglabel, forward, inverse_operator

    fname_out = '/export/research/analysis/human/eclaus/abqdrinq_20159/MEG/analysis/drinq_alpha_peak_power_v' + visit + '.dat'
    np.savetxt(fname_out, freqdata, fmt='%s', delimiter='\t')
