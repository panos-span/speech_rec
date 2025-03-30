import os
import kaldi_io
import numpy as np
from torch.utils.data import Dataset

class TorchSpeechDataset(Dataset):
    def __init__(self, recipe_dir, ali_dir, dset, feature_context=2):
        # Path handling
        if recipe_dir == '../':
            self.recipe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        else:
            self.recipe_dir = os.path.abspath(recipe_dir)
        
        # Ensure alignment directory is correct
        if not os.path.isabs(ali_dir):
            if ali_dir.startswith('../'):
                self.ali_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ali_dir))
            else:
                self.ali_dir = os.path.join(self.recipe_dir, ali_dir)
        else:
            self.ali_dir = ali_dir
        
        self.feature_context = feature_context
        self.dset = dset

        print(f"Recipe dir: {self.recipe_dir}")
        print(f"Alignment dir: {self.ali_dir}")
        print(f"Dataset: {self.dset}")
        
        # Check directory existence
        if not os.path.exists(self.ali_dir):
            raise FileNotFoundError(f"Alignment directory not found: {self.ali_dir}")
        print(f"Alignment directory exists: {self.ali_dir}")

        # Load data
        self.features, self.labels_internal = self.read_data()
        self.feats, self.labels, self.uttids, self.end_indices = self.unify_data(self.features, self.labels_internal)

    def read_data(self):
        feat_path = os.path.join(self.recipe_dir, 'data', self.dset, 'feats.scp')
        print(f"Feature file: {feat_path}")
        
        label_path = self.ali_dir
        feat_opts = "apply-cmvn --utt2spk=ark:{0} ark:{1} ark:- ark:- |". \
            format(
                os.path.join(self.recipe_dir, 'data', self.dset, 'utt2spk'),
                os.path.join(self.recipe_dir, 'data', self.dset, self.dset + '_cmvn_speaker.ark')
            )
        feat_opts += " add-deltas --delta-order=2 ark:- ark:- |"

        if self.feature_context:
            feat_opts += " splice-feats --left-context={0} --right-context={0} ark:- ark:- |". \
                format(str(self.feature_context))
        label_opts = 'ali-to-pdf'

        print(f"Feature command: ark:copy-feats scp:{feat_path} ark:- | {feat_opts}")
        print(f"Label command: gunzip -c {label_path}/ali*.gz | {label_opts} {label_path}/final.mdl ark:- ark:-|")

        try:
            feats = {k: m for k, m in kaldi_io.read_mat_ark(
                'ark:copy-feats scp:{} ark:- | {}'.format(feat_path, feat_opts))}
            print(f"Successfully read {len(feats)} feature matrices")
            
            try:
                lab = {k: v for k, v in kaldi_io.read_vec_int_ark(
                    'gunzip -c {0}/ali*.gz | {1} {0}/final.mdl ark:- ark:-|'.format(label_path, label_opts))
                    if k in feats}
                print(f"Successfully read {len(lab)} label vectors")
                
                # Filter features to match labels
                feats = {k: v for k, v in feats.items() if k in lab}
                print(f"After filtering, {len(feats)} feature matrices match labels")
                
                return feats, lab
            except Exception as e:
                print(f"Error reading labels: {e}")
                # Return empty dictionaries on error
                return {}, {}
        except Exception as e:
            print(f"Error reading features: {e}")
            return {}, {}

    def unify_data(self, feats, lab, optional_array=None):
        if not feats or not lab:
            print("Warning: Empty features or labels")
            # Return minimal structure with empty arrays
            return np.array([]).reshape(0, 1), np.array([]), [], [0]
            
        try:
            fea_conc = np.concatenate([v for k, v in sorted(feats.items())])
            lab_conc = np.concatenate([v for k, v in sorted(lab.items())])

            if optional_array:
                opt_conc = np.concatenate([v for k, v in sorted(optional_array.items())])
            
            names = [k for k, v in sorted(lab.items())]
            end_snt = 0
            end_indices = [0]  # Start with 0

            for k, v in sorted(lab.items()):
                end_snt += v.shape[0]
                end_indices.append(end_snt)

            lab = lab_conc.astype('int64')

            if optional_array:
                opt = opt_conc.astype('int64')
                return fea_conc, lab, opt, names, end_indices

            return fea_conc, lab, names, end_indices
        except Exception as e:
            print(f"Error in unify_data: {e}")
            # Return minimal structure with empty arrays
            return np.array([]).reshape(0, 1), np.array([]), [], [0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.feats[idx]), torch.LongTensor([self.labels[idx]])[0]

    def __len__(self):
        return len(self.labels)