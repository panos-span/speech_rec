#!/bin/bash

# Path variables
KALDI_ROOT=/path/to/kaldi
EXP=$KALDI_ROOT/egs/usc

# Create required directories
mkdir -p $EXP/data/local/lang
mkdir -p $EXP/data/lang

# Extract unique phones from the train set
cat $EXP/data/train/text | cut -d ' ' -f2- | tr ' ' '\n' | sort -u > $EXP/data/local/lang/phones.txt

# Create silence and non-silence phone lists
grep -v '^sil$' $EXP/data/local/lang/phones.txt > $EXP/data/local/lang/nonsilence_phones.txt
echo "sil" > $EXP/data/local/lang/silence_phones.txt
echo "sil" > $EXP/data/local/lang/optional_silence.txt

# Create the lexicon file
cat > $EXP/data/local/lang/lexicon.txt << EOF
<SIL> sil
EOF

# Add all phones to lexicon
cat $EXP/data/local/lang/nonsilence_phones.txt | while read phone; do
  echo "$phone $phone" >> $EXP/data/local/lang/lexicon.txt
done

# Prepare language data
utils/prepare_lang.sh \
  --position-dependent-phones false \
  $EXP/data/local/lang "<UNK>" $EXP/data/local/lang_tmp $EXP/data/lang

# Create the training text for language model
cat $EXP/data/train/text | cut -d ' ' -f2- > $EXP/data/local/lm_train.txt

# Train language model using IRSTLM
build-lm.sh -i $EXP/data/local/lm_train.txt -n 3 -o $EXP/data/local/lm.arpa

# Convert ARPA file to FST format
utils/format_lm.sh \
  $EXP/data/lang $EXP/data/local/lm.arpa "<UNK>" $EXP/data/lang_test

echo "Language model creation completed"