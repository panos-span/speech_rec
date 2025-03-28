#!/bin/bash
# format_lm.sh - Script to create the FST of the grammar (G.fst)
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc
UTILS_DIR=$USC_DIR/utils
STEPS_DIR=$USC_DIR/steps

# Source path.sh to set up the environment
source $USC_DIR/path.sh

echo "Creating FST of the grammar (G.fst) following timit format procedure..."

# Check if utils is properly linked, if not link it
if [ ! -e $UTILS_DIR ]; then
    echo "Utils directory not found. Creating symbolic link..."
    ln -s $KALDI_ROOT/egs/wsj/s5/utils $UTILS_DIR
fi

# Check if steps is properly linked, if not link it
if [ ! -e $STEPS_DIR ]; then
    echo "Steps directory not found. Creating symbolic link..."
    ln -s $KALDI_ROOT/egs/wsj/s5/steps $STEPS_DIR
fi

# First, ensure the basic lang directory is properly created
if [ ! -d $USC_DIR/data/lang ]; then
    echo "Error: $USC_DIR/data/lang directory does not exist."
    echo "Please run utils/prepare_lang.sh first to create the lexicon FST."
    exit 1
fi

# Check if words.txt exists in lang directory
if [ ! -f $USC_DIR/data/lang/words.txt ]; then
    echo "Error: $USC_DIR/data/lang/words.txt not found."
    echo "Please run utils/prepare_lang.sh first with the command:"
    echo "utils/prepare_lang.sh $USC_DIR/data/local/dict '<UNK>' $USC_DIR/data/local/lang $USC_DIR/data/lang"
    exit 1
fi

# Process both unigram and bigram LMs
for lm_suffix in ug bg; do
    lm_name="lm_phone_$lm_suffix"
    
    # Create lang directory for current LM type
    lang_dir=$USC_DIR/data/lang_$lm_suffix
    mkdir -p $lang_dir
    cp -r $USC_DIR/data/lang/* $lang_dir/
    
    # Check if the ARPA LM exists
    arpa_lm=$USC_DIR/data/local/nist_lm/$lm_name.arpa.gz
    if [ ! -f $arpa_lm ]; then
        echo "Error: ARPA language model $arpa_lm not found."
        echo "Please run build_lm.sh and compile_lm.sh first."
        exit 1
    fi
    
    # Create a temporary uncompressed version of the ARPA LM
    temp_arpa=${arpa_lm%.gz}.tmp
    echo "Decompressing ARPA model to $temp_arpa..."
    gunzip -c $arpa_lm > $temp_arpa
    
    # Check if filter_arpa.pl exists and is executable
    if [ -x $UTILS_DIR/filter_arpa.pl ]; then
        # Get the list of words from words.txt (excluding special symbols)
        echo "Extracting word list from $lang_dir/words.txt..."
        cut -d' ' -f1 $lang_dir/words.txt | grep -v "<" > $lang_dir/wordlist.tmp
        
        # Filter the ARPA model to only include words from words.txt
        echo "Filtering ARPA model to match vocabulary in $lang_dir/words.txt..."
        $UTILS_DIR/filter_arpa.pl $lang_dir/wordlist.tmp < $temp_arpa > $temp_arpa.filtered
        
        if [ -s $temp_arpa.filtered ]; then
            echo "Filtered ARPA model created successfully."
            mv $temp_arpa.filtered $temp_arpa
        else
            echo "Warning: Filtered ARPA model is empty. Using original ARPA model."
        fi
    else
        echo "Warning: $UTILS_DIR/filter_arpa.pl not found or not executable."
        echo "Proceeding with unfiltered ARPA model."
    fi
    
    # Format the LM as FST
    echo "Converting ARPA $lm_suffix LM to FST format..."
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$lang_dir/words.txt $temp_arpa $lang_dir/G.fst
    
    # Cleanup temporary file
    rm $temp_arpa
    [ -f $lang_dir/wordlist.tmp ] && rm $lang_dir/wordlist.tmp
    
    # Check if G.fst was created
    if [ ! -f $lang_dir/G.fst ]; then
        echo "Error: Failed to create G.fst for $lm_suffix language model."
        exit 1
    fi
    
    # Check if the FST is stochastic
    echo "Checking if $lm_suffix FST is stochastic..."
    fstisstochastic $lang_dir/G.fst
    
    # Make the FST deterministic and minimal
    echo "Optimizing $lm_suffix FST..."
    fstdeterminizestar --use-log=true $lang_dir/G.fst | \
        fstminimizeencoded | \
        fstarcsort --sort_type=ilabel > $lang_dir/G.fst.tmp
    mv $lang_dir/G.fst.tmp $lang_dir/G.fst
    
    # Final check
    echo "Final check for $lm_suffix FST..."
    fstisstochastic $lang_dir/G.fst
    
    echo "$lm_suffix G.fst created successfully at $lang_dir/G.fst"
done

echo "Grammar FSTs created successfully following timit format procedure."
echo "You can now proceed with creating HCLG graphs and running decoders."