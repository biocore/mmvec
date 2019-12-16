

# Parameters that you will need to tune sorted in terms of priority
microbe_file=../examples/cf/otus_nt.biom
metabolite_file=../examples/cf/lcms_nt.biom
latent_dim=3
learning_rate=1e-5
epochs=6000
outprior=1
inprior=1
beta1=0.85
beta2=0.90
batch_size=1000

RESULTS_DIR=summary/
mmvec paired-omics \
      --microbe-file $microbe_file \
      --metabolite-file $metabolite_file \
      --min-feature-count 10 \
      --num-testing-examples 10 \
      --learning-rate $learning_rate \
      --beta1 $beta1 \
      --beta2 $beta2 \
      --summary-dir $RESULTS_DIR \
      --epochs $epochs \
      --summary-interval 10 \
      --checkpoint-interval 3600 \
      --batch-size $batch_size \
      --latent-dim $latent_dim \
      --input-prior $inprior \
      --output-prior $outprior
      # --top-k 10 \
      # --threads 1 \