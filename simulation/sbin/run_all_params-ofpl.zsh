repetitions=$1
start=$2
output=$3
effect=$4
allowrtn=$5
idio=$6
echo "usage: run_all_params-ofpl.zsh <repetitions> <start> <output> <effect> <allowrtn> <idio>"
echo " |- repetitions: number of repetitions"
echo " |- start: first repetition number and random seed"
echo " |- output: output directory"
echo " |- effect: treatment effect"
echo " |- allowrtn: whether to allow returning individuals to be treated"
echo " |- idio: whether to allow idiosyncrasies"
echo "Generating commands for $repetitions repetitions # $start => $(($start+$repetitions-1))
\t with effect=$effect
\t and allowrtn=$allowrtn
\t and idio=$idio individuals"

policies=(null high-risk low-risk age-first)

scale_factors=(0.02 0.7)
term_lengths=(1000)
# default 30
maxreturns=30
# default 5
beta=10
# default 80
cap=100

# Generate commands to cmd.sh
if [[ -f $output/cmd.sh ]]; then
  rm $output/cmd.sh
fi
if [[ -d $output/logs ]]; then
  rm -r $output/logs
fi
mkdir -p $output/logs
for ((k=$start; k<$start+$repetitions; k++)); do
  for scale_factor in $scale_factors; do
    for term_length in $term_lengths; do
      nm=tl-${term_length}-sf-${scale_factor}
      outd=$output/$nm
      if [[ -d $outd ]]; then
        rm -r $outd/logs
      fi
      mkdir -p $outd/logs
      for policy in $policies; do
        echo "python -u run_policy.py \
    --prison_rate_scaler $scale_factor \
    --length_scaler 1.0 \
    --beta_arrival $beta \
    --max_returns $maxreturns \
    --max_offenses 35 \
    --T_max 40000 --p_length 100 \
    --rel_off_probation ${term_length} \
    --treatment_capacity $cap \
    --treatment_effect $effect \
    --bool_return_can_be_treated $allowrtn \
    --bool_keep_idiosyncratic_effect $idio \
    $policy $k $outd &> $outd/logs/${policy}_${k}_${nm}.log " >> $output/cmd.sh
      done
    done
  done
done

echo "Commands generated in cmd.sh."
echo "see:  $output/cmd.sh"
echo "parallel run with: \n
cat $output/cmd.sh | xargs -I {} -P 30 bash -c \"{}\" &"
