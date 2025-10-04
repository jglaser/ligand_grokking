# Collect unique stems (grouping by everything except trailing _SEED.txt)
mapfile -t stems < <(ls -1 $1/*_out_*_*.txt | sed -E "s/_[0-9]+\.txt$//" | sort -u)

for stem in "${stems[@]}"; do
  files=("${stem}"_*.txt)
  echo "==== ${stem} ===="

  # Compute mean/std/k; allow for no matches without failing the script
  stats=$( (grep -h "Test Set ROC AUC" "${files[@]}" || true) | \
      awk '$NF ~ /^[0-9.]+$/ { c++; s+=$NF; ss+=$NF*$NF } END { if (c > 1) { m = s / c; sd = sqrt((ss - s^2/c) / (c-1)); printf "--mean %.4f --std %.4f --k %d\n", m, sd, c } }' )

  if [ -n "$stats" ]; then
    echo "stats $stats"
    if python p_value.py $stats 2>&1; then
      :
    else
      echo "python p_value.py failed for ${stem}"
    fi
  else
    echo "Error: Not enough data points or no matching lines."
  fi
  echo
done
