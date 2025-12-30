number=$(ls | grep -E '^data[0-9]+$' | wc -l)
rm -r final;
mkdir final;
for i in `seq 1 1 $number`;do cp get_energy.py data$i/.;cd data$i;python3 get_energy.py ;cd ..;done;
for i in `seq 1 1 $number`;do cat data$i/neb_data/barrier.txt >> final/barrier;cp data$i/neb_data/ts.traj final/ts$i.traj;cat data$i/pair.txt >> final/index.txt; echo "" >> final/index.txt;done;
