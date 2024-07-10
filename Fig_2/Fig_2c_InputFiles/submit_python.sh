# rm -rf *0
for i in 700 800 900 1000 1050 1100 1130 1160 1189 1220 1250 1300 1350 1400 1500 1600 1700
do

mkdir $i
cd $i
cp ../armadillo.py .
cp ../BoseFermiExpansion.py .
cp ../bath_gen_Drude_PSD.py .
cp ../default.json .
cp ../gen_input.py .
sed -i "s/wc = 1000 /wc = $i/g" gen_input.py
echo $i

cat << Eof > sbatch_python.sh
#!/bin/bash
#SBATCH -p debug
#SBATCH -o output_py.log
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -N 1

python3 gen_input.py

Eof

chmod +x sbatch_python.sh
sbatch sbatch_python.sh
cd ../

done
