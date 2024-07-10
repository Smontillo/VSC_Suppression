# rm -rf *0
for i in 0.0010 0.0015 0.0020 0.0025 0.0030 0.0035 0.0040 0.0045 0.0050 0.0055 0.0060
do

mkdir $i
cd $i
cp ../armadillo.py .
cp ../BoseFermiExpansion.py .
cp ../bath_gen_Drude_PSD.py .
cp ../default.json .
cp ../gen_input.py .
sed -i "s/eta_c =  0.0025 /eta_c = $i /g" gen_input.py
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
