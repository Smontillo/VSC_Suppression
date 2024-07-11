# rm -rf *0
for i in 0.001 0.0015 0.002 0.0025 0.003 0.0035 0.004 0.0045 0.005 0.0055
do

mkdir $i
cd $i
cp ../armadillo.py .
cp ../BoseFermiExpansion.py .
cp ../bath_gen_Drude_PSD.py .
cp ../default.json .
cp ../gen_input.py .
sed -i "s/eta_c =  0.0015 /eta_c = $i /g" gen_input.py
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
