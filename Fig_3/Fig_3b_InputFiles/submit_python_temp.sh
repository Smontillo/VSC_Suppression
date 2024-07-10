for j in 0.0010 0.0015 0.0020 0.0025 0.0030 0.0035 0.0040 0.0045 0.0050 0.0055 0.0060
do

cd $j

for i in 280 290 300 310 320 330
do

mkdir $i
cd $i
sed -i "s/temp = 300 /temp = $i/g" gen_input.py
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

cd ../
done