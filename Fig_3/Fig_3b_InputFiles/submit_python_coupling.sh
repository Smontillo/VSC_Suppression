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

done