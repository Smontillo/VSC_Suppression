
mkdir summary

for i in 700 800 900 1000 1100 1175 1200 1300 1400 1500 1600
do

cd $i

#cp prop-rho.dat ../summary/$i.dat
head -n -1 prop-rho.dat > temp.txt ; mv temp.txt ../summary/$i.dat
cd ../

done
