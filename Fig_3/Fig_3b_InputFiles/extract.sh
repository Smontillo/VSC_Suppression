
mkdir summary
 
for i in 280 290 300 310 320 330
do

cd $i

#cp prop-rho.dat ../summary/$i.dat
head -n -1 prop-rho.dat > temp.txt ; mv temp.txt ../summary/$i.dat
cd ../

done
