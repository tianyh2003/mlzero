./mlzero -t 1
mv ../log ../log2
rm -rf ../ckpt/*
mkdir ../log


./mlzero -t 2
mv ../log ../log2
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 4
mv ../log ../log4
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 8
mv ../log ../log8
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 16
mv ../log ../log16
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 32
mv ../log ../log16
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 64
mv ../log ../log16
rm -rf ../ckpt/*
mkdir ../log
