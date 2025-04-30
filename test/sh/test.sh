./mlzero -t 1
mv ../log ../logall/log1
rm -rf ../ckpt/*
mkdir ../log


./mlzero -t 2
mv ../log ../logall/log2
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 4
mv ../log ../logall/log4
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 8
mv ../log ../logall/log8
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 16
mv ../log ../logall/log16
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 32
mv ../log ../logall/log32
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 64
mv ../log ../logall/log64
rm -rf ../ckpt/*
mkdir ../log

./mlzero -t 128
mv ../log ../logall/log128
rm -rf ../ckpt/*
mkdir ../log
