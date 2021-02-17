for num in {10..1000..10}
do    
echo $num    
python test.py -c config.json -r "/scratch/aryansho.cse.iitbhu/DenFLOW/saved/models/DenFlow/0215_112330/checkpoint-epoch$num.pth" -d 1 | tee -a results.txt 
done