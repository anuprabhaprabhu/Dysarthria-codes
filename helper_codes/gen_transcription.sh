src_fldr=$1
echo "Working"
echo $src_fldr
for txt_file in "$src_fldr"*.txt; do
    b=$(cat "$txt_file" )
 
    echo  -e  $txt_file '\t' $b  >> $src_fldr/transcript.txt
    
done
echo "Completed!!"