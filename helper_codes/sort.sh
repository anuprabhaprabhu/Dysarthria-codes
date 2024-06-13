
src_fldr=$1

for txt_file in "$src_fldr"*.txt; do
    cat $txt_file | sort | sponge $txt_file
    sed -i '/^$/d' $txt_file
    
done
echo "Completed!!"