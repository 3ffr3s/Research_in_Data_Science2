#editcap -F libpcap -T ether filename.pcapng filename.pcap
search_dir='../../../dataset/CICIDS2017'
if [ ! -d "$search_dir/split_pcaps" ];then 
    mkdir "$search_dir/split_pcaps"
fi
for entry in $search_dir/*.pcap
do
  pcap_dir=$entry
  new_dir="${search_dir}/split_pcaps/`expr substr $entry 29 3`"
  if [ ! -d $new_dir ];then 
    mkdir $new_dir
  fi
    mono SplitCap.exe -r $pcap_dir -s flow -o $new_dir
done