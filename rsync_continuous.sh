
while true
do
rsync -r -e "ssh -i $HOME/.ssh/id_rsa" tgan4199@muspelheim.cs.usyd.edu.au:docker-tf/AdaCNN/test-start-big-5/ada_cnn_tensorboard_data/ ./tensorboard_sync
sleep 30
done
