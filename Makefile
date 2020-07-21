
visualising_data/OccTraj_in_mpnet_format: visualising_data/OccTraj_in_mpnet_format.tar.gz
	cd visualising_data && tar xzf OccTraj_in_mpnet_format.tar.gz
	touch $@

clean:
	rm -rf visualising_data/OccTraj_in_mpnet_format
