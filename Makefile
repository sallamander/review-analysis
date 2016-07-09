work/.folder_structure_sentinel: 
	mkdir -p data/reviews/amazon
	mkdir data/word_embeddings
	mkdir work

	touch work/.folder_structure_sentinel

folders: work/.folder_structure_sentinel

##################################
# Wikipedia + Gigaword embedding # 
##################################

data/word_embeddings/glove.6B.50d.txt: 
	curl http://nlp.stanford.edu/data/glove.6B.zip -O
		
	unzip glove.6B.zip

	# The embeddings require the number of words and dimenions in the first line.
	echo '400000 50' | cat - glove.6B.50d.txt > temp && mv temp glove.6B.50d.txt
	echo '400000 100' | cat - glove.6B.100d.txt > temp && mv temp glove.6B.100d.txt
	echo '400000 200' | cat - glove.6B.200d.txt > temp && mv temp glove.6B.200d.txt
	echo '400000 300' | cat - glove.6B.300d.txt > temp && mv temp glove.6B.300d.txt

	mv *.txt data/word_embeddings
	rm glove.6B.zip

###############################
# Amazon food reviews Dataset # 
###############################

data/reviews/amazon/finefoods.txt: 
	curl http://snap.stanford.edu/data/finefoods.txt.gz  -O
	gunzip finefoods.txt.gz
	mv finefoods.txt data/reviews/amazon/food_reviews.txt

word_embeddings: data/word_embeddings/glove.6B.50d.txt 
data: data/reviews/amazon/finefoods.txt
