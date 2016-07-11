work/.folder_structure_sentinel: 
	mkdir -p data/reviews/amazon
	mkdir data/word_embeddings
	mkdir -p work/reviews/amazon

	touch $@

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

word_embeddings: data/word_embeddings/glove.6B.50d.txt 

###############################
# Amazon food reviews Dataset # 
###############################

data/reviews/amazon/food_reviews.txt: 
	curl http://snap.stanford.edu/data/finefoods.txt.gz  -O
	gunzip finefoods.txt.gz
	mv finefoods.txt $@

work/reviews/amazon/raw_food_reviews.csv: data/reviews/amazon/food_reviews.txt
	python review_analysis/data_setup/parse_amazon.py $< \
		work/reviews/amazon/

################
# Model inputs # 
################

work/reviews/amazon/filtered_ratios.npy: review_analysis/data_setup/parse_amazon.py

ratios: work/reviews/amazon/filtered_ratios.npy

work/embedding_weights.npy work/vec_reviews.npy: review_analysis/utils/preprocessing.py \
	work/reviews/amazon/filtered_ratios.npy \
	work/reviews/amazon/filtered_tokenized_reviews.pkl \
	data/word_embeddings/glove.6b.300d.txt
	python $< 300

embedding reviews: work/embedding_weights.npy work/vec_reviews.npy

data: folders data/reviews/amazon/food_reviews.txt \
	work/reviews/amazon/raw_food_reviews.csv word_embeddings
inputs: data ratios embedding reviews

all: folders data inputs
