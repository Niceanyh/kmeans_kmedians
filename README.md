# CA2 Data Clustering

This assignment requires you to implement the **k-means** and **k-medians** clustering algorithms using the Python programming language.


# Preview
- [Requirement](#Requirement)
- [Development](#Development)
- [Result](#Result)

# Requirement
[(Back to top)](#CA2 Data Clustering)
**Requirement:**
```python 3.8.8```
```numpy 1.20.1``` 
```matplotlib 3.3.4```

 install numpy and matplotlib:
```pip install numpy```
```pip install matplotlib```

**Developing environment :**

```MacOS 12.1``` 

To use this project, first clone the repo on your device using the command below:

```git init```

```git clone https://github.com/Niceanyh/kmeans_kmedians```


# Development
[(Back to top)](#table-of-contents)

**KMeans**

	KMeans.fit() - return a list of predicted labels
	KMeans.random_representatives() - return a random representative of the dataset

**KMedians**

	KMedians.fit() - return a list of predicted labels
	KMedians.random_representatives() - return a random representative of the dataset

**Gloable function**

	l1_distance(): - return l1 distance of two data
	l2_distance(): - return l2 distance of two data
	bcubed(): - return precision, recall, fscore of a model
	load_data() / make_data(): - data preprocessing
	show_plt(): -draw plot and save img
	l2_norm(): - l2 normalise dataset
	



# Result
[(Back to top)](#table-of-contents)

	KMeans K = 1

	Precision: 0.32871344537029246 
	Recall: 1.0 
	Fscore: 0.4723145424389065 
	
	---
	KMeans K = 2

	Precision: 0.6623558453999484 
	Recall: 1.0 
	Fscore: 0.7468488584696537 

	---
	KMeans K = 3

	Precision: 0.8096898340127133 
	Recall: 0.9821786354529158 
	Fscore: 0.8694560916928985 

	---
	KMeans K = 4

	Precision: 0.9047330039990591 
	Recall: 0.9061858323786992 
	Fscore: 0.9042511821824691 

	---
	KMeans K = 5

	Precision: 0.9235065657142034 
	Recall: 0.6826770101187937 
	Fscore: 0.761598867831124 

	---
	KMeans K = 6

	Precision: 0.8510848987913208 
	Recall: 0.5541639665095369 
	Fscore: 0.6206131459535091 

	---
	KMeans K = 7

	Precision: 0.9130244989733161 
	Recall: 0.513259124042173 
	Fscore: 0.6271314405454571 

	---
	KMeans K = 8

	Precision: 0.9180905740538766 
	Recall: 0.49306342896703687 
	Fscore: 0.5972273490491005 

	---
	KMeans K = 9

	Precision: 0.8946352118829183 
	Recall: 0.7202289287859347 
	Fscore: 0.7817403813592132 

	---
	KMeans Normalized K = 1

	Precision: 0.32871344537029246 
	Recall: 1.0 
	Fscore: 0.4723145424389065 

	---
	KMeans Normalized K = 2

	Precision: 0.6623558453999484 
	Recall: 1.0 
	Fscore: 0.7468488584696538 

	---
	KMeans Normalized K = 3

	Precision: 0.8181760352059984 
	Recall: 0.9939892439101551 
	Fscore: 0.8775968384431981 

	---
	KMeans Normalized K = 4

	Precision: 0.9340775365651872 
	Recall: 0.934725297901508 
	Fscore: 0.9339363088464115 

	---
	KMeans Normalized K = 5

	Precision: 0.9050895207243985 
	Recall: 0.8271138087574824 
	Fscore: 0.857548285880988 

	---
	KMeans Normalized K = 6

	Precision: 0.8725821375362659 
	Recall: 0.6524582007816536 
	Fscore: 0.7147209485707906 

	---
	KMeans Normalized K = 7

	Precision: 0.9341714875321775 
	Recall: 0.5530033148563333 
	Fscore: 0.6582442004401168 

	---
	KMeans Normalized K = 8

	Precision: 0.9266569003756477 
	Recall: 0.6127884681512454 
	Fscore: 0.7140937977428354 

	---
	KMeans Normalized K = 9

	Precision: 0.9314255916339574 
	Recall: 0.5547986164191823 
	Fscore: 0.6288825539825098 

	---
	Kmedians  K = 1

	Precision: 0.32871344537029246 
	Recall: 1.0 
	Fscore: 0.4723145424389065 

	---
	Kmedians  K = 2

	Precision: 0.6623558453999484 
	Recall: 1.0 
	Fscore: 0.7468488584696538 

	---
	Kmedians  K = 3

	Precision: 0.783748361730013 
	Recall: 0.9322409568479195 
	Fscore: 0.8298118090840616 

	---
	Kmedians  K = 4

	Precision: 0.9089039314855559 
	Recall: 0.9099820993828116 
	Fscore: 0.9085034576891343 

	---
	Kmedians  K = 5

	Precision: 0.9118568305832201 
	Recall: 0.6688176226434619 
	Fscore: 0.7485042462566517 

	---
	Kmedians  K = 6

	Precision: 0.7869578597118844 
	Recall: 0.610979569193123 
	Fscore: 0.582405739963939 

	---
	Kmedians  K = 7

	Precision: 0.9244408466750617 
	Recall: 0.6226749010815693 
	Fscore: 0.724464190671926 

	---
	Kmedians  K = 8

	Precision: 0.7672610328535084 
	Recall: 0.5963006439113339 
	Fscore: 0.5773751039463099 

	---
	Kmedians  K = 9

	Precision: 0.907913457292842 
	Recall: 0.6132065029084409 
	Fscore: 0.6921712147446147 

	---
	Kmedians Normalized  K = 1

	Precision: 0.32871344537029246 
	Recall: 1.0 
	Fscore: 0.4723145424389065 

	---
	Kmedians Normalized  K = 2

	Precision: 0.6623558453999484 
	Recall: 1.0 
	Fscore: 0.7468488584696538 

	---
	Kmedians Normalized  K = 3

	Precision: 0.7880659165062835 
	Recall: 0.9382431981912058 
	Fscore: 0.8355321733422348 

	---
	Kmedians Normalized  K = 4

	Precision: 0.9218726025084574 
	Recall: 0.9226363227298539 
	Fscore: 0.9218956519201491 

	---
	Kmedians Normalized  K = 5

	Precision: 0.9378980346870254 
	Recall: 0.8638996360273336 
	Fscore: 0.889613908146424 

	---
	Kmedians Normalized  K = 6

	Precision: 0.9272686916409763 
	Recall: 0.6495173121172046 
	Fscore: 0.7195783583132603 

	---
	Kmedians Normalized  K = 7

	Precision: 0.938405153365776 
	Recall: 0.6267049961257903 
	Fscore: 0.7244909498983041 

	---
	Kmedians Normalized  K = 8

	Precision: 0.9226233398695664 
	Recall: 0.5393554860839569 
	Fscore: 0.6364114287379039 

	---
	Kmedians Normalized  K = 9

	Precision: 0.9435486621671187 
	Recall: 0.5648684176915474 
	Fscore: 0.6683450363421451 

	---
