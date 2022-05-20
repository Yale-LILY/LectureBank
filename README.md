# LILY LectureBank


This is the repo for LectureBank Corpus, with all batches and updates. We release meta data in this repo. 

Note that we also have a few works using part of the corpus, you can find more details in the [LB-Paper](https://github.com/Yale-LILY/LectureBank/tree/master/LB-Paper) folder.



## Meta Data

### data-versions
`lb*.tsv`: data with different versions. 


`ID, Instructor, Title, Topic, URL, Venue, Year`


- `ID`: Id of each line.
- `Instructor`: The author name(s).
- `Title`: File tile.
- `Topic`: The Topic Number, check `taxonomy.csv` for topic name. 
- `URL`: Online URL.
- `Year`: Year of the course.
- `Venue`: Name of the university, or `GitHub`.

We went through a URL check on May, 2022, here are the valid resource numbers: 
- 1020 lb1.tsv
- 308 lb2.tsv
- 3564 lb3.tsv
- 3136 lb4.tsv
- 1321 lb5.tsv

**NOTE**: we combined all five batches of LectureBank, and remove duplicates and invlaid urls. All data can be found in `alldata.tsv` with a total number to be **7104**. 


## Taxonomy

NLP taxonomy release. 
In the file `taxonomy.csv`, we include the taxonomy with 320 topics in a tree structure. The topic ID for each topic shows the parent node. For example, `233 (Relation Extraction)` has a parent node to be `23 (Part of Speech Tagging)`, and topic `23` has its parent node to be `2 (Language Modeling, Syntax, Parsing)`.


- `Topic ID`: Id of topic.
- `Topic`: topic name.

 You can find how this was created in our paper [CLICKER: A Computational LInguistics Classification Scheme for Educational Resources](https://arxiv.org/abs/2112.08578).

## Other resources

Please visit our website [AAN.how](https://aan.how/).
