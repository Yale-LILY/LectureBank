
# LectureBank: a corpus for NLP Education and Prerequisite Chain Learning

This is the github page for our paper [*What Should I Learn First: Introducing LectureBank for NLP Education and Prerequisite Chain Learning*](https://arxiv.org/abs/1811.12181) in the proceedings AAAI 2019. 

Code for replicating the results will be uploaded soon. Stay tuned.


![An example of prerequisite relations from lecture slides depicted as a directed graph](https://github.com/Yale-LILY/LectureBank/blob/master/imgs/slide_diagram_final.png)

The list of descriptions can be found in the following:

### LectureBank Dataset
LectureBank Dataset is a manually-collected dataset of lecture slides. We collected 1352 online lecture files from 60 courses covering 5 different domains,  including Natural Language Processing (nlp), Machine Learning (ml), Artificial Intelligence (ai), Deep Learning (dl) and Information Retrieval (ir).  In addition, we release the corresponding annotations for each slide file to the taxonomy described below.  We also provide an additional vocabulary list of size 1221 extracted from the corpus. 

### LectureBank version 3 (updated 2021-06-14)

The newest release, lecturebank3.tsv, contains 5,000 entries. 

The format has changed slightly from the earlier versions.

`(ID, Title, URL, Topic_ID, Year, Instructor, Path, Venue)`

- `Instructor`: The instructors' name(s).
- `Path`: Path of the file

### lecturebank.tsv
Each line identifies a lecture file. Format:

`(ID, Title, URL, Topic_ID, Year, Author, Domain, Venue)`


- `ID`: Id of each line.
- `Title`: File tile.
- `URL`: Online URL.
- `Topic_ID`: Classified taxonomy Topic ID, referring topics from `taxonomy.tsv`.
- `Year`: Year of the course.
- `Author`: The author name(s).
- `Domain`: The domain (nlp, ir, dl, ml, ai).
- `Venue`: Name of the university, or `GitHub`.

### LectureBank1and2.tsv

This is a combined version of LectureBank1 and LectureBank2. We will release LectureBank3 and a cross-domain version of LectureBank! Stay tuned!

### download_all.py
The scripts of downloading the resources from the urls of `lecturebank.tsv`. After running the scripts, all the resources will be downloaded into `data_lecturebank/` folder (change the `base_path` if you want), organized by the `Domain` (for example, `nlp`, `ir`). 
The code is in python3, and you will need to install [`wget`](https://pypi.org/project/wget/) to run it.
Run with:
`python3 download_all.py`. It may take an hour or less for the resources to be downloaded.

Due to the change of the links by the owner, some of the URLs may have broken.

### taxonomy.tsv
Contains taxonomy topics and corresponding IDs referred by `lecturebank.tsv`.


### 208topics.csv
Contains the 208 topics which we annotated, format:

`(ID, Topic, Wiki_Page_URL)`

### prerequisite_annotation.csv
Contains the prerequisite chain annotation for each possible pair from the 208 topics. Format:

`(Source_Topic_ID, Target_Topic_ID, If_prerequisite)`


### vocabulary.txt
Contains 1221 vocabulary terms combined from taxonomy, 208 topics and terms extracted from LectureBank.



### LectureBank1and2.tsv
This is a combined version of LectureBank1 and LectureBank2. We will release LectureBank3 and a cross-domain version of LectureBank! Stay tuned!







